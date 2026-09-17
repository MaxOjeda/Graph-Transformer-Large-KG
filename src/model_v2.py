"""Graph Transformer con PODA A* y READOUT GLOBAL -- el modelo del paper (ACM WWW).

Version MINIMA de `src/model.py`: solo lo necesario para `--model sparse_state`, que es el
unico modelo que va a la publicacion. `model.py` conserva ademas el full attention, NBFNet,
las variantes sparse/expander y todos los ablations historicos (2 593 lineas); aca quedan
las ~770 que se usan.

QUE ES EL MODELO
----------------
Tres piezas, de abajo hacia arriba:

1. `PrunedSparseAttentionLayer` -- atencion relacional sobre una lista de aristas POR QUERY
   `(B, L)`, no sobre el grafo entero. Logit = `q.k/sqrt(d) + b[cabeza, rel]` y mensaje
   `alpha * (v_src (.) g[rel])` (composicion DistMult, el canal que razona por caminos).
   Con `prune_attn='sigmoid'` la agregacion NO se normaliza: bajo poda el segment-softmax
   escondia que se habia descartado evidencia (un nodo con 100 entrantes de las que se eligen
   5 quedaba "como si tuviera 5 vecinos"). NBFNet y A*Net SUMAN, por eso su poda no pierde.

2. `SparseStateGraphTransformer` -- la busqueda A* y el ESTADO DISPERSO. Por capa: top-K nodos
   por prioridad aprendida, top-L aristas salientes ordenadas por la prioridad del DESTINO
   (que es la heuristica de A*: estima la distancia RESTANTE), y atencion solo sobre esas L.
   El estado vive en SLOTS `(B, S, d)` en vez de `(B, N, d)`: a 2.5 M entidades eso es la
   diferencia entre 30 MB y 15.6 GB, y es lo que hace que ogbl-wikikg2 corra donde NBFNet
   da OOM.

3. `GlobalInducedBlock` -- el READOUT GLOBAL, y la unica pieza que no es message passing:
   M=32 tokens inductores por query hacen pool sobre los K nodos expandidos y se difunden a
   los S activos. Es lo que convierte el modelo en un Graph Transformer y no en una GAT
   relacional. Medido: +0.010 de MRR en FB15k-237 y +0.027 en YAGO3-10.

TODOS LOS FLAGS NUEVOS ARRANCAN EN CERO
---------------------------------------
`global_tokens`, `hop_pe`, `rel_frontier`, `fallback='ppr'` e `indicator='ppr'` tienen su
salida inicializada en 0, asi que con el flag activo el forward inicial es BIT A BIT el del
modelo sin el flag (`diag_global_equiv.py` lo verifica). La ablacion es limpia y un flag solo
puede sumar.

Referencias: `OBJETIVOS_Y_PLAN_WWW.md` (plan y analisis), `RESULTADOS_PAPER_WWW.md` (tablas),
`SESSION_NOTES.md` (bitacora). El kernel de busqueda esta en `src/sparse_state.py`.
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

try:
    from src.sparse_state import (CSRGraph, global_to_slot, global_to_slot_flat,
                                  variadic_topks, personalized_pagerank, ppr_bins)
except ImportError:
    from .sparse_state import (CSRGraph, global_to_slot, global_to_slot_flat,
                               variadic_topks, personalized_pagerank, ppr_bins)


# ============================================================================
# Helpers
# ============================================================================
def _make_relation_linear(hidden_dim, num_relation):
    """`Linear(d, R*d)` con init peso=0 / bias=1 => rel == 1 al arranque (ver doc arriba)."""
    lin = nn.Linear(hidden_dim, num_relation * hidden_dim)
    nn.init.zeros_(lin.weight)
    nn.init.ones_(lin.bias)
    return lin

def generate_expander_edges(num_nodes, degree, seed=0):
    """Grafo aleatorio d-regular simetrico (permutation algorithm de Exphormer,
    generate_random_regular_graph1). Devuelve (src, dst) long en CPU, sin self-loops;
    simetrico: si (x,y) esta, (y,x) tambien."""
    if num_nodes <= degree + 1:
        # grafo demasiado chico: conectar todos con todos (sin self-loops).
        idx = torch.arange(num_nodes)
        src = idx.repeat_interleave(num_nodes)
        dst = idx.repeat(num_nodes)
    else:
        g = torch.Generator().manual_seed(seed)
        base = torch.arange(num_nodes)
        senders = base.repeat(degree)                                  # [0..n-1]*degree
        receivers = torch.cat([base[torch.randperm(num_nodes, generator=g)]
                               for _ in range(degree)])
        src = torch.cat([senders, receivers])
        dst = torch.cat([receivers, senders])                          # simetrizar
    mask = src != dst                                                  # quitar self-loops
    return src[mask].contiguous(), dst[mask].contiguous()

# ============================================================================
# Capa de atencion relacional sobre aristas seleccionadas por query
# ============================================================================
class PrunedSparseAttentionLayer(nn.Module):
    """Capa de atencion sparse sobre aristas SELECCIONADAS POR QUERY.

    Igual que `SparseRelationalAttentionLayer` salvo que `edges` viene con forma (B, L) en
    vez de (E,): cada query tiene su propia lista de aristas. Eso obliga a cambiar todos los
    lookups de `x[:, idx]` a `gather`, y los `index_add_` a `scatter_add_`, porque los
    indices dejan de ser compartidos entre elementos del batch.
    """

    def __init__(self, hidden_dim, num_heads, num_relation, drop, dependent=False,
                 prune_attn='softmax'):
        super().__init__()
        assert hidden_dim % num_heads == 0
        assert prune_attn in ('softmax', 'sigmoid')
        # 'sigmoid': agregacion SIN NORMALIZAR. Hipotesis (2026-08-25): bajo poda, el
        # segment-softmax normaliza sobre las aristas SELECCIONADAS, asi que un nodo con 100
        # entrantes de las que se eligen 5 queda "como si tuviera 5 vecinos" -- la
        # normalizacion ESCONDE que se descarto evidencia. NBFNet/A*Net suman, asi que
        # descartar aristas reduce la suma y el modelo puede notarlo; por eso su poda es sin
        # perdida y la nuestra no. Sin poda esto se probo (--attn sigmoid, 2026-07-22) y fue
        # neutro; bajo poda el problema es cualitativamente distinto.
        self.prune_attn = prune_attn
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.self_rel = num_relation
        R = num_relation + 1
        self.num_rel_table = R
        self.dependent = dependent

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        self.rel_bias = nn.Parameter(torch.zeros(num_heads, R))
        if dependent:
            self.rel_value = None
            self.relation_linear = _make_relation_linear(hidden_dim, R)
        else:
            self.rel_value = nn.Parameter(torch.ones(num_heads, R, self.head_dim))

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, edges, q_emb=None, gate=None, rel_bias_q=None):
        """x: (B, N, D). edges: (src, rel, dst), cada uno (B, L).
        gate: (B, L) opcional -- prioridad del nodo FUENTE de cada arista, en [0,1]. Pesa el
        MENSAJE, no el estado persistente. Es la Ec. 12 de A*Net
        (`h <- + s(x)*(h (x) w)`): el score del emisor modula lo que emite.
        ⚠️ NO aplicarlo como `x = sigmoid(s) * x` sobre el stream residual: con sigmoid ~0.5
        el estado queda escalado 0.5^L y se desvanece en 6 capas. A*Net no lo sufre porque
        tiene `short_cut`; nuestro residual pre-LN no lo restaura.
        rel_bias_q: (B, L) opcional -- compatibilidad aprendida entre la relacion de la arista y
        la relacion de la QUERY (`--rel_frontier`). Se suma al logit, compartido entre cabezas.
        Es el mismo termino que ordena la frontera, asi que recibe gradiente por aqui (weight
        sharing, como la prioridad de A*Net). Init cero => no-op."""
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        src, rel, dst = edges
        L = src.size(1)

        h = self.norm1(x)
        q = self.to_q(h).view(B, N, H, hd)
        k = self.to_k(h).view(B, N, H, hd)
        v = self.to_v(h).view(B, N, H, hd)

        # Gather por query: (B,L) -> (B,L,H,hd). Es el reemplazo de `q[:, dst]`.
        idx4 = lambda i: i.view(B, L, 1, 1).expand(B, L, H, hd)
        q_e = q.gather(1, idx4(dst))
        k_e = k.gather(1, idx4(src))
        v_e = v.gather(1, idx4(src))

        # Bias relacional por arista: rel_bias es (H,R) -> (R,H) -> gather con rel (B,L).
        # ⚠️ `index_select`, NO `tabla[rel]`: la indexacion avanzada tiene un backward por
        # atomicos que cuesta hasta 2x (regla del proyecto, 2026-08-09, medida con profiler).
        # La escribi mal en esta capa nueva; es bit a bit identico.
        rel_flat = rel.reshape(-1)
        bias_e = self.rel_bias.t().contiguous().index_select(0, rel_flat).view(B, L, H)
        logit = (q_e * k_e).sum(-1) * self.scale + bias_e     # (B, L, H)
        if rel_bias_q is not None:
            logit = logit + rel_bias_q.unsqueeze(-1)

        # Segment-softmax por nodo destino. `index_add_` no sirve: los indices cambian por
        # elemento del batch => scatter_reduce_/scatter_add_ sobre la dimension de nodo.
        idx3 = dst.view(B, L, 1).expand(B, L, H)
        if self.prune_attn == 'sigmoid':
            # Sin normalizar: el peso total por nodo CRECE con la evidencia disponible.
            alpha = self.drop(torch.sigmoid(logit))
        else:
            node_max = logit.new_full((B, N, H), float('-inf'))
            node_max.scatter_reduce_(1, idx3, logit, reduce='amax', include_self=False)
            logit = (logit - node_max.gather(1, idx3)).exp()
            denom = logit.new_zeros(B, N, H)
            denom.scatter_add_(1, idx3, logit)
            alpha = self.drop(logit / (denom.gather(1, idx3) + 1e-9))

        # Modulacion relacional del valor (DistMult).
        if self.dependent:
            rel_emb = self.relation_linear(q_emb).view(B, self.num_rel_table, H, hd)
            g_e = rel_emb.gather(1, idx4(rel))                # (B,L,H,hd)
        else:
            g_e = self.rel_value.permute(1, 0, 2).contiguous() \
                      .index_select(0, rel_flat).view(B, L, H, hd)
        msg = alpha.unsqueeze(-1) * (v_e * g_e)
        if gate is not None:
            msg = msg * gate.view(B, L, 1, 1)

        out = msg.new_zeros(B, N, H, hd)
        out.scatter_add_(1, idx4(dst), msg)
        out = out.reshape(B, N, D)
        x = x + self.drop(self.to_out(out))
        x = x + self.ffn(self.norm2(x))
        return x

# ============================================================================
# Readout global especifico por relacion (control del bloque global, M=2 fijo)
# ============================================================================
class RelationalGlobalReadout(nn.Module):
    """Readout global ESPECIFICO POR RELACION, condicionado a la relacion de la query.

    Para cada query (h, q, ?) agrega los estados de los nodos que tienen alguna arista de
    tipo q -- por separado los que la tienen ENTRANTE y los que la tienen SALIENTE -- y
    proyecta esa evidencia global a un termino aditivo del score por nodo.

    Es informacion que `rawl2` no puede ver: no depende del camino de h a v, sino de una
    propiedad GLOBAL del grafo relativa a la relacion consultada.

    Costo: O(B*N*d) en tiempo y memoria; los indices por relacion se precomputan UNA vez por
    grafo (no por batch) y se cachean.
    """

    def __init__(self, hidden_dim, num_relation):
        super().__init__()
        self.num_relation = num_relation
        self.proj_in = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)
        # Init CERO => al arranque el modelo es EXACTAMENTE el baseline sin readout, y solo
        # diverge al entrenar. Misma convencion de control que --dependent y lowrank.
        self.to_score = nn.Linear(hidden_dim * 2, 1)
        nn.init.zeros_(self.to_score.weight)
        nn.init.zeros_(self.to_score.bias)
        self._cache = {}

    def _masks(self, graph, device):
        """(R, N) booleanas: que nodos tienen arista SALIENTE / ENTRANTE de cada relacion."""
        key = (id(graph), graph.num_nodes)
        if key not in self._cache:
            ei = graph.edge_index.to(device)
            h, r, t = ei[:, 0], ei[:, 1], ei[:, 2]
            R, N = self.num_relation, graph.num_nodes
            out_m = torch.zeros(R, N, dtype=torch.bool, device=device)
            in_m = torch.zeros(R, N, dtype=torch.bool, device=device)
            out_m[r, h] = True
            in_m[r, t] = True
            self._cache[key] = (out_m, in_m)
        return self._cache[key]

    def forward(self, x, graph, r_index):
        """x: (B,N,D) -> termino aditivo del score, (B,N)."""
        B, N, D = x.shape
        out_m, in_m = self._masks(graph, x.device)
        # Media de los estados de los nodos con arista saliente/entrante de tipo r_q.
        mo = out_m[r_index].float()                       # (B,N)
        mi = in_m[r_index].float()
        go = (mo.unsqueeze(-1) * x).sum(1) / mo.sum(1, keepdim=True).clamp(min=1)   # (B,D)
        gi = (mi.unsqueeze(-1) * x).sum(1) / mi.sum(1, keepdim=True).clamp(min=1)
        g = torch.cat([self.proj_in(gi), self.proj_out(go)], dim=-1)                # (B,2D)
        return self.to_score(g)                                                     # (B,1)

# ============================================================================
# BLOQUE GLOBAL: tokens inductores por query (Set Transformer / Perceiver)
# ============================================================================
class GlobalInducedBlock(nn.Module):
    """Canal GLOBAL del GT podado: M tokens inductores por query (Set Transformer / Perceiver).

    Por que existe (OBJETIVOS_Y_PLAN_WWW.md §3.2, §3.6): la atencion con soporte = adyacencia es un
    C-MPNN acotado por rawl2, la MISMA clase que NBFNet y A*Net (medido: los tres fallan en las
    mismas queries, Jaccard 0.79-0.84). Lo unico que sale de la clase sin embeddings de entidad
    es un READOUT GLOBAL condicionado a la query (Huang et al. 2023, Thm 5.3: C-MPNN + READ
    captura erFO3_cnt). Y su evidencia dice que el readout por SUMA empeora y el especifico por
    relacion mejora => tiene que ser SELECTIVO, o sea atencion.

    Mecanica, dos atenciones cruzadas por capa:
      (1) POOL:  M tokens (aprendidos + proyeccion de q_emb) atienden sobre un POOL de nodos
                 activos -- por defecto los K nodos que la busqueda expandio en esta capa
                 (`global_pool='topk'`), o todos los S activos (`'active'`).
      (2) BCAST: cada nodo activo atiende sobre los M tokens y suma el resultado al residual.
    Costo O(B*M*(P + S)) con P = K o S: lineal en el conjunto podado e INDEPENDIENTE de las
    aristas, asi que escala igual que la poda (en wikikg2 P = K = 5 001).

    ⚠️ La proyeccion de salida `out` se inicializa en CERO => al arrancar el bloque es un no-op
    EXACTO (x + 0) y el modelo es bit a bit el de antes. Convencion del proyecto (`--dependent`,
    `RelationalGlobalReadout`): la ablacion es limpia y el flag solo puede sumar.

    Prediccion falsable, escrita antes de medir (§3.6): debe mover el estrato de cardinalidad
    4-30 y las queries donde los modelos discrepan; NO debe mover el estrato 101+.
    """

    def __init__(self, hidden_dim, num_heads, num_tokens, drop=0.0):
        super().__init__()
        self.M = num_tokens
        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.tok_q = nn.Linear(hidden_dim, hidden_dim)
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.pool = nn.MultiheadAttention(hidden_dim, num_heads, dropout=drop, batch_first=True)
        self.bcast = nn.MultiheadAttention(hidden_dim, num_heads, dropout=drop, batch_first=True)
        self.ffn_t = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, valid, q_emb, pool_idx=None, pool_ok=None):
        """x: (B,S,D) estados por slot. valid: (B,S) bool. q_emb: (B,D).
        pool_idx/pool_ok: (B,P) slots del pool y su validez; None => pool = todos los activos."""
        B, S, D = x.shape
        xs = self.norm_x(x)
        if pool_idx is not None:
            src = xs.gather(1, pool_idx.unsqueeze(-1).expand(-1, -1, D))
            ok = pool_ok
        else:
            src, ok = xs, valid
        # ⚠️ nn.MultiheadAttention devuelve NaN si una fila queda TODA enmascarada; el pool
        # siempre contiene al head (prioridad 1), pero se garantiza >= 1 clave valida igual.
        kpm = ~ok
        kpm = kpm & ~(kpm.all(dim=1, keepdim=True) & (torch.arange(kpm.size(1), device=x.device) == 0))
        T = self.tokens.unsqueeze(0) + self.tok_q(q_emb).unsqueeze(1)          # (B,M,D)
        T = T + self.pool(T, src, src, key_padding_mask=kpm, need_weights=False)[0]
        T = T + self.ffn_t(self.norm_t(T))
        g = self.bcast(xs, T, T, need_weights=False)[0]                          # (B,S,D)
        return x + self.out(g) * valid.unsqueeze(-1).to(x.dtype)

# ============================================================================
# El modelo: busqueda A* + estado disperso + bloque global
# ============================================================================
def _survivors(active, new_active, SENT):
    """Cuantos ids del activo VIEJO siguen en el nuevo (por fila). Solo para diagnostico."""
    na, _ = new_active.sort(dim=1)
    slot = torch.searchsorted(na, active).clamp(max=na.size(1) - 1)
    return ((na.gather(1, slot) == active) & (active != SENT)).sum(1)

class SparseStateGraphTransformer(nn.Module):
    """GT con poda de ARISTAS **y de ESTADO**: la ruta a ogbl-wikikg2.

    Diferencia con `PrunedSparseGraphTransformer`: alli el estado es `(B, N, d)` denso, que a
    2.5 M nodos son 2.6 GB por capa (15.6 GB retenidos por 6 capas) y hace la corrida
    inviable -- medido en YAGO3-10, donde 8.5x de nodos costo 10.9x de tiempo y 21.8 GB.

    Aca los nodos activos de cada query viven en SLOTS de tamano fijo:
        active (B,S) ids globales ORDENADOS   |   hidden (B,S,d)
    Con S=30 000 en wikikg2 son 30 MB en vez de 15.6 GB.

    ⚠️ INVARIANTE: `active` se mantiene ORDENADO por fila, porque el mapeo global->slot va con
    `searchsorted`. La alternativa (un tensor (B,N) de posiciones) es exactamente el objeto que
    hace inviable wikikg2. Los slots libres se rellenan con el centinela `num_nodes`.

    La capa de atencion se reutiliza SIN CAMBIOS (`PrunedSparseAttentionLayer`): opera con
    listas de aristas (B,L) y estados (B,*,d) via gather/scatter sobre la dim 1, asi que no
    distingue si esa dim son nodos globales o slots.
    """

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 node_slots=4096, top_nodes=1024, edge_budget=8192, edge_cap=64,
                 dependent=False, prune_attn='sigmoid', rel_readout=False, evict='id',
                 grad_ckpt=False, test_top_nodes=0, test_edge_budget=0,
                 indicator='onehot', num_indicator_bin=10, edge_dropout=0.0,
                 break_tie=False, exp_degree=0,
                 global_tokens=0, global_from=2, global_pool='topk',
                 hop_pe=False, rel_frontier=0.0, fallback='scalar'):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        # ---------------- semana 2 del plan (OBJETIVOS_Y_PLAN_WWW.md §4) ----------------
        # Todo con INIT CERO: con los flags activos el forward al arrancar es bit a bit el
        # mismo que sin ellos (verificado por diag_global_equiv.py). Cada flag solo puede sumar.
        R_table = num_relation + 1 + (1 if exp_degree else 0)   # reales + self-loop (+ R_exp)
        # (a) Canal GLOBAL: M tokens inductores por query desde la capa `global_from`
        #     (las capas 0-1 son degeneradas: casi todo el activo esta en estado de borde).
        self.global_from = global_from
        self.global_pool = global_pool
        self.global_blocks = nn.ModuleList([
            GlobalInducedBlock(hidden_dim, num_heads, global_tokens, drop)
            if (global_tokens and l >= global_from) else nn.Identity()
            for l in range(num_layer)])
        # (b) PE RELATIVA de SALTO: el nodo recibe, al ACTIVARSE, el embedding de la capa en la
        #     que entro (= su distancia en saltos al head por el camino que lo alcanzo). Gratis:
        #     no hay que seguir ningun tensor, es el indice de capa en el merge. Sin identidad
        #     de entidad. Init cero.
        self.hop_pe = hop_pe
        if hop_pe:
            self.hop_emb = nn.Embedding(num_layer + 1, hidden_dim)
            nn.init.zeros_(self.hop_emb.weight)
        # (c) FRONTERA CONSCIENTE DE LA RELACION: la prioridad de A*Net no ve la relacion de la
        #     arista (`score` es f(hidden, query), model.py:322), asi que en la capa 0 elige la
        #     primera frontera a ciegas (por eso necesitan `break_tie`). Aqui se suma
        #     lambda * <q_emb, rel_compat[r]> a la prioridad del destino, y el MISMO termino entra
        #     al logit de la atencion para que reciba gradiente (weight sharing). Init cero.
        self.rel_frontier = rel_frontier
        if rel_frontier:
            self.rel_compat = nn.Embedding(R_table, hidden_dim)
            nn.init.zeros_(self.rel_compat.weight)
        # (d) FALLBACK INFORMADO para los nodos que la busqueda no visita: en WN18RR el 12 % de
        #     los fallos comunes recibe el score escalar de fallback (rank = N) y ahi se pierde
        #     el MR (§3.4.5). 'ppr' suma un termino aprendido por bin de PageRank personalizado
        #     desde el head: informacion de POSICION global, sin identidad de entidad. Init cero.
        self.fallback = fallback
        if fallback == 'ppr':
            self.fallback_bin = nn.Embedding(num_indicator_bin, 1)
            nn.init.zeros_(self.fallback_bin.weight)
        self.self_rel = num_relation          # relacion reservada del self-loop
        self.exp_rel = num_relation + 1       # relacion reservada de las aristas EXPANDER
        self.exp_degree = exp_degree          # 0 = sin expander
        self._exp_cache = {}
        self.evict = evict         # 'id' (historico, arbitrario) | 'prio' (top-S por prioridad)
        self.grad_ckpt = grad_ckpt  # recomputa activaciones en el backward (ver forward)
        self.test_K = test_top_nodes    # 0 = mismo presupuesto que en train
        self.test_L = test_edge_budget
        # --- opciones que A*Net activa SOLO en wikikg2 (wikikg2_astarnet.yaml) ---
        self.indicator = indicator          # 'onehot' | 'ppr'
        self.num_indicator_bin = num_indicator_bin
        self.edge_dropout = edge_dropout
        self.break_tie = break_tie
        if indicator == 'ppr':
            # Condicion de borde con DISTANCIA. Su paper: "instead of using a boundary
            # condition of mostly zeros, we find it is better to incorporate distance
            # information", 1_q(u=v) = 1(u=v)q + 1(u!=v) p_{u,v}, con p_{u,v} un embedding
            # sobre el PPR DISCRETIZADO. Importa a escala: con alpha=0.2 % en wikikg2 la
            # busqueda alcanza <= 7.7 % de N y con boundary cero el 92 % restante llega al
            # readout SIN NINGUNA informacion.
            self.distance = nn.Embedding(num_indicator_bin, hidden_dim)
            # ⚠️ INIT EN CERO, NO el default. Con el init por defecto N(0,1) cada fila tiene
            # norma ~5.2, asi que TODO nodo no-fuente arranca con un vector de esa magnitud y
            # el contraste del labeling trick ("fuente = q_emb, resto = CERO") pasa de infinito
            # a ~1: la senal queda ahogada por el boundary. Medido: valid 0.404 -> 0.121 en
            # FB15k-237 y 0.532 -> 0.395 en WN18RR (jobs 93072/93071).
            # Con init 0 el forward es IDENTICO al de onehot al arrancar, asi que 'ppr' es una
            # generalizacion ESTRICTA: el modelo puede aprender a usar la distancia, o dejarla
            # en cero y no perder nada. Mismo truco que `_make_relation_linear` (w=0, b=1).
            nn.init.zeros_(self.distance.weight)
        self.S = node_slots        # cupo de slots de estado
        self.K = top_nodes         # nodos que se expanden por capa
        self.L = edge_budget       # aristas que se propagan por capa
        self.cap = edge_cap        # tope de salientes POR NODO (evita que un hub coma el cupo)
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        self.layers = nn.ModuleList([
            # ⚠️ La capa dimensiona sus tablas a num_relation+1 (self-loop). Con expander
            # hace falta una fila MAS para R_exp, asi que se le pasa num_relation+1 y queda
            # R = num_relation+2: [0..R-1] reales, R=self_rel, R+1=exp_rel.
            PrunedSparseAttentionLayer(hidden_dim, num_heads,
                                       num_relation + (1 if exp_degree else 0), drop,
                                       dependent=dependent, prune_attn=prune_attn)
            for _ in range(num_layer)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.prio_g = nn.Linear(hidden_dim * 2, hidden_dim)
        self.unvisited_score = nn.Parameter(torch.zeros(1))
        self.rel_readout = RelationalGlobalReadout(hidden_dim, num_relation) \
            if rel_readout else None

    def _expander(self, num_nodes, device):
        """Grafo aleatorio d-regular simetrico, cacheado por tamano (es fijo por grafo)."""
        if num_nodes not in self._exp_cache:
            self._exp_cache[num_nodes] = generate_expander_edges(
                num_nodes, self.exp_degree, seed=0)
        src, dst = self._exp_cache[num_nodes]
        return src.to(device), dst.to(device)

    def _boundary(self, nodes, bins, B, S, N, dev):
        """Condicion de borde de los slots. `nodes` (B,S) ids globales (SENT = no activo).

        'onehot' = cero salvo la fuente (labeling trick de NBFNet).
        'ppr'    = embedding del BIN de PageRank personalizado (lo que usa A*Net en wikikg2).
        """
        if self.indicator != 'ppr':
            return torch.zeros(B, S, self.hidden_dim, device=dev)
        return self.distance(bins.gather(1, nodes.clamp(max=N - 1)).long())

    def _priority(self, x, q_emb):
        """Prioridad por slot. `prio_g` es Linear(2D -> D) sobre [x, q].

        ⚠️ NO materializar `cat([x, q.expand(M)])`: es (B, M, 2D) y en wikikg2 (M=S=386 604)
        son 792 MB POR CAPA, retenidos para el backward -- ~9-12 GB de los 23 medidos, y el
        punto exacto donde reventaba el smoke test (job 92652).
        Un Linear sobre una concatenacion se parte exactamente en dos:
            W [x;q] + b  =  x W_x^T + q W_q^T + b
        y como q_emb es (B, D) su termino es (B, 1, D) que se DIFUNDE sobre M, sin expandir.
        Matematicamente identico; solo cambia el orden de las sumas (fp).
        """
        W = self.prio_g.weight                       # (D, 2D)
        d = x.size(-1)
        g = torch.nn.functional.linear(x, W[:, :d], self.prio_g.bias)
        g = g + torch.nn.functional.linear(q_emb, W[:, d:]).unsqueeze(1)
        return torch.sigmoid(self.readout(self.norm_out(x * g)).squeeze(-1))

    def _select_dense(self, csr, k_nodes, k_ok, keep_csr, active, prio, base,
                      B, K, L, N, cap, dev, comp=None):
        """Ruta ORIGINAL: pool RECTANGULAR (B, K*cap) con tope `cap` por nodo.

        ⚠️ Toma las PRIMERAS `cap` aristas de la fila CSR, o sea en el orden en que venian en
        el archivo del dataset -- criterio arbitrario, no de relevancia. Se conserva para
        reproducir lo ya medido y para el test de equivalencia; la ruta nueva es `cap <= 0`.
        """
        # ⚠️ `k_nodes` puede traer el centinela SENT=N cuando hay menos de K activos, y
        # `csr.degree` solo tiene N entradas => indexar antes de enmascarar dispara un
        # device-side assert. Se clampa y se anula despues con `k_ok`.
        flat = k_nodes.reshape(-1).clamp(max=N - 1)               # (B*K,)
        deg = csr.degree[flat].clamp(max=cap)                     # (B*K,)
        start = csr.indptr[flat]                                  # (B*K,)
        off = torch.arange(cap, device=dev).view(1, cap)
        eidx = (start.view(-1, 1) + off).clamp(max=csr.src.numel() - 1)   # (B*K,cap)
        e_ok = (off < deg.view(-1, 1)) & k_ok.reshape(-1, 1)
        eidx = eidx.view(B, K * cap)
        e_ok = e_ok.view(B, K * cap)
        if keep_csr is not None:
            e_ok = e_ok & keep_csr[eidx]     # aristas de la query, fuera

        e_src, e_dst, e_rel = csr.src[eidx], csr.dst[eidx], csr.rel[eidx]

        # ordenar por prioridad del DESTINO (heuristica de A*, §3.2). Los destinos todavia
        # no activos usan la prioridad del estado cero: un escalar por query.
        d_slot, d_found = global_to_slot(active, e_dst)
        d_prio = torch.where(d_found, prio.gather(1, d_slot), base.expand(-1, K * cap))
        if comp is not None:
            d_prio = d_prio + self.rel_frontier * comp.gather(1, e_rel)
        d_prio = d_prio.masked_fill(~e_ok, float('-inf'))
        sel = d_prio.topk(min(L, K * cap), dim=1).indices
        return (e_src.gather(1, sel), e_dst.gather(1, sel),
                e_rel.gather(1, sel), e_ok.gather(1, sel))

    def forward(self, batched_data):
        graph = batched_data['graph']
        h_index, r_index = batched_data['h_index'], batched_data['r_index']
        N, B = graph.num_nodes, h_index.size(0)
        dev = h_index.device
        S, K, L, cap = self.S, self.K, self.L, self.cap
        if not self.training and self.test_K:
            # PRESUPUESTO DE BUSQUEDA EN EVALUACION, distinto del de entrenamiento. Es lo que
            # hace A*Net en wikikg2 (`wikikg2_astarnet.yaml`): node_ratio 0.002 al entrenar y
            # test_node_ratio 0.01 al evaluar, o sea 5x MAS nodos expandidos en test.
            # ⚠️ En FB15k-237 / WN18RR / YAGO sus dos ratios COINCIDEN (test_node_ratio no
            # aparece en esas configs), por eso esta asimetria solo importa en wikikg2.
            # `L` escala con K para mantener beta constante (su es = beta*ks*E/N).
            K = self.test_K
            L = self.test_L or int(self.L * self.test_K / self.K)
            S = max(S, 6 * L)      # la cota del alcance crece con L; sin esto habria desalojo

        # CSR sobre el grafo COMPLETO (una sola vez, cacheado por grafo). La mascara de la
        # query NO se aplica aca: se aplica al recolectar candidatas, via `keep_csr`.
        ei_full = graph.edge_index.to(dev)
        loop = torch.arange(N, device=dev)
        ei_full = torch.cat([ei_full,
                             torch.stack([loop, torch.full_like(loop, self.self_rel), loop], 1)])
        n_exp = 0
        if self.exp_degree:
            # ARISTAS EXPANDER (grafo aleatorio d-regular simetrico, estilo Exphormer). Se
            # agregan al grafo ANTES de construir el CSR, asi que toda la maquinaria de
            # busqueda (top-K, top-L, atencion, slots) las trata como aristas mas -- solo que
            # con una relacion reservada propia R_exp, nunca una relacion real del KG.
            # ⚠️ El sobrecosto del expander era 1.76x hasta que se encontro que la causa no
            # era el expander sino `tabla[:, rel]` en el lookup relacional (2026-08-09); con
            # `index_select` bajo a 1.08x. `sparse_state` ya usa index_select.
            e_src, e_dst = self._expander(N, dev)
            n_exp = e_src.numel()
            ei_full = torch.cat([ei_full, torch.stack(
                [e_src, torch.full_like(e_src, self.exp_rel), e_dst], 1)])
        csr = CSRGraph.get(ei_full, N, dev, key=id(graph))
        gm = batched_data.get('graph_mask')
        if gm is not None:
            # Las self-loops y las expander NUNCA se enmascaran: `graph_mask` solo habla de
            # aristas REALES del KG (una arista expander no es un hecho, no puede filtrarse).
            keep = torch.cat([gm.to(dev),
                              torch.ones(N + n_exp, dtype=torch.bool, device=dev)])
            keep_csr = keep[csr.order]
        else:
            keep_csr = None

        q_emb = self.query_emb(r_index)
        ar = torch.arange(B, device=dev)
        bins = None
        if self.indicator == 'ppr' or self.fallback == 'ppr':
            # (B, N) uint8: el UNICO denso sobre N, una vez por batch (no por capa).
            ppr = personalized_pagerank(ei_full, N, h_index)
            bins = ppr_bins(ppr, N, self.num_indicator_bin)
            del ppr
        # compatibilidad relacion-de-arista x relacion-de-query, (B, R_table), una vez por batch
        comp = (q_emb @ self.rel_compat.weight.t()) if self.rel_frontier else None
        SENT = N                                     # centinela: mayor que cualquier id
        active = torch.full((B, S), SENT, dtype=torch.long, device=dev)
        if getattr(self, '_all_active', False):
            # Solo para el TEST DE EQUIVALENCIA: arrancar con TODOS los nodos activos, para
            # que el FFN/residual se aplique al mismo conjunto que en el modelo denso. Sin
            # esto el test esta MAL PLANTEADO: el denso aplica el FFN a los N nodos desde la
            # capa 0, asi que un nodo que se activa en la capa 2 ya llega con deriva
            # acumulada, y la diferencia aparece tambien en los activos.
            assert S >= N, 'el test all_active necesita S >= N'
            active[:, :N] = torch.arange(N, device=dev).unsqueeze(0)
        active[:, 0] = h_index if not getattr(self, '_all_active', False) else active[:, 0]
        active, _ = active.sort(dim=1)               # invariante: ORDENADO
        hidden = self._boundary(active, bins, B, S, N, dev)
        slot_h, _ = global_to_slot(active, h_index.unsqueeze(1))
        # la fuente SIEMPRE lleva la query (+ PE de salto 0 si esta activa)
        hidden[ar, slot_h.squeeze(1)] = q_emb + (self.hop_emb.weight[0] if self.hop_pe else 0.0)
        prio = torch.zeros(B, S, device=dev)
        prio[ar, slot_h.squeeze(1)] = 1.0

        for li, layer in enumerate(self.layers):
            valid = active != SENT
            # (1) top-K nodos activos por prioridad
            p = prio.masked_fill(~valid, float('-inf'))
            k_slot = p.topk(min(K, S), dim=1).indices                 # (B,K)
            k_nodes = active.gather(1, k_slot)                        # (B,K) ids globales
            k_ok = valid.gather(1, k_slot)

            base = self._priority(torch.zeros(B, 1, self.hidden_dim, device=dev), q_emb)

            if cap <= 0:
                # ---------- (2') RUTA SIN PADDING (Alg. 2 de A*Net) ----------
                # Se toman TODAS las salientes de los K nodos, sin tope por nodo. El pool es
                # un arreglo PLANO (total,) con su query id por elemento, y el recorte a L es
                # un unico top-k por muestra via `variadic_topks`. Ver src/sparse_state.py.
                flat = k_nodes.reshape(-1).clamp(max=N - 1)              # (B*K,)
                deg = csr.degree[flat] * k_ok.reshape(-1)                # 0 para invalidos
                total = int(deg.sum())
                owner = torch.repeat_interleave(torch.arange(B * K, device=dev), deg)
                offs = torch.cumsum(deg, 0) - deg
                eidx = csr.indptr[flat][owner] + (torch.arange(total, device=dev) - offs[owner])
                qid = owner.div(K, rounding_mode='floor')                # (total,)
                if keep_csr is not None:
                    m = keep_csr[eidx]
                    eidx, qid = eidx[m], qid[m]
                if self.edge_dropout and self.training:
                    # EDGE DROPOUT estructural (`edge_dropout: 0.2` en wikikg2). Ellos lo
                    # aplican como Dropout sobre `edge_weight` (model.py:97-99), que con
                    # p=0.2 pone el 20 % de los pesos en 0 -- o sea ELIMINA esas aristas del
                    # message passing. Aca no hay `edge_weight`, asi que el equivalente es
                    # sacarlas del pool de candidatas antes del top-L.
                    # ⚠️ Solo en train, como ellos (su Dropout es no-op en eval).
                    m = torch.rand(eidx.shape, device=dev) >= self.edge_dropout
                    eidx, qid = eidx[m], qid[m]
                e_src, e_dst, e_rel = csr.src[eidx], csr.dst[eidx], csr.rel[eidx]

                # (3') prioridad del DESTINO, igual que en la ruta densa
                d_slot, d_found = global_to_slot_flat(active, e_dst, qid, N)
                d_prio = torch.where(d_found, prio.reshape(-1)[qid * S + d_slot],
                                     base.reshape(-1)[qid])
                if comp is not None:
                    # frontera consciente de la relacion: + lambda * compat(r_arista, r_query)
                    d_prio = d_prio + self.rel_frontier * comp[qid, e_rel]
                size = torch.bincount(qid, minlength=B)
                ks = size.clamp(max=L)
                _, sel = variadic_topks(d_prio, size, ks, break_tie=self.break_tie)

                # de vuelta a (B,L) rectangular: el padding queda solo en la SALIDA, que es
                # B*L y no B*K*cap. `s_ok` marca las columnas realmente usadas.
                nsel = int(ks.sum())
                sel_q = torch.repeat_interleave(torch.arange(B, device=dev), ks)
                pos_q = torch.arange(nsel, device=dev) - (ks.cumsum(0) - ks).repeat_interleave(ks)
                s_src = torch.zeros(B, L, dtype=torch.long, device=dev)
                s_dst = torch.zeros(B, L, dtype=torch.long, device=dev)
                s_rel = torch.full((B, L), self.self_rel, dtype=torch.long, device=dev)
                s_ok = torch.zeros(B, L, dtype=torch.bool, device=dev)
                s_src[sel_q, pos_q] = e_src[sel]
                s_dst[sel_q, pos_q] = e_dst[sel]
                s_rel[sel_q, pos_q] = e_rel[sel]
                s_ok[sel_q, pos_q] = True
            else:
                s_src, s_dst, s_rel, s_ok = self._select_dense(
                    csr, k_nodes, k_ok, keep_csr, active, prio, base, B, K, L, N, cap, dev,
                    comp=comp)

            # (4) incorporar destinos nuevos al conjunto activo (manteniendolo ORDENADO)
            merged = torch.cat([active, torch.where(s_ok, s_dst, torch.full_like(s_dst, SENT))], 1)
            merged, _ = merged.sort(dim=1)
            dup = torch.zeros_like(merged, dtype=torch.bool)
            dup[:, 1:] = merged[:, 1:] == merged[:, :-1]
            merged = merged.masked_fill(dup, SENT)
            merged, _ = merged.sort(dim=1)
            # ⚠️ DESALOJO. Cuando el activo desborda S, hay que tirar nodos. `merged` esta
            # ordenado por ID GLOBAL, asi que `merged[:, :S]` se queda con los S ids MAS
            # CHICOS: un criterio ARBITRARIO, dictado por como estan numeradas las entidades
            # y no por su relevancia. En YAGO eso deja el 34.6 % de las respuestas sin score
            # (ver SESSION_NOTES 2026-08-27). A*Net no desaloja NUNCA: su `graph.score` es un
            # VirtualTensor sobre los N nodos que solo ACUMULA claves (model.py:322), y su
            # `node_ratio` limita quien SE EXPANDE, no quien recibe score.
            if self.evict == 'keep' and merged.size(1) > S:
                # DESALOJO 'keep': los INCUMBENTES nunca se desalojan; los nuevos solo entran
                # en los slots que sobran, por orden de la heuristica A* (posicion en el top-L
                # de aristas, que ya viene ordenado por prioridad del destino).
                #
                # Por que no basta 'prio' (medido, job 92457 -> valid_mrr 0.073): la cabeza de
                # prioridad tiene DOS roles en conflicto. Es el `gate` de la atencion, que la
                # empuja a ~0 para los nodos irrelevantes, Y es la clave de desalojo. Los
                # nuevos entran con `base` (prioridad del estado cero), un ESCALAR que no
                # participa del rol de gate y por lo tanto nunca baja. Entrenando 2 epocas:
                # base 0.574 contra mediana de incumbentes 0.13 => en la capa 5 sobrevive el
                # 6 % del estado acumulado. Cada capa tira lo aprendido y refresca con nodos
                # en estado cero. Comparar un score APRENDIDO contra una CONSTANTE esta mal.
                nk = torch.where(s_ok,
                                 1.0 - torch.arange(s_dst.size(1), device=dev).float() / s_dst.size(1),
                                 torch.full_like(s_dst, -1, dtype=torch.float)).expand(B, -1)
                cat_ids = torch.cat([active, torch.where(s_ok, s_dst, torch.full_like(s_dst, SENT))], 1)
                cat_key = torch.cat([2.0 + prio, nk], 1)
                p1 = cat_key.argsort(dim=1, descending=True)          # key DESC
                ids1, key1 = cat_ids.gather(1, p1), cat_key.gather(1, p1)
                p2 = ids1.argsort(dim=1, stable=True)                 # id ASC, ESTABLE
                ids2, key2 = ids1.gather(1, p2), key1.gather(1, p2)
                dup = torch.zeros_like(ids2, dtype=torch.bool)
                dup[:, 1:] = ids2[:, 1:] == ids2[:, :-1]              # dentro de un id, sobrevive
                key2 = key2.masked_fill(dup | (ids2 == SENT), float('-inf'))  # el de mayor key
                new_active = ids2.gather(1, key2.topk(S, dim=1).indices)
                new_active, _ = new_active.sort(dim=1)
                if getattr(self, '_trace', None) is not None:
                    self._trace.append(dict(
                        cand=int((merged != SENT).sum(1).float().mean()),
                        nuevos_ofrecidos=int(((~global_to_slot(active, merged)[1]) & (merged != SENT)).sum(1).float().mean()),
                        nuevos_admitidos=int(((new_active != SENT).sum(1)
                                              - _survivors(active, new_active, SENT)).float().mean()),
                        incumbentes_vivos=int(_survivors(active, new_active, SENT).float().mean()),
                        base=float(base.mean()),
                        prio_incumb_mediana=float(prio[active != SENT].median()) if (active != SENT).any() else -1.0))
            elif self.evict == 'prio' and merged.size(1) > S:
                # los candidatos ya activos traen su prioridad; los nuevos, la del estado cero
                m_slot, m_found = global_to_slot(active, merged)
                m_prio = torch.where(m_found, prio.gather(1, m_slot),
                                     base.expand(-1, merged.size(1)))
                m_prio = m_prio.masked_fill(merged == SENT, float('-inf'))
                new_active = merged.gather(1, m_prio.topk(S, dim=1).indices)
                if getattr(self, '_trace', None) is not None:
                    self._trace.append(dict(
                        cand=int((merged != SENT).sum(1).float().mean()),
                        nuevos_ofrecidos=int(((~m_found) & (merged != SENT)).sum(1).float().mean()),
                        nuevos_admitidos=int(((new_active != SENT).sum(1)
                                              - _survivors(active, new_active, SENT)).float().mean()),
                        incumbentes_vivos=int(_survivors(active, new_active, SENT).float().mean()),
                        base=float(base.mean()),
                        prio_incumb_mediana=float(prio[active != SENT].median()) if (active != SENT).any() else -1.0))
                new_active, _ = new_active.sort(dim=1)   # restaurar el invariante ORDENADO
            else:
                new_active = merged[:, :S]
            # remapear el estado a los slots nuevos
            old_slot, old_found = global_to_slot(new_active, active)   # donde quedo cada viejo
            keep = old_found & (active != SENT)
            # ⚠️ SLOT CENTINELA (S) para los descartados. Mandarlos al slot 0 -- lo obvio --
            # los hace escribir CERO sobre un nodo REAL: `scatter_` es indefinido con indices
            # duplicados y los ceros pisan el estado. Es la MISMA categoria de bug que rompio
            # la frontera el 2026-08-25 (c). Se reserva un slot extra y se descarta despues.
            # Los slots que NO reciben estado viejo se quedan con la CONDICION DE BORDE
            # (cero en 'onehot'; el embedding del bin de PPR en 'ppr').
            nb = self._boundary(new_active, bins, B, S, N, dev)
            if self.hop_pe:
                # PE de salto: los slots que NO reciben estado viejo son los nodos que se
                # activan en ESTA capa => entraron a li+1 saltos del head. Los viejos se
                # sobreescriben abajo con su estado, asi que el termino solo queda en los nuevos.
                nb = nb + self.hop_emb.weight[li + 1]
            nh = torch.cat([nb, hidden.new_zeros(B, 1, self.hidden_dim)], 1)
            np_ = prio.new_zeros(B, S + 1)
            idx = old_slot.masked_fill(~keep, S)
            nh.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                        hidden * keep.unsqueeze(-1))
            np_.scatter_(1, idx, prio * keep)
            active, hidden, prio = new_active, nh[:, :S], np_[:, :S]

            # (5) atencion en espacio de SLOTS
            a_src, _ = global_to_slot(active, s_src)
            a_dst, _ = global_to_slot(active, s_dst)
            gate = prio.gather(1, a_src) * s_ok
            rel_bias_q = (comp.gather(1, s_rel) * s_ok) if comp is not None else None
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                # GRADIENT CHECKPOINTING. Medido (job 92610, YAGO, batch 8, beta=100 %): la
                # memoria crece 3.5 GB POR CAPA de forma perfectamente lineal (2.94 / 6.55 /
                # 10.26 / 13.89 / 17.53 / 21.06 GB) y son TODAS activaciones por arista
                # retenidas para el backward -- un (B,L,d) son 221 MB y la capa guarda el
                # equivalente a ~16. Checkpointing guarda solo la ENTRADA de la capa
                # (B,S,d) = 126 MB y recomputa el resto en el backward.
                # ⚠️ `use_reentrant=False`: la version reentrante no soporta que la funcion
                # cierre sobre tensores que requieren grad (q_emb, gate) ni que no haya
                # ninguna entrada con requires_grad, que es justo el caso de la capa 0.
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, (a_src, s_rel, a_dst), q_emb, gate, rel_bias_q,
                    use_reentrant=False)
            else:
                hidden = layer(hidden, (a_src, s_rel, a_dst), q_emb=q_emb, gate=gate,
                               rel_bias_q=rel_bias_q)
            # ---- canal GLOBAL (tokens inductores) despues de la agregacion local ----
            gblock = self.global_blocks[li]
            if not isinstance(gblock, nn.Identity):
                valid_now = active != SENT
                if self.global_pool == 'topk':
                    # el pool son los K nodos que la busqueda EXPANDIO en esta capa, en el
                    # layout de slots NUEVO (k_slot era del layout anterior al merge)
                    p_idx, p_found = global_to_slot(active, k_nodes)
                    p_ok = k_ok & p_found
                    p_idx = p_idx.masked_fill(~p_ok, 0)
                else:
                    p_idx, p_ok = None, None
                if self.grad_ckpt and self.training and torch.is_grad_enabled():
                    hidden = torch.utils.checkpoint.checkpoint(
                        gblock, hidden, valid_now, q_emb, p_idx, p_ok, use_reentrant=False)
                else:
                    hidden = gblock(hidden, valid_now, q_emb, p_idx, p_ok)
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                prio = torch.utils.checkpoint.checkpoint(
                    self._priority, hidden, q_emb, use_reentrant=False)
            else:
                prio = self._priority(hidden, q_emb)

        # score final: activos con readout, el resto con el fallback aprendido
        sc = self.readout(self.norm_out(hidden)).squeeze(-1)
        if self.rel_readout is not None:
            sc = sc + self.rel_readout(self.norm_out(hidden), graph, r_index)
        # ⚠️ FUERA DE LUGAR y con slot centinela. La version obvia
        #     out.scatter_(1, idx, torch.where(v, sc, out.gather(1, idx)))
        # rompe autograd (`gather` se necesita para el backward y `scatter_` modifica `out`
        # in-place) Y vuelve a chocar con indices duplicados en el slot 0.
        v = active != SENT
        idx = active.masked_fill(~v, N)                      # invalidos -> columna extra
        z = sc.new_zeros(B, N + 1)
        scored = z.scatter(1, idx, torch.where(v, sc, torch.zeros_like(sc)))
        cover = z.scatter(1, idx, v.to(sc.dtype))
        fb = self.unvisited_score.expand(B, N)
        if self.fallback == 'ppr':
            # score de fallback por BIN de PPR desde el head: los no visitados dejan de ser
            # todos iguales y el ranking de la cola pasa a estar ordenado por posicion global.
            fb = fb + self.fallback_bin(bins.long()).squeeze(-1)
        return torch.where(cover[:, :N] > 0, scored[:, :N], fb)


# ============================================================================
# Sparse attention + expander graphs (estilo Exphormer).
# ============================================================================
# Las aristas expander (grafo aleatorio d-regular) dan atajos estructurales fuera del
# horizonte de propagacion. Fieles a Exphormer, NO llevan relacion KG real (son aleatorias
# => seria inyectar hechos falsos) sino un tipo de arista aprendido propio: aqui una
# relacion sintetica reservada R_exp = num_relation+1 (el self-loop usa num_relation), que
# entra en los dos canales relacionales del sparse (bias b[head,R_exp] y valor g[R_exp]).
# Inductive-safe: R_exp se comparte train/test y las aristas no dependen de identidad de
# entidad. En Exphormer el edge_attr expander es un nn.Embedding(1) compartido; aqui su
# analogo es la fila R_exp de rel_bias/rel_value. Ver SESSION_NOTES.md (2026-07-20).
