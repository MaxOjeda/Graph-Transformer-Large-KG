"""Relational Full-Attention Graph Transformer (RFAT) para KGC inductivo.

Proyecto "Attention" (2026-06): partir desde cero para responder una pregunta de
sanidad antes de invertir en atencion sparse/lineal:

    Un Graph Transformer con atencion FULL (densa, all-pairs O(N^2)) -- que es el
    techo de expresividad de cualquier variante sparse -- ¿supera a NBFNet en
    FB15k-237 inductivo v1? Si la version densa no gana, la sparse no tiene caso.

Diseno (justificado en transformer_vs_nbfnet.tex):
  * NO hay embeddings de entidad ni positional encoding de nodo => inductivo puro
    (el grafo de test tiene entidades disjuntas; nada que transferir salvo relaciones).
  * Labeling trick (NBFNet): x^0_v = emb(r_q) si v == head, si no 0. Ancla la
    propagacion a la fuente y condiciona a la relacion de la query.
  * Cada capa: atencion multi-head DENSA all-pairs (reach global en una capa) con
       (i)  bias escalar relacional b[head, rel] en pares conectados por arista
            (estilo Graphormer: le dice a la atencion quien es vecino y por que relacion);
       (ii) correccion de valor relacional sum_edges alpha * (V_w (.) g[rel])
            (composicion estilo DistMult; es lo que da poder de razonamiento por caminos).
  * Readout puntual MLP(x^L_v) -> score(B, N). Loss = CE de grafo completo (en lightning).

Una sola torre de atencion. NADA de V-RMPNN / QK-RMPNN (eso era KnowFormer y fallo).
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
# RWSE -- Random-Walk Structural Encoding (Dwivedi et al., "Graph Neural Networks
# with Learnable Structural and Positional Representations", ICLR 2022).
# ============================================================================
# p_k(v) = (P^k)_{vv}, con P = D^-1 A la matriz de transicion del random walk sobre
# la adyacencia RELATION-AGNOSTIC del grafo (k=1..walk_length). Es la probabilidad de
# que un random walk que parte en v vuelva a v en exactamente k pasos: una firma
# estructural local de cada nodo. PURAMENTE ESTRUCTURAL => inductivo (NO usa identidad
# de nodo; depende solo de la estructura, transfiere al grafo de test disjunto). Por eso
# NO viola la lista negra #3 (que prohibe PE de nodo / embeddings de entidad).


def compute_rwse(edge_index, num_nodes, walk_length):
    """Devuelve (N, walk_length): diag(P^k) para k=1..walk_length.

    edge_index: (E, 3) = (h, r, t). Se ignora la relacion; se usa la adyacencia
    simetrizada (el grafo ya trae aristas reversas, se simetriza por robustez).
    """
    device = edge_index.device
    src, dst = edge_index[:, 0], edge_index[:, 2]
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    deg = A.sum(-1, keepdim=True).clamp(min=1.0)
    P = A / deg                                   # D^-1 A
    Pk = P.clone()
    diags = []
    for _ in range(walk_length):
        diags.append(torch.diagonal(Pk).clone())  # diag(P^k)
        Pk = Pk @ P
    return torch.stack(diags, dim=-1)             # (N, walk_length)


def rwse_features(graph, device, walk_length, proj):
    """RWSE proyectada a hidden_dim, (N, hidden). Cachea el RWSE crudo (N, walk_length)
    en el objeto graph: es fijo por split (train/val = train graph; test = grafo ind)."""
    cache = getattr(graph, '_rwse_cache', None)
    if cache is None or cache.size(-1) != walk_length or cache.device != device:
        ei = graph.edge_index.to(device)
        cache = compute_rwse(ei, graph.num_nodes, walk_length)
        graph._rwse_cache = cache
    return proj(cache)                            # (N, hidden)


def compute_lappe(edge_index, num_nodes, k):
    """Devuelve (N, k): los k autovectores no triviales del Laplaciano normalizado
    simetrico L = I - D^-1/2 A D^-1/2, correspondientes a los k autovalores mas chicos
    (se salta el trivial ~0, indice 0). Si N-1 < k se rellena con ceros.

    Como RWSE: depende SOLO de la estructura (no de identidad de nodo) => inductivo-safe,
    transfiere al grafo de test disjunto. NO viola lista negra #3. La relacion se ignora
    (adyacencia simetrizada). Los autovectores tienen ambiguedad de signo -> se hace
    sign-flip aleatorio en train (ver lappe_features)."""
    device = edge_index.device
    src, dst = edge_index[:, 0], edge_index[:, 2]
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    deg = A.sum(-1)
    dinv = deg.clamp(min=1.0).pow(-0.5)
    L = torch.eye(num_nodes, device=device) - dinv.unsqueeze(1) * A * dinv.unsqueeze(0)
    # eigh: autovalores ascendentes. Se computa en float64 para estabilidad numerica.
    evals, evecs = torch.linalg.eigh(L.double())
    evecs = evecs.to(A.dtype)                     # (N, N), columnas = autovectores
    # Saltar el trivial (indice 0), tomar los siguientes k.
    pe = evecs[:, 1:k + 1]                         # (N, <=k)
    if pe.size(1) < k:                            # grafo chico: rellenar con ceros
        pad = torch.zeros(num_nodes, k - pe.size(1), device=device)
        pe = torch.cat([pe, pad], dim=1)
    return pe                                     # (N, k)


def lappe_features(graph, device, k, proj, training):
    """LapPE proyectado a hidden_dim, (N, hidden). Cachea los autovectores crudos (N, k)
    en el objeto graph (fijos por split). En train se aplica sign-flip aleatorio por
    autovector para no memorizar el signo arbitrario de eigh (tratamiento canonico de
    LapPE); en eval se usan tal cual."""
    cache = getattr(graph, '_lappe_cache', None)
    if cache is None or cache.size(-1) != k or cache.device != device:
        ei = graph.edge_index.to(device)
        cache = compute_lappe(ei, graph.num_nodes, k)
        graph._lappe_cache = cache
    pe = cache
    if training:
        sign = torch.randint(0, 2, (k,), device=device, dtype=pe.dtype) * 2 - 1  # +-1
        pe = pe * sign.unsqueeze(0)
    return proj(pe)                               # (N, hidden)


# ----------------------------------------------------------------------------
# Source-conditioned random-walk labeling (labeling trick nativo de transformer).
# ----------------------------------------------------------------------------
# A DIFERENCIA de RWSE (diag(P^k), global por nodo) y LapPE (autovectores del
# Laplaciano, globales), esto es CONDICIONADO A LA QUERY: para cada query, el feature
# del nodo v es [P^1[head,v], ..., P^K[head,v]] = probabilidad de landing de un random
# walk de k pasos DESDE el head. Motivacion: con el labeling clasico de NBFNet
# (x^0_v = emb(r_q) si v==head, si no 0) todos los nodos no-source arrancan IDENTICOS
# (cero) => la capa 1 del full attention es degenerada (Q/K/V iguales) y el transformer
# gasta capas reconstruyendo asimetria via las aristas (rehaciendo message passing). Este
# labeling rompe esa simetria dandole a cada nodo una coordenada de proximidad al head,
# distinta por query. Depende SOLO de estructura + head (no de identidad de nodo) =>
# inductivo-safe, transfiere al grafo disjunto. NO viola lista negra #3 (no es PE de
# nodo estatica: cambia con la query, es el labeling trick condicionado a la fuente).


def compute_source_rw(edge_index, num_nodes, walk_length):
    """Devuelve (walk_length, N, N): P^k para k=1..walk_length con P = D^-1 A (adyacencia
    simetrizada, relacion ignorada como en RWSE/LapPE). La fila head de P^k son las
    probabilidades de landing de un random walk de k pasos que arranca en head."""
    device = edge_index.device
    src, dst = edge_index[:, 0], edge_index[:, 2]
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    deg = A.sum(-1, keepdim=True).clamp(min=1.0)
    P = A / deg                                   # D^-1 A
    Pk = P.clone()
    powers = []
    for _ in range(walk_length):
        powers.append(Pk.clone())                 # P^k
        Pk = Pk @ P
    return torch.stack(powers, dim=0)             # (walk_length, N, N)


def source_rw_features(graph, device, walk_length, proj, h_index):
    """Labeling condicionado a la query. Devuelve (B, N, hidden): para cada query b del
    batch, feature del nodo v = proj([P^1[head_b,v], ..., P^K[head_b,v]]). Cachea el stack
    (K, N, N) por split (estructura fija). El gather por head es constante (no grad); el
    grad fluye solo por proj (barato)."""
    cache = getattr(graph, '_source_rw_cache', None)
    if cache is None or cache.size(0) != walk_length or cache.device != device:
        ei = graph.edge_index.to(device)
        cache = compute_source_rw(ei, graph.num_nodes, walk_length)
        graph._source_rw_cache = cache
    feats = cache[:, h_index, :]                  # (K, B, N)
    feats = feats.permute(1, 2, 0)                # (B, N, K)
    return proj(feats)                            # (B, N, hidden)


# ----------------------------------------------------------------------------
# Edge dropout (DropEdge estructural) -- regularizacion contra el overfit AL GRAFO.
# ----------------------------------------------------------------------------
# Diagnostico acumulado (SESSION_NOTES): el sparse tiene la firma valid^/testv
# (gap 0.10-0.16 vs 0.033 de NBFNet) => sobre-ajusta la TOPOLOGIA del train graph y
# transfiere mal al grafo inductivo disjunto. Las 4 hipotesis mecanicas probadas
# (normalizacion, ancla, logits de estado, PEs) atacaban la ARQUITECTURA; ninguna
# ataco el modo de fallo directamente. `--drop` es dropout de features/pesos de
# atencion, y graph_mask solo quita la arista de la query (anti-fuga, no augmentacion):
# hasta ahora NADA perturba la estructura del grafo en train.
#
# DropEdge elimina al azar una fraccion p de aristas en CADA forward de train => el
# modelo ve un grafo distinto por batch y no puede anclar la agregacion aprendida a
# la topologia exacta del train. En eval el grafo va SIEMPRE completo.
#
# Se aplica DESPUES de graph_mask (que indexa sobre el edge_index completo) y ANTES
# de concatenar self-loops, para que ningun nodo quede sin aristas entrantes y el
# segment-softmax del sparse no vea un conjunto vacio.
#
# Nota para --model nbfnet: su PNA calcula scale=log(deg) con deg>=1 (el boundary
# entra como self-mensaje) y usa 1/scale.clamp(min=1e-6); con deg==1 ese scaler
# explota a 1e6. Eso ya existe en el baseline, pero p alto crea mas nodos de grado 1
# y lo amplifica. Revisar si se activa edge_drop sobre nbfnet.


def apply_edge_dropout(edge_index, p, training):
    """Quita al azar una fraccion p de las aristas (E, 3) en train. No-op en eval."""
    if not training or p <= 0.0:
        return edge_index
    keep = torch.rand(edge_index.size(0), device=edge_index.device) >= p
    return edge_index[keep]


# ----------------------------------------------------------------------------
# Relational Path Bias (RPB) -- opcion B: encoding de camino RELACIONAL como
# termino PAR (bias) en la atencion, no como feature de nodo en x^0.
# ----------------------------------------------------------------------------
# Diagnostico de source_rw / RWSE / LapPE: re-codificar estructura como feature
# de nodo en x^0 es redundante con el MP dentro del horizonte y diluye el label
# de la fuente. RPB es distinto en 3 ejes: (a) es RELACIONAL (compone tipos de
# relacion a lo largo del camino, no adyacencia relation-agnostic), (b) esta
# CONDICIONADO A LA QUERY (la compatibilidad de cada arista con r_q modula el
# camino), (c) entra como BIAS PAR en el logit de atencion, no corrompe la
# condicion de borde. La profundidad K del camino esta DESACOPLADA de num_layer,
# asi que inyecta evidencia relacional compuesta FUERA del horizonte de L saltos
# -- el unico margen que GOALS.md deja abierto para superar al message passing.
#
# Mecanica: potencial de evidencia sembrado en la fuente s, propagado K saltos
# por el grafo dirigido, con cada arista pesada por phi_q(rel)=tanh(<u_{r_q},w_rel>)
# (score DistMult del tipo de relacion contra la query, acotado). p^k[b,v] =
# (1/deg_in[v]) sum_{i->v} phi_q(rel) p^{k-1}[b,i]. Es un path-ranking score
# diferenciable, query-conditioned, fuente->candidato. Se proyecta (K -> H) con
# init CERO => el modelo arranca IDENTICO al full attention baseline y aprende a
# usar el prior de camino. Escalar por nodo (no dilucion 64-dim como source_rw).
# Inductivo-safe: solo aristas, tipos de relacion (compartidos) y la fuente.


class RelationalPathBias(nn.Module):
    """Prior de camino relacional query-conditioned como bias per-key de atencion."""

    def __init__(self, num_relation, num_heads, hops, dim):
        super().__init__()
        self.num_relation = num_relation
        self.num_heads = num_heads
        self.hops = hops
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        # u_{r_q}: embedding de la relacion de la query; w_rel: embedding de la
        # relacion de cada arista. phi = tanh(<u,w>/sqrt(dim)) in [-1,1].
        self.query_emb = nn.Embedding(num_relation, dim)
        self.rel_emb = nn.Embedding(num_relation, dim)
        # K evidencias por nodo -> bias por cabeza. Init CERO (baseline al arranque).
        self.to_bias = nn.Linear(hops, num_heads)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)

    def forward(self, edges, h_index, r_index, num_nodes):
        """Devuelve bias (B, H, N): evidencia de camino relacional s->v por query,
        proyectada a un bias per-key para el logit de atencion."""
        src, rel, dst = edges
        B = h_index.size(0)
        N = num_nodes
        E = src.size(0)
        device = h_index.device

        # phi_q(rel) para cada arista y cada query del batch: (B, E), acotado.
        u = self.query_emb(r_index)                      # (B, dim)
        w = self.rel_emb.weight                          # (R, dim)
        phi_rel = torch.tanh((u @ w.t()) * self.scale)   # (B, R)
        phi_e = phi_rel[:, rel]                           # (B, E)

        # grado de entrada (normaliza la propagacion; usa el grafo ya enmascarado).
        deg_in = torch.zeros(N, device=device)
        deg_in.index_add_(0, dst, torch.ones(E, device=device))
        deg_in = deg_in.clamp(min=1.0)                    # (N,)

        # potencial sembrado en la fuente.
        p = torch.zeros(B, N, device=device)
        p[torch.arange(B, device=device), h_index] = 1.0
        src_idx = src.unsqueeze(0).expand(B, E)
        dst_idx = dst.unsqueeze(0).expand(B, E)
        evid = []
        for _ in range(self.hops):
            msg = phi_e * p.gather(1, src_idx)            # (B, E) evidencia src->dst
            p_new = torch.zeros(B, N, device=device)
            p_new.scatter_add_(1, dst_idx, msg)           # agrega en dst
            p_new = p_new / deg_in.unsqueeze(0)           # normaliza por grado
            evid.append(p_new)
            p = p_new
        evid = torch.stack(evid, dim=-1)                  # (B, N, K)
        bias = self.to_bias(evid)                         # (B, N, H)
        return bias.permute(0, 2, 1)                      # (B, H, N)


# ============================================================================
# Parametrizacion relacional del VALOR (`--rel_param`)
# ============================================================================
# Motivacion (medido 2026-08-08): NBFNet gasta el 96 % de sus parametros en
# `relation_linear`, que le da a CADA relacion una matriz d x d por capa (R*d^2).
# Los graph transformers de este harness le dan a cada relacion solo un escalar por
# cabeza (`rel_bias`) y un vector diagonal DistMult (`rel_value`) => R*(H+d). Con la
# config vigente eso son 72 parametros por relacion por capa contra 1024 de NBFNet:
# **14x menos capacidad relacional**. En FB15k-237 (R=360-474) NBFNet tiene 6.2x mas
# parametros en total y gana; en WN18RR (R=22) los conteos casi coinciden (222 K vs
# 216 K) y el sparse EMPATA a NBFNet (0.738 vs 0.740). Correlacion sugerente, no
# probada: esta es la intervencion que la testea.
#
# Es la primera intervencion del proyecto dirigida a la CAPACIDAD RELACIONAL, no a la
# agregacion (opciones A/C, refutadas) ni a encodings estructurales (lista negra #7).
# No toca la lista negra: las relaciones se comparten train/test por construccion, asi
# que sigue siendo inductive-safe (a diferencia de embeddings de entidad o PE de nodo).
#
# 'diag'    : comportamiento historico. msg = v[src] (.) g[rel].
# 'lowrank' : msg = v[src] (.) g[rel] + (v[src] @ U[rel]) @ V[rel]^T, con
#             U,V de forma (H, R, hd, k). Anade 2*H*R*hd*k parametros por capa.
#             **U y V se inicializan en CERO** => en la inicializacion el modelo es
#             EXACTAMENTE el 'diag' (mismo forward bit a bit), y solo diverge al
#             entrenar. Mismo truco de control que se uso con exp_typing path==single.
#             k = hd recupera una matriz por (cabeza, relacion) de rango completo.
#
# Costo: el termino nuevo es O(B*E*hd*k) en tiempo y el gather de U/V es O(H*E*hd*k)
# en memoria, independiente del batch. Barato en los splits inductivos (E ~ 10-22 K);
# en TRANSDUCTIVO (E = 558 K) el gather pesa ~1 GB por capa => no usar ahi sin revisar.


# ============================================================================
# Representacion relacional GENERADA DESDE LA QUERY (`--dependent`)
# ============================================================================
# Es el mecanismo de NBFNet / A*Net (`dependent: yes` en sus configs). En vez de una
# tabla estatica por relacion, cada capa GENERA las representaciones relacionales a
# partir del embedding de la relacion de la query:
#
#     rel = relation_linear(q)  ->  (B, R, d)      [una vez por QUERY, no por arista]
#
# y esa (B,R,d) se usa como modulacion DistMult del valor, exactamente como su
# `relation_input` (`AStarNet/reasoning/layer.py:47,77`; `NBFNet-PyG/nbfnet/layers.py:46,56`).
#
# POR QUE ESTE Y NO `--rel_param lowrank` (medido 2026-08-09 / 2026-08-10):
#   El costo NO esta en cuantos parametros hay por relacion, sino en DONDE se aplican.
#     dependent : se aplica una vez por QUERY   -> O(B*R*d^2) en tiempo, R << E  => gratis
#     lowrank   : se aplica una vez por ARISTA  -> gather (E,H,hd,hd)            => 2.3x mas lento
#   Por eso NBFNet puede darle d^2 parametros a cada relacion y ser 9x mas rapido que
#   nuestro full attention, mientras `lowrank` nos costaba 2.3x en transductivo para un
#   efecto medido de -0.0001. Ver SESSION_NOTES 2026-08-08 (k) y 2026-08-10 (b).
#
# CUANTA CAPACIDAD DA (params por relacion dirigida, por capa, d=32/H=8):
#   `diag` (lo que ya teniamos) ......  H + d  =   40   == su `dependent: no`
#   `--dependent` ....................  d^2+d  = 1056   == su `dependent: yes`
#   ULTRA (GNN de entidades) .........      0          (las computa del grafo de relaciones)
#
# QUE USAN ELLOS EN CADA DATASET (leido de AStarNet/config/transductive/):
#   WN18RR `no` | FB15k-237 `yes` | YAGO3-10 `yes` | ogbl-wikikg2 `yes`
#   => WN18RR es la excepcion, no la regla, y NO es por tener pocas relaciones
#      (YAGO3-10 tiene ~37 y usa `yes`). En wikikg2 `relation_linear` es ~99 % de los
#      6.83 M parametros del modelo, y CERO son embeddings de entidad: por eso escala.
#
# ⚠️ NO TRANSFIERE ZERO-SHOT: `Linear(d, R*d)` tiene R horneado en la forma del parametro
#    => sirve para el paper TRANSDUCTIVO, pero esta cerrado para las Etapas 2-3 (zero-shot),
#    donde la ruta es ULTRA (cero parametros por relacion). ULTRA lo desactiva a proposito
#    (`dependent: False, project_relations: True`, `ULTRA/ultra/layers.py:50-60`).
#
# INIT: `relation_linear` arranca con peso CERO y bias UNO => `rel == 1` para toda relacion,
#    que es exactamente el init de `rel_value` (`torch.ones`) ⇒ **el forward al arranque es
#    bit a bit identico al modo estatico**, y solo diverge al entrenar. Mismo truco de control
#    que `lowrank` (init en 0) y que `exp_typing path == single`. El gradiente de W es
#    dL/drel * q^T, distinto por relacion ⇒ no queda muerto (el modo de fallo que `lowrank`
#    tuvo con U y W ambos en cero). ⚠️ NBFNet usa el init por defecto de nn.Linear (bias 0);
#    esto es una desviacion DELIBERADA para que el ablation tenga control exacto.
#
# INCOMPATIBLE con `--rel_param lowrank` (ambos parametrizan el valor) y con
# `--exp_typing {ultra,path}` (recomponen `rel_value`, que en este modo no existe).


class RelationalAttentionLayer(nn.Module):
    """Una capa: atencion full all-pairs con bias y valor relacional + FFN (pre-LN)."""

    def __init__(self, hidden_dim, num_heads, num_relation, drop,
                 rel_param='diag', rel_rank=4, dependent=False):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim debe ser divisible por num_heads"
        assert rel_param in ('diag', 'lowrank')
        assert not (dependent and rel_param == 'lowrank'), \
            "--dependent y --rel_param lowrank parametrizan ambos el valor: elegir uno"
        self.rel_param = rel_param
        self.dependent = dependent
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_relation = num_relation
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # (i) bias escalar relacional por (cabeza, relacion).
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, num_relation))
        # (ii) modulacion DistMult del valor por (cabeza, relacion). Init ~1 => arranca
        #      cerca de "pasar el valor del vecino sin transformar".
        #      Con --dependent la tabla estatica se reemplaza por una GENERADA desde la
        #      query, una vez por batch (ver bloque `--dependent`).
        if dependent:
            self.rel_value = None
            self.relation_linear = _make_relation_linear(hidden_dim, num_relation)
        else:
            self.rel_value = nn.Parameter(torch.ones(num_heads, num_relation, self.head_dim))
        # (ii-b) correccion de rango bajo por relacion (ver bloque `--rel_param`).
        #        Init en CERO => al arranque el forward es identico al modo 'diag'.
        if rel_param == 'lowrank':
            # Init estilo LoRA: U aleatorio, V en CERO. El producto arranca en 0 (forward
            # identico a 'diag') pero AMBOS reciben gradiente. Inicializar los dos en cero
            # deja la correccion muerta para siempre: d/dU ~ V = 0 y d/dV ~ vU = 0.
            self.rel_u = nn.Parameter(torch.randn(num_heads, num_relation,
                                                  self.head_dim, rel_rank) * 0.02)
            self.rel_w = nn.Parameter(torch.zeros(num_heads, num_relation,
                                                  self.head_dim, rel_rank))

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drop = nn.Dropout(drop)

    def _rel_matrix(self):
        """Matriz de valor por relacion: M[r,h] = diag(g[h,r]) + U[h,r] @ W[h,r]^T, (R,H,hd,hd).

        Algebra: v (.) g[r] + (v @ U[r]) @ W[r]^T == v @ (diag(g[r]) + U[r] W[r]^T). Identico
        al termino anterior, pero **precomputado por RELACION**: cuesta O(H*R*hd^2*k), que es
        independiente de E. Antes el forward hacia DOS gathers por arista (U y W, cada uno
        (E,H,hd,k)) mas un intermedio (B,E,H,k); ahora hace UN gather (E,H,hd,hd) y un solo
        einsum, y el costo deja de depender de k.

        Motivo (medido 2026-08-08 en FB15k-237 transductivo, E=558735): la version por-arista
        costaba 3.7x sobre el modo diag y **casi no bajaba con k** (k=8: 1.42 it/s, k=2: 1.54),
        porque el cuello era mover tensores por arista, no los FLOPs. El layout (R,H,hd,hd)
        ademas hace que el gather sea sobre la dimension 0 y contiguo (antes habia un permute).
        """
        M = torch.einsum('hrdk,hrfk->rhdf', self.rel_u, self.rel_w)
        return M + torch.diag_embed(self.rel_value).permute(1, 0, 2, 3)

    def _rel_value_edges(self, rel, q_emb, B, H, hd):
        """Modulacion DistMult por arista, layout (.,H,E,hd) de esta capa.

        Estatico  -> (1,H,E,hd), broadcast en el batch (el grafo es fijo).
        dependent -> (B,H,E,hd): `relation_linear(q)` da (B,R,d) UNA vez por batch y de ahi
        se lee por arista con index_select (regla 2026-08-09: nunca `tabla[:, rel]`).
        """
        if not self.dependent:
            return self.rel_value[:, rel, :].unsqueeze(0)          # (1,H,E,hd)
        R = self.num_relation
        rel_emb = self.relation_linear(q_emb).view(B, R, H, hd)    # (B,R,H,hd)
        return rel_emb.index_select(1, rel).permute(0, 2, 1, 3)    # (B,H,E,hd)

    def forward(self, x, edges, path_bias=None, q_emb=None):
        """x: (B, N, D). edges: (src, rel, dst) cada uno (E,), arista src --rel--> dst.
        path_bias: (B, H, N) opcional -- bias per-key de camino relacional (RPB).
        q_emb: (B, D) embedding de la relacion de la query; solo lo usa --dependent.

        El nodo destino dst agrega informacion de src (igual que el message passing de
        NBFNet para predecir cola: la evidencia fluye de la fuente hacia el candidato).
        """
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        src, rel, dst = edges

        h = self.norm1(x)
        q = self.to_q(h).view(B, N, H, hd).transpose(1, 2)  # (B,H,N,hd)
        k = self.to_k(h).view(B, N, H, hd).transpose(1, 2)
        v = self.to_v(h).view(B, N, H, hd).transpose(1, 2)

        # Logits densos all-pairs.
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,N,N)

        # (i) bias relacional: a la celda (dst, src) se suma rel_bias[:, rel].
        #     Construimos un denso (H,N,N) (compartido en el batch: el grafo es fijo).
        bias = logits.new_zeros(H, N * N)
        flat_idx = dst * N + src                      # posicion (dst,src) aplanada
        bias.index_add_(1, flat_idx, self.rel_bias[:, rel])  # (H, E) -> columnas
        logits = logits + bias.view(1, H, N, N)

        # (i-b) bias de camino relacional (RPB): per-key (src=dim -1), broadcast en
        #       el nodo que atiende (dst). dst atiende mas a src ricos en evidencia
        #       de camino relacional compuesto desde la fuente de la query.
        if path_bias is not None:
            logits = logits + path_bias.unsqueeze(2)  # (B,H,1,N) -> broadcast en dst

        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)

        # Salida base full attention.
        out = torch.matmul(attn, v)                   # (B,H,N,hd)

        # (ii) correccion de valor relacional sobre aristas (composicion DistMult):
        #      out[dst] += alpha[dst, src] * (V[src] (.) g[rel]).
        a_e = attn[:, :, dst, src]                    # (B,H,E)
        v_e = v[:, :, src, :]                         # (B,H,E,hd)
        if self.rel_param == 'lowrank':
            rel_msg = torch.einsum('bhed,ehdf->bhef', v_e, self._rel_matrix()[rel])
        else:
            g_e = self._rel_value_edges(rel, q_emb, B, H, hd)  # (1|B,H,E,hd)
            rel_msg = v_e * g_e                           # (B,H,E,hd) diagonal DistMult
        msg = a_e.unsqueeze(-1) * rel_msg             # (B,H,E,hd)
        out.index_add_(2, dst, msg)                   # scatter-add en el destino

        out = out.transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.to_out(out))           # residual atencion
        x = x + self.ffn(self.norm2(x))               # residual FFN (pre-LN)
        return x


class GraphTransformer(nn.Module):
    """Stack de capas RelationalAttention + labeling trick + readout puntual."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 use_rwse=False, rwse_dim=16, use_lappe=False, lappe_dim=16,
                 use_source_rw=False, source_rw_dim=8,
                 use_rpb=False, rpb_hops=4, rpb_dim=16, edge_drop=0.0,
                 rel_param='diag', rel_rank=4, dependent=False):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.edge_drop = edge_drop
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.use_source_rw = use_source_rw
        self.source_rw_dim = source_rw_dim
        self.use_rpb = use_rpb

        # Embedding de relacion para el labeling trick (query). Compartido train/test.
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        # RWSE: encoding estructural por nodo (inductivo) sumado a x^0.
        if use_rwse:
            self.rwse_proj = nn.Linear(rwse_dim, hidden_dim)
        # LapPE: autovectores del Laplaciano por nodo (inductivo) sumados a x^0.
        if use_lappe:
            self.lappe_proj = nn.Linear(lappe_dim, hidden_dim)
        # Source-conditioned RW: labeling condicionado a la query (K landing probs
        # desde el head por nodo). NO lleva unsqueeze en forward: ya es (B, N, .).
        if use_source_rw:
            self.source_rw_proj = nn.Linear(source_rw_dim, hidden_dim)
        # RPB: bias de camino relacional query-conditioned (opcion B). Compartido
        # entre capas (se computa una vez por batch y se pasa a cada capa).
        if use_rpb:
            self.rpb = RelationalPathBias(num_relation, num_heads, rpb_hops, rpb_dim)

        self.layers = nn.ModuleList([
            RelationalAttentionLayer(hidden_dim, num_heads, num_relation, drop,
                                     rel_param=rel_param, rel_rank=rel_rank,
                                     dependent=dependent)
            for _ in range(num_layer)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batched_data):
        h_index = batched_data['h_index']   # (B,)
        r_index = batched_data['r_index']   # (B,)
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)     # (E, 3) = (h, r, t)
        # En train se quita la arista de la query (y su reversa) via graph_mask.
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        edge_index = apply_edge_dropout(edge_index, self.edge_drop, self.training)
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        edges = (src, rel, dst)

        # Labeling trick: x^0_v = emb(r_q) si v == head, si no 0.
        x = torch.zeros(B, N, self.hidden_dim, device=device)
        q = self.query_emb(r_index)                   # (B, D)
        x[torch.arange(B, device=device), h_index] = q
        # RWSE estructural sumado a todos los nodos (broadcast en el batch).
        if self.use_rwse:
            x = x + rwse_features(graph, device, self.rwse_dim, self.rwse_proj).unsqueeze(0)
        if self.use_lappe:
            x = x + lappe_features(graph, device, self.lappe_dim, self.lappe_proj,
                                   self.training).unsqueeze(0)
        if self.use_source_rw:
            x = x + source_rw_features(graph, device, self.source_rw_dim,
                                       self.source_rw_proj, h_index)

        # Bias de camino relacional (opcion B): se computa una vez y entra en el
        # logit de cada capa como termino par (per-key), sin tocar x^0.
        path_bias = self.rpb(edges, h_index, r_index, N) if self.use_rpb else None

        for layer in self.layers:
            x = layer(x, edges, path_bias, q_emb=q)

        x = self.norm_out(x)
        score = self.readout(x).squeeze(-1)           # (B, N)
        return score


# ============================================================================
# NBFNet (baseline para comparacion apples-to-apples en este mismo harness).
# ============================================================================
# Reimplementacion fiel del Neural Bellman-Ford Network (Zhu et al. 2021) con la
# config inductiva de FB15k-237 (NBFNet/config/inductive/fb15k237.yaml):
#   message=distmult, aggregate=pna, short_cut=yes, layer_norm=yes, dependent=yes.
# Mismo interfaz que GraphTransformer: forward(batched_data) -> score (B, N), misma
# data y mismo eval full-filtered. SIN atencion: solo message passing condicionado a
# la fuente. Es el caso "w/o attention" y el techo que el RFAT debe superar.


def _make_relation_linear(hidden_dim, num_relation):
    """`Linear(d, R*d)` con init peso=0 / bias=1 => rel == 1 al arranque (ver doc arriba)."""
    lin = nn.Linear(hidden_dim, num_relation * hidden_dim)
    nn.init.zeros_(lin.weight)
    nn.init.ones_(lin.bias)
    return lin


def _scatter_reduce(msg, index, num_nodes, reduce):
    """Agrega msg (B, M, d) por index (M,) en (B, N, d) con la reduccion dada."""
    B, M, d = msg.shape
    out = msg.new_zeros(B, num_nodes, d)
    idx = index.view(1, M, 1).expand(B, M, d)
    out.scatter_reduce_(1, idx, msg, reduce=reduce, include_self=False)
    return out


class NBFNetConv(nn.Module):
    """Una capa de Bellman-Ford generalizado (mensajes DistMult dependientes de query)."""

    def __init__(self, input_dim, output_dim, num_relation, aggregate='pna', layer_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.aggregate = aggregate

        # dependent=yes: las relaciones por capa se derivan de la query.
        self.relation_linear = nn.Linear(input_dim, num_relation * input_dim)

        n_aggr = 12 if aggregate == 'pna' else 1   # pna: 4 aggregadores x 3 scalers
        self.linear = nn.Linear((n_aggr + 1) * input_dim, output_dim)  # +1: concat input
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None

    def forward(self, h, query, edges, num_nodes, boundary):
        """h: (B,N,d) estado previo. query: (B,d). edges: (src,rel,dst). boundary: (B,N,d)."""
        B, N, d = h.shape
        src, rel, dst = edges
        E = src.size(0)

        rel_emb = self.relation_linear(query).view(B, self.num_relation, d)  # (B, R, d)

        # Mensaje DistMult sobre aristas: m_e = h[src] (.) rel_emb[rel].
        msg_edge = h[:, src, :] * rel_emb[:, rel, :]                 # (B, E, d)
        # Boundary como self-loop (condicion inicial v=u): se agrega como mensaje propio.
        node_out = torch.cat([dst, torch.arange(N, device=h.device)])  # (E+N,)
        msg = torch.cat([msg_edge, boundary], dim=1)                 # (B, E+N, d)

        if self.aggregate == 'sum':
            update = _scatter_reduce(msg, node_out, N, 'sum')        # (B, N, d)
        else:  # pna
            mean = _scatter_reduce(msg, node_out, N, 'mean')
            sq = _scatter_reduce(msg * msg, node_out, N, 'mean')
            mx = _scatter_reduce(msg, node_out, N, 'amax')
            mn = _scatter_reduce(msg, node_out, N, 'amin')
            std = (sq - mean * mean).clamp(min=1e-6).sqrt()
            feat = torch.stack([mean, mx, mn, std], dim=-1).flatten(-2)  # (B,N,4d)
            deg = torch.zeros(N, device=h.device).index_add_(
                0, node_out, torch.ones(E + N, device=h.device)).unsqueeze(-1)  # (N,1)
            scale = deg.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale,
                                1.0 / scale.clamp(min=1e-6)], dim=-1)        # (N,3)
            update = (feat.unsqueeze(-1) * scales.view(1, N, 1, 3)).flatten(-2)  # (B,N,12d)

        out = self.linear(torch.cat([h, update], dim=-1))           # combine [input, update]
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        out = F.relu(out)
        return out


class NBFNet(nn.Module):
    """Neural Bellman-Ford Network condicionado a la fuente (tail prediction)."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads=None, drop=0.0,
                 aggregate='pna', short_cut=True, edge_drop=0.0):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.short_cut = short_cut
        self.edge_drop = edge_drop

        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        self.layers = nn.ModuleList([
            NBFNetConv(hidden_dim, hidden_dim, num_relation, aggregate=aggregate)
            for _ in range(num_layer)
        ])
        # Readout puntual sobre h^L_{u->v} (concat con la query para condicionar el score).
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, batched_data):
        """Corre el message passing y devuelve los estados de nodo h^L_{u->v} (B, N, d),
        antes del readout. Reusado por SparseNBFValueTransformer como V-stream."""
        h_index = batched_data['h_index']
        r_index = batched_data['r_index']
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        edge_index = apply_edge_dropout(edge_index, self.edge_drop, self.training)
        edges = (edge_index[:, 0], edge_index[:, 1], edge_index[:, 2])

        query = self.query_emb(r_index)                              # (B, d)
        boundary = torch.zeros(B, N, self.hidden_dim, device=device)
        boundary[torch.arange(B, device=device), h_index] = query

        h = boundary
        for layer in self.layers:
            out = layer(h, query, edges, N, boundary)
            if self.short_cut and out.shape == h.shape:
                out = out + h
            h = out
        return h                                                    # (B, N, d)

    def forward(self, batched_data):
        h = self.encode(batched_data)
        score = self.mlp(h).squeeze(-1)                             # (B, N)
        return score


# ============================================================================
# Sparse Graph Transformer: atencion restringida a la ADYACENCIA del grafo.
# ============================================================================
# Variante sparse del RFAT: cada nodo destino atiende SOLO a sus vecinos entrantes
# (aristas), no a los N nodos. Softmax por nodo sobre sus aristas entrantes + un
# self-loop. Estructuralmente es "NBFNet con agregacion APRENDIDA por atencion en vez
# de sum/pna fija" (caso (b) del analisis: graph attention ~ un hop con pesos aprendidos).
# Mismo labeling trick / bias+valor relacional / readout / eval que el RFAT denso.


class SparseRelationalAttentionLayer(nn.Module):
    """Atencion sparse sobre aristas + FFN (pre-LN).

    attn controla la agregacion por nodo destino:
      - 'softmax': segment-softmax (promedio ponderado convexo; ciego al numero de
        aristas de soporte y al grado, porque normaliza a sum(alpha)=1).
      - 'sigmoid': gates sigmoides por arista SIN normalizar => suma ponderada
        aprendida. Conserva el conteo de caminos de evidencia y la sensibilidad al
        grado (como la agregacion sum de NBFNet) manteniendo la selectividad por query.
      - 'degree': segment-softmax reescalado por log(1+grado_in) del destino =>
        reinyecta el conteo como scaler (estilo PNA) sin abandonar el softmax.
      - 'anchor': interpola alpha_final = lambda*alpha_softmax + (1-lambda)*uniforme,
        con lambda aprendido por cabeza (sigmoide). uniforme = 1/grado_in reparte por
        igual entre aristas entrantes (= agregacion media fija, la que transfiere).
        lambda->0 recupera la media GNN; lambda->1 la atencion aprendida. Regulariza
        hacia la agregacion fija para atacar la firma de overfit valid^/test_ (opcion C).
      - 'rel': ablation de transferencia. ELIMINA el termino q.k (dependiente de los
        estados de nodo, cuyo overfit al train graph es el sospechoso dominante) y deja
        el logit puramente relacional: b[head,rel] + <u[head,r_q], w[head,rel]>/sqrt(d)
        (compatibilidad query-relacion aprendida). Todo a nivel de relacion => se
        comparte train/test por construccion, misma transferibilidad que NBFNet.
        Segment-softmax estandar. Si recupera la zona de NBFNet, el overfit estaba en
        q.k; si no, la agregacion aprendida per se queda refutada.
      - 'qc': paquete QC-Exphormer (port del proyecto Exphormer_Max, run trans 0.456
        pre-fuga): (i) Q anclado a x0 (boundary condition, NO lee h acumulada);
        (ii) logit TRILINEAL (q (.) k (.) e[rel]).1 con relacion vectorial e[rel] en vez
        de bias escalar; (iii) conditioning aditivo c_q = emb_capa(r_q) sumado a q/k/e;
        (iv) agregacion exp(clip(logit,+-5)) + SUMA sin normalizar (NBFNet-style).
        El modelo ademas aplica residual Bellman-Ford (x += x0 por capa) en este modo.
    """

    def __init__(self, hidden_dim, num_heads, num_relation, drop, attn='softmax',
                 rel_param='diag', rel_rank=4, dependent=False):
        super().__init__()
        assert hidden_dim % num_heads == 0
        assert attn in ('softmax', 'sigmoid', 'degree', 'anchor', 'rel', 'qc')
        assert rel_param in ('diag', 'lowrank')
        assert not (dependent and rel_param == 'lowrank'), \
            "--dependent y --rel_param lowrank parametrizan ambos el valor: elegir uno"
        self.attn = attn
        self.rel_param = rel_param
        self.dependent = dependent
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # +1 relacion extra reservada al self-loop (indice = num_relation).
        self.self_rel = num_relation
        R = num_relation + 1

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # 'rel' no usa el termino q.k => no crea las proyecciones (quedarian muertas).
        if attn != 'rel':
            self.to_q = nn.Linear(hidden_dim, hidden_dim)
            self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        self.rel_bias = nn.Parameter(torch.zeros(num_heads, R))
        # Con --dependent el valor relacional se GENERA desde la query (ver ese bloque).
        # OJO: R = num_relation+1 aqui (self-loop), +2 en el expander => la tabla generada
        # cubre tambien esas relaciones sinteticas, igual que la estatica.
        self.num_rel_table = R
        if dependent:
            self.rel_value = None
            self.relation_linear = _make_relation_linear(hidden_dim, R)
        else:
            self.rel_value = nn.Parameter(torch.ones(num_heads, R, self.head_dim))
        # Correccion de rango bajo por relacion (ver bloque `--rel_param`). Init en
        # CERO => al arranque el forward es identico al modo 'diag'.
        if rel_param == 'lowrank':
            # Init estilo LoRA: U aleatorio, V en CERO (ver nota en la capa full).
            self.rel_u = nn.Parameter(
                torch.randn(num_heads, R, self.head_dim, rel_rank) * 0.02)
            self.rel_w = nn.Parameter(torch.zeros(num_heads, R, self.head_dim, rel_rank))

        # opcion C: lambda por cabeza para mezclar softmax con la media uniforme.
        # init en 0 => sigmoid=0.5 (mezcla neutra); se aprende hacia media o atencion.
        if attn == 'anchor':
            self.anchor_logit = nn.Parameter(torch.zeros(num_heads))

        # 'rel': tablas de compatibilidad query-relacion (init chico => logit ~ 0
        # => arranca como softmax(b[head,rel]), cuasi-uniforme como el base).
        if attn == 'rel':
            self.rel_att_q = nn.Parameter(
                torch.randn(num_heads, num_relation, self.head_dim) * 0.01)
            self.rel_att_k = nn.Parameter(
                torch.randn(num_heads, R, self.head_dim) * 0.01)

        # 'qc': relacion vectorial del logit trilineal (init unos => al inicio el
        # logit se reduce a q.k estandar) + conditioning c_q con proyecciones
        # near-zero (receta QC-Exphormer: estable al arranque).
        if attn == 'qc':
            self.rel_e = nn.Parameter(torch.ones(num_heads, R, self.head_dim))
            self.q_rel_emb = nn.Embedding(num_relation, hidden_dim)
            nn.init.normal_(self.q_rel_emb.weight, std=0.01)
            self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.proj_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
            for m in (self.proj_q, self.proj_k, self.proj_e):
                nn.init.normal_(m.weight, std=0.01)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drop = nn.Dropout(drop)

    def _rel_matrix(self):
        """Matriz de valor por relacion: M[r,h] = diag(g[h,r]) + U[h,r] @ W[h,r]^T, (R,H,hd,hd).

        Algebra: v (.) g[r] + (v @ U[r]) @ W[r]^T == v @ (diag(g[r]) + U[r] W[r]^T). Identico
        al termino anterior, pero **precomputado por RELACION**: cuesta O(H*R*hd^2*k), que es
        independiente de E. Antes el forward hacia DOS gathers por arista (U y W, cada uno
        (E,H,hd,k)) mas un intermedio (B,E,H,k); ahora hace UN gather (E,H,hd,hd) y un solo
        einsum, y el costo deja de depender de k.

        Motivo (medido 2026-08-08 en FB15k-237 transductivo, E=558735): la version por-arista
        costaba 3.7x sobre el modo diag y **casi no bajaba con k** (k=8: 1.42 it/s, k=2: 1.54),
        porque el cuello era mover tensores por arista, no los FLOPs. El layout (R,H,hd,hd)
        ademas hace que el gather sea sobre la dimension 0 y contiguo (antes habia un permute).
        """
        M = torch.einsum('hrdk,hrfk->rhdf', self.rel_u, self.rel_w)
        return M + torch.diag_embed(self.rel_value).permute(1, 0, 2, 3)

    def _rel_lookup(self, table, rel, axis):
        """Lee una tabla relacional por arista con `index_select`, NO con `table[:, rel]`.

        ⚠️ NO "simplificar" esto de vuelta a indexacion avanzada: cuesta hasta 2x.

        Medido 2026-08-09 (FB15k-237 transductivo, A100, fwd+bwd end-to-end):

            modelo                     x[:, rel]   index_select   speedup
            sparse                       5.87        7.52          1.28x
            sparse_exp deg3              3.33        6.83          2.08x
            sparse_exp deg3 + lowrank    1.39        2.58          1.87x

        CAUSA (profiler, job 91473): el 99.5 % del sobrecosto del expander estaba en
        `aten::_index_put_impl_` + `indexing_backward`, que es el BACKWARD de la indexacion
        avanzada. Ese backward acumula un gradiente por ARISTA en la fila de su relacion, con
        atomicos; los tramos de relacion unica —N self-loops y sobre todo **6*N aristas
        expander, todas con R_exp**— concentran decenas de miles de atomicos sobre UNA fila.
        El backward de `index_select` es `index_add_`, que el mismo profiler mide en 5-6 ms
        contra 90-209 ms de `_index_put_impl_`.

        EFECTO PRINCIPAL: el sobrecosto del expander cae de **1.76x a 1.08x** => deja de ser
        un problema. (Se probo ademas leer los tramos de relacion unica por `expand` en vez de
        gather: **neutro (1.00x)**, no se conserva. Y ordenar aristas por `dst`: tambien
        neutro, ver entrada 2026-08-08 (l).)
        """
        return table.index_select(axis, rel)

    def _exp_rel_params(self, bias_e, g_e, ctx, B):
        """Reemplaza el bias/valor relacional de las aristas EXPANDER (que son el tramo
        final de la lista de aristas, desde ctx['start']) segun la variante de tipado.
        'single' no llama a esto: usa la fila R_exp de las tablas como cualquier relacion."""
        assert not self.dependent, \
            "--exp_typing {ultra,path} recompone rel_value, que --dependent no materializa"
        s = ctx['start']
        if ctx['typing'] == 'path':
            # Composicion a lo largo del camino: bias = suma de b[r_k], valor = producto
            # DistMult de g[r_k]. Las posiciones fuera del camino son neutras (0 y 1).
            paths, pmask = ctx['paths'], ctx['mask']              # (Ex,K), (Ex,K)
            b = self.rel_bias[:, paths]                           # (H, Ex, K)
            b = (b * pmask.unsqueeze(0)).sum(-1)                  # (H, Ex)
            gg = self.rel_value[:, paths, :]                      # (H, Ex, K, hd)
            gg = torch.where(pmask.view(1, *pmask.shape, 1), gg, torch.ones_like(gg))
            gg = gg.prod(dim=2)                                   # (H, Ex, hd)
            bias_new = b.transpose(0, 1).unsqueeze(0)             # (1, Ex, H)
            g_new = gg.permute(1, 0, 2).unsqueeze(0)              # (1, Ex, H, hd)
        else:  # 'ultra': relacion prestada = mezcla de las tablas pesada por w(r_q).
            w = ctx['w']                                          # (B, Rr) softmax
            Rr = w.size(1)
            b = torch.einsum('br,hr->bh', w, self.rel_bias[:, :Rr])            # (B,H)
            gg = torch.einsum('br,hrd->bhd', w, self.rel_value[:, :Rr, :])     # (B,H,hd)
            Ex = bias_e.size(1) - s
            bias_new = b.unsqueeze(1).expand(B, Ex, -1)
            g_new = gg.unsqueeze(1).expand(B, Ex, -1, -1)
        nb = bias_new.size(0)
        bias_e = torch.cat([bias_e[:, :s].expand(nb, -1, -1), bias_new], dim=1)
        g_e = torch.cat([g_e[:, :s].expand(nb, -1, -1, -1), g_new], dim=1)
        return bias_e, g_e

    def forward(self, x, edges, x0=None, r_index=None, exp_ctx=None, q_emb=None):
        """edges: (src, rel, dst) YA aumentadas con self-loops (rel==self_rel).
        q_emb (B,D) es el embedding de la relacion de la query; solo lo usa --dependent.
        x0 (B,N,D) es el estado inicial (labeling trick) y r_index (B,) la relacion
        de la query; solo los usan los modos 'qc' (Q anclado a x0 + c_q) y 'rel' (c_q).
        Los demas modos los ignoran.
        exp_ctx (opcional) describe el tipado de las aristas expander ('ultra'/'path');
        None => las relaciones se leen de las tablas por indice, como siempre."""
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        src, rel, dst = edges
        E = src.size(0)

        h = self.norm1(x)
        v = self.to_v(h).view(B, N, H, hd)

        # Parametros relacionales por arista. Por defecto son un lookup por indice en las
        # tablas; con exp_ctx las aristas expander (tramo final) los reciben compuestos
        # segun la variante de tipado ('ultra' = relacion prestada, 'path' = composicion).
        bias_e = self._rel_lookup(self.rel_bias, rel, 1).transpose(0, 1).unsqueeze(0)
        # En 'lowrank' el valor relacional se aplica como MATRIZ por relacion (_rel_matrix),
        # asi que no se materializa el gather diagonal (un tensor (1,E,H,hd) de mas).
        if self.rel_param == 'lowrank':
            g_e = None
        elif self.dependent:
            # (B,R,d) una vez por batch -> lectura por arista con index_select. (B,E,H,hd)
            rel_emb = self.relation_linear(q_emb).view(B, self.num_rel_table, H, hd)
            g_e = rel_emb.index_select(1, rel)
        else:
            g_e = self._rel_lookup(self.rel_value, rel, 1) \
                      .permute(1, 0, 2).unsqueeze(0)      # (1, E, H, hd)
        if exp_ctx is not None:
            bias_e, g_e = self._exp_rel_params(bias_e, g_e, exp_ctx, B)

        if self.attn == 'rel':
            # Logit SIN estados de nodo: bias relacional + compatibilidad (r_q, rel).
            cq = self.rel_att_q[:, r_index, :]                       # (H, B, hd)
            ck = self.rel_att_k[:, rel, :]                           # (H, E, hd)
            logit = torch.einsum('hbd,hed->beh', cq, ck) * self.scale  # (B, E, H)
            logit = logit + bias_e
        elif self.attn == 'qc':
            # Trilineal QC-Exphormer: Q desde x0 (no lee h), e[rel] vectorial,
            # conditioning aditivo c_q en los tres canales.
            q = self.to_q(x0).view(B, N, H, hd)
            k = self.to_k(h).view(B, N, H, hd)
            cq = self.q_rel_emb(r_index)                             # (B, D)
            qc = self.proj_q(cq).view(B, 1, H, hd)
            kc = self.proj_k(cq).view(B, 1, H, hd)
            ec = self.proj_e(cq).view(B, 1, H, hd)
            e = self.rel_e[:, rel, :].permute(1, 0, 2).unsqueeze(0)  # (1, E, H, hd)
            logit = ((q[:, dst] + qc) * (k[:, src] + kc)
                     * (e + ec)).sum(-1) * self.scale                # (B, E, H)
        else:
            q = self.to_q(h).view(B, N, H, hd)
            k = self.to_k(h).view(B, N, H, hd)
            # Logit por arista: (q_dst . k_src)/sqrt(d) + bias relacional. (B, E, H)
            logit = (q[:, dst] * k[:, src]).sum(-1) * self.scale + bias_e

        if self.attn == 'sigmoid':
            # Gate por arista, sin normalizacion (no necesita estabilizacion por max).
            alpha = torch.sigmoid(logit)                            # (B, E, H)
        elif self.attn == 'qc':
            # exp-suma sin normalizar (QC-Exphormer / NBFNet-style); el clip +-5
            # (mismo de Exphormer_Max) acota el rango sin necesitar max-subtract.
            alpha = torch.exp(logit.clamp(-5.0, 5.0))               # (B, E, H)
        else:
            # Segment-softmax por nodo destino (sobre sus aristas entrantes + self-loop).
            idx = dst.view(1, E, 1).expand(B, E, H)
            node_max = logit.new_full((B, N, H), float('-inf'))
            node_max.scatter_reduce_(1, idx, logit, reduce='amax', include_self=False)
            logit = (logit - node_max.gather(1, idx)).exp()
            denom = logit.new_zeros(B, N, H)
            denom.index_add_(1, dst, logit)
            alpha = logit / (denom.gather(1, idx) + 1e-9)           # (B, E, H)
            if self.attn == 'anchor':
                # media uniforme por nodo destino: 1/grado_in por arista entrante.
                deg = torch.zeros(N, device=x.device)
                deg.index_add_(0, dst, torch.ones(E, device=x.device))
                unif = (1.0 / deg[dst].clamp(min=1.0)).view(1, E, 1)  # (1, E, 1)
                lam = torch.sigmoid(self.anchor_logit).view(1, 1, H)  # (1, 1, H)
                # ambos suman 1 por dst => la mezcla convexa tambien (conserva promedio).
                alpha = lam * alpha + (1.0 - lam) * unif
        alpha = self.drop(alpha)

        # Mensaje relacional (composicion DistMult) ponderado por la atencion.
        v_src = v[:, src]                                          # (B,E,H,hd)
        if self.rel_param == 'lowrank':
            M_e = self._rel_lookup(self._rel_matrix(), rel, 0)   # (E,H,hd,hd)
            rel_msg = torch.einsum('behd,ehdf->behf', v_src, M_e)
        else:
            rel_msg = v_src * g_e                                  # diagonal DistMult
        msg = alpha.unsqueeze(-1) * rel_msg
        out = msg.new_zeros(B, N, H, hd)
        out.index_add_(1, dst, msg)                                # (B, N, H, hd)

        if self.attn == 'degree':
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, dst, torch.ones(E, device=x.device))
            out = out * torch.log1p(deg).view(1, N, 1, 1)

        out = out.reshape(B, N, D)
        x = x + self.drop(self.to_out(out))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# GT disperso con PODA APRENDIDA (`--model sparse_pruned`) -- ruta a wikikg2
# ============================================================================
# Motivo (SESSION_NOTES 2026-08-22 / DISENO_GT_PODA.md): nuestro GT sobre el grafo completo
# pide ~180 GB a batch 1 en ogbl-wikikg2 (32.2 M aristas) => no escala, y NBFNet tampoco
# (OOM a batch 1). El unico mecanismo conocido de escala es PODAR: A*Net propaga sobre el
# 0.2 % de nodos/aristas y logra SOTA. Y medimos que su poda es SIN PERDIDA en inferencia
# (Delta +0.0002 entre 10 % y 100 %).
#
# DIFERENCIA CON A*Net, deliberada: ellos usan un top-k SIN PADDING (Apendice C, multi-key
# sort) porque empaquetan subgrafos de tamano VARIABLE. Aca se usa **top-L de tamano FIJO por
# query**: es `topk` + `gather`, operaciones estandar, y da la misma reduccion de memoria
# (`O(B*L*d)` en vez de `O(B*E*d)`). Se paga algo de computo en padding; se gana que sea
# implementable en dias y no en semanas.
#
# SELECCION ITERATIVA (como A*Net, no estatica): en cada capa se re-puntuan los nodos y se
# re-seleccionan las L aristas. Asi la frontera CRECE desde la fuente, que es lo que hace
# util la poda. Con seleccion estatica el subgrafo no podria expandirse.
#
# WEIGHT SHARING: la funcion de prioridad comparte el MLP con el readout. En A*Net eso es lo
# que le da supervision (no hay etiqueta de "camino importante"); replicarlo es esencial y su
# ablation lo mide: sin weight sharing su MRR cae de 0.411 a 0.374.
#
# EL SCORE MULTIPLICA EL ESTADO DEL NODO, no el peso de la arista:
#     layer_input = sigmoid(score) * hidden
# Es literalmente `AStarNet/reasoning/model.py:319`. Ademas de darle gradiente al selector,
# es lo que en A*Net preserva el kernel fusionado. Aca no tenemos kernel igual (la atencion
# pone pesos por arista que requieren gradiente, ver SESSION_NOTES 2026-08-21 c), pero se
# mantiene la forma porque es donde la supervision fluye.
#
# ⚠️ LIMITE CONOCIDO para wikikg2: el paso de scoring construye `(B, E)` para hacer topk.
# Con E = 32.2 M y B = 32 son ~4 GB solo en esa tabla. Para FB15k-237 (E = 544 K => 70 MB) no
# es problema. A escala hay que puntuar solo la FRONTERA en vez de todas las aristas. Queda
# fuera de esta version a proposito: primero validar correctitud en FB15k-237.


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


class PrunedSparseGraphTransformer(nn.Module):
    """GT disperso con poda aprendida por query (ver bloque de docs arriba).

    Flujo por capa, siguiendo la estructura de A*Net:
      1. puntuar nodos con el MLP COMPARTIDO con el readout (weight sharing)
      2. puntuar cada arista por el score de su nodo FUENTE
      3. top-L aristas por query  ->  (B, L)
      4. escalar el estado del nodo por sigmoid(score)  (por donde entra el gradiente)
      5. atencion sobre esas L aristas
    """

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 edge_ratio=0.1, dependent=False, edge_drop=0.0, prune_attn='softmax',
                 node_ratio=0.1, rel_readout=False):
        super().__init__()
        self.node_ratio = node_ratio
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.self_rel = num_relation
        self.edge_ratio = edge_ratio
        self.edge_drop = edge_drop
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        self.layers = nn.ModuleList([
            PrunedSparseAttentionLayer(hidden_dim, num_heads, num_relation, drop,
                                       dependent=dependent, prune_attn=prune_attn)
            for _ in range(num_layer)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        # WEIGHT SHARING: el mismo MLP puntua nodos para la poda y produce el score final.
        # Es el mecanismo que le da supervision al selector (A*Net mide -0.037 de MRR si se
        # rompe). NO separar en dos cabezas.
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        # Funcion de prioridad de A*Net (§3.1 de A_star_Net_resumen.md):
        #     s(x) = sigma( f( h_q(u,x)  (x)  g([h_q(u,x), q]) ) )
        # `g` es un FFN de 1 capa sobre [h, q] que estima la "distancia restante" al
        # objetivo; el producto con h juega el papel de d(u,x) (x) g(x,v) del A* clasico.
        # `f` es el READOUT COMPARTIDO: sin weight sharing su MRR cae de 0.411 a 0.374,
        # porque no hay supervision directa para la prioridad.
        self.prio_g = nn.Linear(hidden_dim * 2, hidden_dim)
        # Canal global relacional: O(B*N*d), a nivel de NODO => la poda (que descarta
        # ARISTAS) no lo toca. Es lo que puede diferenciarnos sin renunciar a escalar.
        self.rel_readout = RelationalGlobalReadout(hidden_dim, num_relation) \
            if rel_readout else None
        # Score de los nodos NO visitados. Aprendido, no -inf: es lo que arregla el MR, que
        # en A*Net es 4.4x peor que NBFNet en FB15k-237 y 9.0x en WN18RR justamente porque
        # los nodos no visitados quedan sin score (SESSION_NOTES 2026-08-22 b).
        self.unvisited_score = nn.Parameter(torch.zeros(1))

    def _priority(self, x, q_emb):
        """s(x) = sigmoid(readout(x (.) g([x, q]))). Ver `self.prio_g`."""
        B, N, _ = x.shape
        q = q_emb.unsqueeze(1).expand(-1, N, -1)
        g = self.prio_g(torch.cat([x, q], dim=-1))
        return torch.sigmoid(self.readout(self.norm_out(x * g)).squeeze(-1))   # (B,N)


    def forward(self, batched_data):
        graph = batched_data['graph']
        h_index, r_index = batched_data['h_index'], batched_data['r_index']
        N, B = graph.num_nodes, h_index.size(0)
        device = h_index.device

        edge_index = graph.edge_index.to(device)
        if batched_data.get('graph_mask') is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        edge_index = apply_edge_dropout(edge_index, self.edge_drop, self.training)
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        loop = torch.arange(N, device=device)
        src = torch.cat([src, loop]); dst = torch.cat([dst, loop])
        rel = torch.cat([rel, torch.full((N,), self.self_rel, device=device)])
        E = src.size(0)
        L = max(1, min(E, int(self.edge_ratio * E)))

        q_emb = self.query_emb(r_index)                      # (B, D)
        x = torch.zeros(B, N, self.hidden_dim, device=device)
        ar = torch.arange(B, device=device)
        x[ar, h_index] = q_emb

        # Score inicial: solo la fuente es visible (los demas estados son cero, asi que el
        # MLP no tendria de donde puntuar). Equivale al `indicator` de A*Net.
        node_score = torch.zeros(B, N, device=device)
        node_score[ar, h_index] = 1.0
        # Se acumula sobre TODAS las capas: un nodo alcanzado en la capa 3 tiene estado
        # valido aunque no aparezca en la seleccion de la 6.
        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        visited[ar, h_index] = True
        if getattr(self, '_all_visited', False):
            # Solo para el TEST DE EQUIVALENCIA contra SparseStateGraphTransformer: alinear
            # el conjunto de nodos que aportan aristas. Sin esto el denso arranca con solo el
            # head en `visited` y el disperso con todos => ven grafos distintos en la capa 0.
            visited[:] = True

        src_b = src.view(1, E).expand(B, E)
        dst_b = dst.view(1, E).expand(B, E)
        K = max(1, min(N, int(self.node_ratio * N)))
        for layer in self.layers:
            # === SELECCION EN DOS ETAPAS (A_star_Net_resumen.md §3.2) ===
            # (1) top-K NODOS por prioridad, dentro de la frontera ya alcanzada.
            s_node = node_score.masked_fill(~visited, float('-inf'))
            topk_n = s_node.topk(K, dim=1).indices                      # (B,K)
            in_X = torch.zeros_like(visited)
            in_X.scatter_(1, topk_n, True)
            in_X &= visited                        # descarta el relleno del topk de nodos
            # (2) candidatas = aristas SALIENTES de X; se ordenan por la prioridad del nodo
            #     DESTINO, no de la fuente. Es la heuristica de A*: estima la distancia
            #     RESTANTE al objetivo, asi que evalua adonde se llega. Ordenar por la fuente
            #     (lo que hacia la v1/v2) no usa la heuristica en absoluto.
            edge_score = node_score.gather(1, dst_b)
            edge_score = edge_score.masked_fill(~in_X.gather(1, src_b), float('-inf'))
            # ⚠️ FRONTERA CONECTADA (2026-08-25, tras diagnosticar el fallo de la v1):
            # solo son candidatas las aristas cuya FUENTE YA fue alcanzada. Sin esto, el
            # top-L global puede elegir aristas DESCONECTADAS del head y no fluye nada desde
            # la fuente: con un selector malo el modelo colapsa a MRR 0.0001, mientras que
            # A*Net con seleccion ALEATORIA aun da 0.378 -- porque ellos expanden una
            # frontera desde el head POR CONSTRUCCION (seleccionan nodos y luego aristas en
            # SU vecindario). Aca la conectividad pasa de ser algo que el selector debe
            # aprender a ser una garantia estructural; al selector solo le queda decidir
            # CUALES de las aristas alcanzables mirar, que es el problema de A*Net.
            edge_score = edge_score.masked_fill(~visited.gather(1, src_b), float('-inf'))
            # (3) top-L por query dentro de la frontera.
            sel = edge_score.topk(L, dim=1).indices                 # (B, L)
            e_src = src.view(1, E).expand(B, E).gather(1, sel)
            e_rel = rel.view(1, E).expand(B, E).gather(1, sel)
            e_dst = dst.view(1, E).expand(B, E).gather(1, sel)
            # (4) el score del emisor pesa su MENSAJE (Ec. 12 de A*Net). Por aca entra el
            #     gradiente al selector: sin esto `topk` no es diferenciable y el score
            #     nunca aprenderia.
            sel_score = edge_score.gather(1, sel)                    # (B, L)
            # Relleno del top-L cuando la frontera tiene < L aristas: score -inf => gate 0.
            real = torch.isfinite(sel_score)                         # (B, L)
            # (3) el mensaje se pesa por la prioridad del nodo FUENTE (Ec. 12), no por la
            #     del destino que se uso para SELECCIONAR. Son dos usos distintos.
            gate = node_score.gather(1, e_src) * real                # (B, L)
            # (5) atencion sobre el subgrafo seleccionado.
            x = layer(x, (e_src, e_rel, e_dst), q_emb=q_emb, gate=gate)
            # ⚠️ Solo las aristas REALES amplian la frontera. Pero OJO con COMO se marca:
            # `scatter_(1, e_dst, real)` NO sirve, porque `e_dst` tiene INDICES DUPLICADOS
            # (muchas aristas comparten destino) y scatter_ es indefinido ahi: las ~55 K
            # aristas de relleno escriben False y BORRAN las marcas True de las reales segun
            # el orden de escritura. Medido (job 92329): con esa version la frontera crecia
            # ~2x por capa en vez de ~37x (el grado medio), y tras 6 capas usaba el 12 % del
            # presupuesto => el modelo veia 1-2 saltos y el valid caia a 0.15-0.20.
            # `scatter_reduce_(amax)` SI esta definido con duplicados y da semantica de OR.
            newly = e_dst.new_zeros(B, N, dtype=torch.int8)
            newly.scatter_reduce_(1, e_dst, real.to(torch.int8), reduce='amax',
                                  include_self=False)
            visited = visited | newly.bool()
            # (1) re-puntuar para la capa siguiente: la frontera crece desde la fuente.
            node_score = self._priority(x, q_emb)                    # (B, N) en [0,1]

        score = self.readout(self.norm_out(x)).squeeze(-1)
        if self.rel_readout is not None:
            score = score + self.rel_readout(self.norm_out(x), graph, r_index)
        # Nodos nunca tocados por ninguna seleccion -> score aprendido de fallback.
        return torch.where(visited, score, self.unvisited_score.expand_as(score))


# ============================================================================
# Readout relacional GLOBAL (`--rel_readout`) -- la unica salida PROBADA de la clase de NBFNet
# ============================================================================
# Huang, Romero, Ceylan y Barcelo (NeurIPS 2023) prueban que las C-MPNN -- NBFNet incluido --
# estan EXACTAMENTE caracterizadas por `rawl2` (relational asymmetric local 2-WL), y que un
# clasificador binario es capturado por ellas SI Y SOLO SI es expresable en `rFO3_cnt`.
# `rawl2` construye (u,v) mirando vecinos que cambian solo la SEGUNDA coordenada, o sea
# "caminos de u a v": lo que no se escriba asi, NBFNet no lo expresa con ningun entrenamiento.
#
# Su Teorema 5.3 da la unica salida conocida: **el readout global aumenta ESTRICTAMENTE el
# poder expresivo** (`erFO3_cnt` estrictamente contiene a `rFO3_cnt`). Es un teorema, no una
# conjetura.
#
# ⚠️ MATIZ EMPIRICO QUE IMPORTA (su seccion Q5): el readout global PLANO no ayuda y hasta
# DEGRADA en 2 splits de FB15k-237. El que funciona es el ESPECIFICO POR RELACION -- suma
# separada sobre los nodos con arista entrante/saliente de tipo q -- y les dio estado del arte
# en FB15k-237. En WN18RR (relacionalmente disperso) no aporta. **No implementar la version
# plana**: esta medida como peor.
#
# POR QUE ADEMAS RESUELVE NUESTRA TENSION DE ESCALA (medido 2026-08-26): la ventaja del GT
# vivia a nivel de ARISTA (atencion seleccionando entre la inundacion) y la poda la destruye
# porque descarta aristas. Este canal es a nivel de NODO:
#     atencion por arista : O(B*E*d)  -- el muro, y lo que la poda descarta
#     readout relacional  : O(B*N*d)  -- 14x mas barato en wikikg2, y la poda NO lo toca
# => se puede podar la propagacion (velocidad de A*Net) y que la diferenciacion venga de aca.


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


class SparseGraphTransformer(nn.Module):
    """Stack de capas de atencion sparse por adyacencia + labeling trick + readout."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 attn='softmax',
                 use_rwse=False, rwse_dim=16, use_lappe=False, lappe_dim=16,
                 use_source_rw=False, source_rw_dim=8, edge_drop=0.0,
                 rel_param='diag', rel_rank=4, dependent=False, rel_readout=False):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.self_rel = num_relation
        self.edge_drop = edge_drop
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.use_source_rw = use_source_rw
        self.source_rw_dim = source_rw_dim
        self.attn = attn
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        if use_rwse:
            self.rwse_proj = nn.Linear(rwse_dim, hidden_dim)
        if use_lappe:
            self.lappe_proj = nn.Linear(lappe_dim, hidden_dim)
        # Source-conditioned RW: labeling condicionado a la query (K landing probs
        # desde el head por nodo). NO lleva unsqueeze en forward: ya es (B, N, .).
        if use_source_rw:
            self.source_rw_proj = nn.Linear(source_rw_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SparseRelationalAttentionLayer(hidden_dim, num_heads, num_relation, drop,
                                           attn=attn, rel_param=rel_param,
                                           rel_rank=rel_rank, dependent=dependent)
            for _ in range(num_layer)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        # Canal global relacional: la unica salida PROBADA de la clase rawl2 (ver bloque de
        # docs arriba). Cuesta O(B*N*d) y NO toca aristas => compatible con la poda.
        self.rel_readout = RelationalGlobalReadout(hidden_dim, num_relation) \
            if rel_readout else None

    def forward(self, batched_data):
        h_index = batched_data['h_index']
        r_index = batched_data['r_index']
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        edge_index = apply_edge_dropout(edge_index, self.edge_drop, self.training)
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        # Aumentar con self-loops (cada nodo se atiende a si mismo via relacion self).
        loop = torch.arange(N, device=device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
        rel = torch.cat([rel, torch.full((N,), self.self_rel, device=device)])
        edges = (src, rel, dst)

        x = torch.zeros(B, N, self.hidden_dim, device=device)
        q_emb = self.query_emb(r_index)                  # (B, D)
        x[torch.arange(B, device=device), h_index] = q_emb
        if self.use_rwse:
            x = x + rwse_features(graph, device, self.rwse_dim, self.rwse_proj).unsqueeze(0)
        if self.use_lappe:
            x = x + lappe_features(graph, device, self.lappe_dim, self.lappe_proj,
                                   self.training).unsqueeze(0)
        if self.use_source_rw:
            x = x + source_rw_features(graph, device, self.source_rw_dim,
                                       self.source_rw_proj, h_index)

        x0 = x
        for layer in self.layers:
            x = layer(x, edges, x0=x0, r_index=r_index, q_emb=q_emb)
            if self.attn == 'qc':
                # Residual Bellman-Ford (QC-Exphormer): reinyecta la condicion de
                # frontera en cada capa para que la fuente no pierda la senal de query.
                x = x + x0

        x = self.norm_out(x)
        score = self.readout(x).squeeze(-1)
        if self.rel_readout is not None:
            score = score + self.rel_readout(x, graph, r_index)
        return score




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

# ----------------------------------------------------------------------------
# TIPADO de las aristas del expander -- 3 variantes (`--exp_typing`).
# ----------------------------------------------------------------------------
# El expander introduce conexiones entre nodos que NO existian en el grafo original.
# Como en KGC todo canal deberia llevar una relacion valida, se exploran 3 formas de
# asignarles una:
#   'single' (variante 1, la original): UNA relacion nueva generica R_exp compartida por
#       todas las aristas expander. Fiel a Exphormer (exp_edge_attr = nn.Embedding(1)).
#       Da tipo, pero la MISMA etiqueta para cualquier par: el modelo solo puede aprender
#       un gate global "esto es un atajo", no distinguir atajos utiles de inutiles.
#   'ultra' (variante 2): la arista TOMA PRESTADA una relacion aprendida de forma
#       transferible, construida con la logica de ULTRA (Galkin et al., ICLR 2024):
#       se arma el grafo de relaciones (nodos = tipos de relacion, aristas = las 4
#       interacciones fundamentales hh/tt/ht/th), se corre un GNN sobre el CONDICIONADO
#       a r_q, y la relacion prestada es una mezcla (softmax aprendido) de esas
#       representaciones relativas a la query. Query-dependiente, no arbitraria, y
#       transferible por construccion (solo co-ocurrencia de relaciones, sin identidad
#       de entidad). Nota de diseno: la mezcla es POR QUERY, no por arista -- una arista
#       aleatoria no tiene contexto propio del que derivar una relacion distinta.
#   'path' (variante 3): la relacion se deriva de los CAMINOS REALES que conectan a los
#       dos nodos en el grafo original, componiendo las relaciones a lo largo del camino
#       mas corto (<= exp_path_len saltos): bias = suma de b[r_k], valor = producto
#       DistMult de g[r_k]. Es la mas fiel al principio "toda arista con relacion valida".
#       Los pares sin camino <= exp_path_len caen a R_exp (= variante 1 para esas aristas).
#
# TENSION REGISTRADA (vale para la lectura de resultados): cuanto mas informativa es la
# etiqueta, mas se parece la arista a un camino que el message passing ya puede recorrer
# -- en el extremo 'path' la aleatoriedad del expander, que era todo su valor, deja de
# aportar. Ver SESSION_NOTES 2026-08-03 y lista negra #8 de CLAUDE.md.


def build_relation_graph(edge_index, num_relation):
    """Grafo de RELACIONES de ULTRA (port de ULTRA/ultra/tasks.py::build_relation_graph).

    Nodos = tipos de relacion. Aristas = las 4 interacciones fundamentales entre
    relaciones que comparten entidades: 0=head-head, 1=tail-tail, 2=head-tail,
    3=tail-head. Devuelve (r_src, r_type, r_dst).

    Depende solo de CO-OCURRENCIA de relaciones en entidades (no de identidad de
    entidad) => inductivo-safe, transfiere al grafo de test disjunto. El grafo ya trae
    las inversas como relaciones propias (`data.py::encode_triplets`), que es lo que
    ULTRA asume ("expect the graph is already with inverse edges")."""
    device = edge_index.device
    h, r, t = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
    N = int(torch.max(torch.stack([h.max(), t.max()])).item()) + 1
    R = num_relation

    def _incidence(ent, rel):
        M = torch.zeros(N, R, device=device)
        M[ent, rel] = 1.0                                  # (N, R) incidencia
        deg = M.sum(1, keepdim=True).clamp(min=1.0)        # grado de la entidad
        return M, (M / deg).t()                            # (N,R), (R,N) normalizada
    Mh, MhT = _incidence(h, r)
    Mt, MtT = _incidence(t, r)

    # Ahh = Eh^T Eh, Att = Et^T Et, Aht = Eh^T Et, Ath = Et^T Eh (mismo orden que ULTRA).
    mats = [MhT @ Mh, MtT @ Mt, MhT @ Mt, MtT @ Mh]
    src_l, typ_l, dst_l = [], [], []
    for k, Ak in enumerate(mats):
        idx = Ak.nonzero(as_tuple=False)
        src_l.append(idx[:, 0])
        dst_l.append(idx[:, 1])
        typ_l.append(torch.full((idx.size(0),), k, device=device, dtype=torch.long))
    return torch.cat(src_l), torch.cat(typ_l), torch.cat(dst_l)


class RelationGNN(nn.Module):
    """GNN estilo NBFNet sobre el grafo de relaciones de ULTRA, condicionado a la query.

    Boundary = indicador en el nodo-relacion r_q (ULTRA inicializa la relacion de interes
    con unos y el resto en cero) => las representaciones que devuelve son RELATIVAS a la
    query: (B, R, d). Mensajes DistMult con embedding por tipo de interaccion."""

    def __init__(self, hidden_dim, num_layer=2, num_interaction=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inter_emb = nn.ModuleList(
            [nn.Embedding(num_interaction, hidden_dim) for _ in range(num_layer)])
        self.lin = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layer)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layer)])

    def forward(self, rel_graph, num_relation, r_index):
        r_src, r_typ, r_dst = rel_graph
        B = r_index.size(0)
        device = r_index.device
        h = torch.zeros(B, num_relation, self.hidden_dim, device=device)
        h[torch.arange(B, device=device), r_index] = 1.0          # boundary en r_q
        for emb, lin, norm in zip(self.inter_emb, self.lin, self.norm):
            msg = h[:, r_src, :] * emb(r_typ).unsqueeze(0)        # (B, Er, d) DistMult
            agg = torch.zeros_like(h)
            agg.index_add_(1, r_dst, msg)
            h = h + F.relu(norm(lin(torch.cat([h, agg], dim=-1))))  # short-cut
        return h                                                   # (B, R, d)


def compute_expander_paths(edge_index, num_nodes, exp_src, exp_dst, max_len, exp_rel):
    """Para cada arista expander (u,v), la secuencia de relaciones a lo largo de un camino
    mas corto u->v en el grafo REAL (<= max_len saltos).

    Devuelve (paths (Ex, max_len) long, mask (Ex, max_len) bool). Las aristas cuyo par no
    tiene camino de largo <= max_len se marcan con un solo paso = R_exp, o sea caen al
    comportamiento de la variante 1 para esas aristas. Sin gradiente (es estructura fija);
    se cachea por grafo."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    if num_nodes > 6000:
        raise ValueError(
            f"--exp_typing path reconstruye caminos con una matriz de predecesores "
            f"N x N (N={num_nodes} => ~{num_nodes**2*4/1e9:.1f} GB). Soportado solo en "
            f"grafos chicos (inductivo). Para transductivo hace falta otra estrategia.")

    ei = edge_index.cpu().numpy()
    h, r, t = ei[:, 0], ei[:, 1], ei[:, 2]
    A = csr_matrix((np.ones(len(h)), (h, t)), shape=(num_nodes, num_nodes))
    dist, pred = shortest_path(A, method='D', unweighted=True, directed=True,
                               return_predecessors=True)
    rel_of = {}
    for a, rr, b in zip(h, r, t):
        rel_of.setdefault((int(a), int(b)), int(rr))       # una relacion por par (a,b)

    us, vs = exp_src.cpu().numpy(), exp_dst.cpu().numpy()
    paths = np.full((len(us), max_len), exp_rel, dtype=np.int64)
    mask = np.zeros((len(us), max_len), dtype=bool)
    for i, (u, v) in enumerate(zip(us, vs)):
        u, v = int(u), int(v)
        d = dist[u, v]
        seq = []
        if np.isfinite(d) and 1 <= d <= max_len:
            cur = v
            while cur != u:
                prv = int(pred[u, cur])
                if prv < 0:
                    seq = []
                    break
                seq.append(rel_of[(prv, cur)])
                cur = prv
            seq.reverse()
        if not seq or len(seq) > max_len:
            mask[i, 0] = True                              # sin camino => R_exp
            continue
        paths[i, :len(seq)] = seq
        mask[i, :len(seq)] = True
    return torch.from_numpy(paths), torch.from_numpy(mask)


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


class SparseExpanderGraphTransformer(nn.Module):
    """SparseGraphTransformer aumentado con aristas expander (estilo Exphormer).

    Identico al sparse por adyacencia pero concatenando, a las aristas reales + self-loops,
    un grafo aleatorio d-regular cuyas aristas llevan la relacion sintetica reservada
    R_exp (num_relation+1). Las tablas relacionales de la capa se dimensionan a
    num_relation+2 (self-loop en num_relation, expander en num_relation+1)."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 exp_degree=4, exp_seed=0, attn='softmax',
                 use_rwse=False, rwse_dim=16, use_lappe=False, lappe_dim=16,
                 use_source_rw=False, source_rw_dim=8, edge_drop=0.0,
                 exp_typing='single', exp_path_len=3, exp_rel_layer=2,
                 rel_param='diag', rel_rank=4, dependent=False):
        super().__init__()
        assert exp_typing in ('single', 'ultra', 'path')
        # ultra/path recomponen bias y valor DIAGONAL de las aristas expander
        # (`_exp_rel_params`); la correccion de rango bajo NO se compone, se leeria por
        # indice de la fila R_exp y mezclaria dos semanticas en silencio. Se prohibe.
        assert not (rel_param != 'diag' and exp_typing != 'single'), \
            "rel_param='lowrank' solo esta soportado con exp_typing='single'"
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.self_rel = num_relation          # relacion reservada del self-loop
        self.exp_rel = num_relation + 1        # relacion reservada de las aristas expander
        self.edge_drop = edge_drop
        self.exp_typing = exp_typing
        self.exp_path_len = exp_path_len
        self._relg_cache = {}                  # grafo de relaciones (ULTRA) por grafo
        self._paths_cache = {}                 # caminos expander por grafo
        if exp_typing == 'ultra':
            # GNN de ULTRA sobre el grafo de relaciones + cabeza que elige que relacion
            # "presta" la arista expander (softmax sobre las representaciones relativas).
            self.rel_gnn = RelationGNN(hidden_dim, num_layer=exp_rel_layer)
            self.borrow = nn.Linear(hidden_dim, 1)
        self.exp_degree = exp_degree
        self.exp_seed = exp_seed
        self.attn = attn
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.use_source_rw = use_source_rw
        self.source_rw_dim = source_rw_dim
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        if use_rwse:
            self.rwse_proj = nn.Linear(rwse_dim, hidden_dim)
        if use_lappe:
            self.lappe_proj = nn.Linear(lappe_dim, hidden_dim)
        if use_source_rw:
            self.source_rw_proj = nn.Linear(source_rw_dim, hidden_dim)
        # Se pasa num_relation+1 a la capa => reserva una relacion extra (R_exp) ademas
        # del self-loop: R = (num_relation+1)+1 = num_relation+2 filas en rel_bias/rel_value.
        self.layers = nn.ModuleList([
            SparseRelationalAttentionLayer(hidden_dim, num_heads, num_relation + 1, drop,
                                           attn=attn, rel_param=rel_param,
                                           rel_rank=rel_rank, dependent=dependent)
            for _ in range(num_layer)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self._exp_cache = {}   # num_nodes -> (src, dst) en CPU (generado una vez por grafo)

    def _expander(self, num_nodes, device):
        if num_nodes not in self._exp_cache:
            self._exp_cache[num_nodes] = generate_expander_edges(
                num_nodes, self.exp_degree, self.exp_seed)
        src, dst = self._exp_cache[num_nodes]
        return src.to(device), dst.to(device)

    def forward(self, batched_data):
        h_index = batched_data['h_index']
        r_index = batched_data['r_index']
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        edge_index = apply_edge_dropout(edge_index, self.edge_drop, self.training)
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        # Aumentar con self-loops (relacion self_rel).
        loop = torch.arange(N, device=device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
        rel = torch.cat([rel, torch.full((N,), self.self_rel, device=device)])
        # Aristas expander (relacion sintetica reservada R_exp). No se enmascaran con
        # graph_mask: son estructurales e independientes de la query.
        exp_src, exp_dst = self._expander(N, device)
        exp_start = src.size(0)                # las expander son el tramo FINAL
        src = torch.cat([src, exp_src])
        dst = torch.cat([dst, exp_dst])
        rel = torch.cat([rel, torch.full((exp_src.size(0),), self.exp_rel, device=device)])
        edges = (src, rel, dst)

        # Tipado de las aristas expander (variantes 2 y 3). 'single' no arma contexto:
        # la fila R_exp de las tablas se usa por indice como cualquier otra relacion.
        exp_ctx = None
        if self.exp_typing == 'ultra':
            key = (id(graph), N)
            if key not in self._relg_cache:
                # El grafo de relaciones se arma sobre el grafo COMPLETO del split (no el
                # de este batch): es estructura fija, no depende de la query ni del drop.
                self._relg_cache[key] = build_relation_graph(
                    graph.edge_index.to(device), self.num_relation)
            rel_repr = self.rel_gnn(self._relg_cache[key], self.num_relation, r_index)
            w = torch.softmax(self.borrow(rel_repr).squeeze(-1), dim=-1)   # (B, R)
            exp_ctx = {'typing': 'ultra', 'start': exp_start, 'w': w}
        elif self.exp_typing == 'path':
            key = (id(graph), N)
            if key not in self._paths_cache:
                paths, pmask = compute_expander_paths(
                    graph.edge_index, N, exp_src.cpu(), exp_dst.cpu(),
                    self.exp_path_len, self.exp_rel)
                self._paths_cache[key] = (paths.to(device), pmask.to(device))
            paths, pmask = self._paths_cache[key]
            exp_ctx = {'typing': 'path', 'start': exp_start,
                       'paths': paths, 'mask': pmask}

        x = torch.zeros(B, N, self.hidden_dim, device=device)
        q_emb = self.query_emb(r_index)                  # (B, D)
        x[torch.arange(B, device=device), h_index] = q_emb
        if self.use_rwse:
            x = x + rwse_features(graph, device, self.rwse_dim, self.rwse_proj).unsqueeze(0)
        if self.use_lappe:
            x = x + lappe_features(graph, device, self.lappe_dim, self.lappe_proj,
                                   self.training).unsqueeze(0)
        if self.use_source_rw:
            x = x + source_rw_features(graph, device, self.source_rw_dim,
                                       self.source_rw_proj, h_index)

        x0 = x
        for layer in self.layers:
            x = layer(x, edges, x0=x0, r_index=r_index, exp_ctx=exp_ctx, q_emb=q_emb)
            if self.attn == 'qc':
                # Residual Bellman-Ford (QC-Exphormer): reinyecta la condicion de
                # frontera en cada capa para que la fuente no pierda la senal de query.
                x = x + x0

        x = self.norm_out(x)
        return self.readout(x).squeeze(-1)


# ============================================================================
# Sparse attention con V proveniente de un stream NBFNet (lista negra #6, a proposito).
# ============================================================================
# Variante experimental pedida explicitamente para completar la tabla. Recrea el patron
# refutado de KnowFormer (V-stream separado alimentando la atencion): Q,K salen del stream
# de atencion (labeling trick), pero V sale de las representaciones de nodo de un NBFNet
# corrido aparte (h^L_{u->v}, query-conditioned). La atencion sparse solo REPONDERA por
# adyacencia lo que NBFNet ya calculo => se espera redundante (mejor caso ~= NBFNet). Se
# implementa para tenerlo medido, no porque se espere que gane (ver SESSION_NOTES.md).


class SparseNBFValueLayer(nn.Module):
    """Igual a SparseRelationalAttentionLayer pero V proviene de un tensor externo
    (v_src, el stream NBFNet) en vez del estado x. Q,K siguen saliendo de x."""

    def __init__(self, hidden_dim, num_heads, num_relation, drop):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.self_rel = num_relation
        R = num_relation + 1

        self.norm1 = nn.LayerNorm(hidden_dim)   # Q,K (desde x)
        self.norm_v = nn.LayerNorm(hidden_dim)  # V (desde el stream NBFNet)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        self.rel_bias = nn.Parameter(torch.zeros(num_heads, R))
        self.rel_value = nn.Parameter(torch.ones(num_heads, R, self.head_dim))

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, v_src, edges):
        """x: (B,N,D) estado de atencion (Q,K). v_src: (B,N,D) stream NBFNet (V).
        edges: (src, rel, dst) YA aumentadas con self-loops (rel==self_rel)."""
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        src, rel, dst = edges
        E = src.size(0)

        h = self.norm1(x)
        q = self.to_q(h).view(B, N, H, hd)
        k = self.to_k(h).view(B, N, H, hd)
        v = self.to_v(self.norm_v(v_src)).view(B, N, H, hd)         # V del stream NBFNet

        # Logit por arista: (q_dst . k_src)/sqrt(d) + bias relacional. (B, E, H)
        logit = (q[:, dst] * k[:, src]).sum(-1) * self.scale
        logit = logit + self.rel_bias[:, rel].transpose(0, 1).unsqueeze(0)  # (1,E,H)

        # Segment-softmax por nodo destino (sobre sus aristas entrantes + self-loop).
        idx = dst.view(1, E, 1).expand(B, E, H)
        node_max = logit.new_full((B, N, H), float('-inf'))
        node_max.scatter_reduce_(1, idx, logit, reduce='amax', include_self=False)
        logit = (logit - node_max.gather(1, idx)).exp()
        denom = logit.new_zeros(B, N, H)
        denom.index_add_(1, dst, logit)
        alpha = logit / (denom.gather(1, idx) + 1e-9)              # (B, E, H)
        alpha = self.drop(alpha)

        # Mensaje relacional (composicion DistMult) ponderado por la atencion.
        g = self.rel_value[:, rel, :].permute(1, 0, 2).unsqueeze(0)  # (1, E, H, hd)
        msg = alpha.unsqueeze(-1) * v[:, src] * g
        out = msg.new_zeros(B, N, H, hd)
        out.index_add_(1, dst, msg)                                # (B, N, H, hd)

        out = out.reshape(B, N, D)
        x = x + self.drop(self.to_out(out))
        x = x + self.ffn(self.norm2(x))
        return x


class SparseNBFValueTransformer(nn.Module):
    """Atencion sparse por adyacencia cuyo V proviene de un stream NBFNet separado.

    El stream NBFNet se corre una vez para producir h^L_{u->v} (B,N,d); esas
    representaciones son el V de TODAS las capas de atencion (estilo KnowFormer V-stream).
    Q,K vienen del propio stream de atencion (labeling trick). Ambos streams se entrenan
    end-to-end (el gradiente fluye al NBFNet via V). Mismo readout/eval que el sparse."""

    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 aggregate='pna', short_cut=True, use_rwse=False, rwse_dim=16,
                 use_lappe=False, lappe_dim=16, use_source_rw=False, source_rw_dim=8,
                 edge_drop=0.0):
        super().__init__()
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.self_rel = num_relation
        self.edge_drop = edge_drop
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim
        self.use_lappe = use_lappe
        self.lappe_dim = lappe_dim
        self.use_source_rw = use_source_rw
        self.source_rw_dim = source_rw_dim

        # V-stream: NBFNet completo (se reusa encode() para tomar los estados de nodo).
        # Recibe el mismo edge_drop para no quedar sin regularizar; ojo: sortea su
        # PROPIA mascara, asi que en train los streams Q/K y V ven subgrafos distintos.
        self.nbf = NBFNet(num_relation, num_layer, hidden_dim,
                          aggregate=aggregate, short_cut=short_cut, edge_drop=edge_drop)

        # Q,K-stream: labeling trick propio.
        self.query_emb = nn.Embedding(num_relation, hidden_dim)
        if use_rwse:
            self.rwse_proj = nn.Linear(rwse_dim, hidden_dim)
        if use_lappe:
            self.lappe_proj = nn.Linear(lappe_dim, hidden_dim)
        # Source-conditioned RW: labeling condicionado a la query (K landing probs
        # desde el head por nodo). NO lleva unsqueeze en forward: ya es (B, N, .).
        if use_source_rw:
            self.source_rw_proj = nn.Linear(source_rw_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SparseNBFValueLayer(hidden_dim, num_heads, num_relation, drop)
            for _ in range(num_layer)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batched_data):
        h_index = batched_data['h_index']
        r_index = batched_data['r_index']
        graph = batched_data['graph']
        N = graph.num_nodes
        device = h_index.device
        B = h_index.size(0)

        edge_index = graph.edge_index.to(device)
        if 'graph_mask' in batched_data and batched_data['graph_mask'] is not None:
            edge_index = edge_index[batched_data['graph_mask'].to(device)]
        edge_index = apply_edge_dropout(edge_index, self.edge_drop, self.training)
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        loop = torch.arange(N, device=device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
        rel = torch.cat([rel, torch.full((N,), self.self_rel, device=device)])
        edges = (src, rel, dst)

        # V-stream: representaciones de nodo de NBFNet (query-conditioned). Mismo
        # graph_mask via batched_data => consistente con el stream de atencion.
        v_src = self.nbf.encode(batched_data)                      # (B, N, d)

        # Q,K-stream: labeling trick (+ RWSE / LapPE estructural).
        x = torch.zeros(B, N, self.hidden_dim, device=device)
        x[torch.arange(B, device=device), h_index] = self.query_emb(r_index)
        if self.use_rwse:
            x = x + rwse_features(graph, device, self.rwse_dim, self.rwse_proj).unsqueeze(0)
        if self.use_lappe:
            x = x + lappe_features(graph, device, self.lappe_dim, self.lappe_proj,
                                   self.training).unsqueeze(0)
        if self.use_source_rw:
            x = x + source_rw_features(graph, device, self.source_rw_dim,
                                       self.source_rw_proj, h_index)

        for layer in self.layers:
            x = layer(x, v_src, edges)

        x = self.norm_out(x)
        return self.readout(x).squeeze(-1)

