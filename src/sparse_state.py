"""Estructuras para propagacion con ESTADO DISPERSO de nodos.

Motivo (DISENO_GT_PODA.md, adenda 2026-08-26): el muro de escala no es el numero de aristas
sino el ESTADO DENSO de nodos. Medido en YAGO3-10: 4.1x de aristas pero 8.5x de nodos dio
10.9x mas lento y 21.8 GB. Extrapolado a wikikg2 (N=2.5 M) el estado (B,N,d) retenido por 6
capas son 15.6 GB, antes de la atencion.

Aca los nodos "activos" de cada query viven en SLOTS de tamano fijo (B,S) y el estado es
(B,S,d). Con S=30 000 en wikikg2 son 30 MB en vez de 15.6 GB.

⚠️ REGLA: NUNCA materializar un tensor (B, N). Es exactamente lo que se esta evitando. El
mapeo global->slot va con `searchsorted` sobre el conjunto activo ORDENADO.
"""
import torch


class CSRGraph:
    """Adyacencia en CSR ordenada por nodo FUENTE, para sacar las salientes de un conjunto
    de nodos sin recorrer las E aristas.

    Se construye UNA vez por grafo (se cachea por id), no por batch.
    """

    _cache = {}

    def __init__(self, edge_index, num_nodes, device):
        # edge_index: (E,3) = (src, rel, dst)
        src, rel, dst = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
        order = torch.argsort(src)
        # Se guarda la permutacion para poder traducir mascaras del orden ORIGINAL al de CSR.
        # Es lo que permite construir el CSR UNA vez sobre el grafo completo y aplicar la
        # mascara por batch despues, en vez de reconstruirlo (ver nota de cache abajo).
        self.order = order
        self.src = src[order].contiguous()
        self.rel = rel[order].contiguous()
        self.dst = dst[order].contiguous()
        self.num_nodes = num_nodes
        deg = torch.bincount(self.src, minlength=num_nodes)
        self.indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        torch.cumsum(deg, 0, out=self.indptr[1:])
        self.degree = deg

    @classmethod
    def get(cls, edge_index, num_nodes, device, key=None):
        """⚠️ La clave NO debe depender del numero de aristas del batch.

        `graph_mask` cambia por batch (quita las aristas de las queries), asi que una clave
        con `edge_index.shape[0]` construye y CACHEA un CSR nuevo en cada paso: en YAGO son
        ~91 MB cada uno y la memoria crecio de 6.8 a 20.7 GB hasta reventar (job 92427).
        El CSR se construye sobre el grafo COMPLETO y la mascara se aplica despues, al
        recolectar candidatas, usando `self.order`.
        """
        k = (key, num_nodes)
        if k not in cls._cache:
            if len(cls._cache) > 4:          # techo defensivo
                cls._cache.clear()
            cls._cache[k] = CSRGraph(edge_index, num_nodes, device)
        return cls._cache[k]

    def out_edges(self, nodes):
        """Aristas salientes de `nodes` (n,) -> indices de arista (m,) y a que nodo de
        `nodes` pertenece cada una (m,). Sin bucles: repeat_interleave sobre los grados."""
        deg = self.degree[nodes]                                  # (n,)
        total = int(deg.sum())
        if total == 0:
            z = nodes.new_zeros(0)
            return z, z
        owner = torch.repeat_interleave(torch.arange(len(nodes), device=nodes.device), deg)
        # posicion dentro del tramo de cada nodo
        offs = torch.cumsum(deg, 0) - deg                         # inicio por nodo
        within = torch.arange(total, device=nodes.device) - offs[owner]
        eidx = self.indptr[nodes][owner] + within
        return eidx, owner


def global_to_slot(active_sorted, ids):
    """Mapea ids globales a slots usando busqueda binaria sobre el activo ORDENADO.

    active_sorted: (B,S) ids globales ordenados por fila (relleno con num_nodes+1 al final).
    ids: (B,K) ids globales a mapear.
    Devuelve (slot, found): slot (B,K) y mascara (B,K) de si el id estaba activo.

    ⚠️ Se usa searchsorted a proposito: la alternativa obvia -- un tensor (B,N) de posiciones --
    es justamente el objeto que hace inviable wikikg2 (160 MB solo en int64, y crece con N).
    """
    slot = torch.searchsorted(active_sorted, ids)
    slot = slot.clamp(max=active_sorted.size(1) - 1)
    found = active_sorted.gather(1, slot) == ids
    return slot, found


# ============================================================================
# TOP-K SIN PADDING (Alg. 2 de A*Net). PORTADO DE `AStarNet/reasoning/functional.py`.
# ============================================================================
# Motivo (2026-08-30): armar el pool de candidatas como una matriz RECTANGULAR (B, K*cap)
# obliga a un tope `cap` de salientes POR NODO, y ese tope se tomaba como las PRIMERAS `cap`
# aristas de la fila CSR -- o sea en el orden en que venian en el archivo del dataset. No es
# computo desperdiciado: DESCARTA EVIDENCIA de forma arbitraria. Medido en YAGO3-10, con todo
# lo demas igual: cap 64 -> MRR 0.4498, cap 1024 -> 0.5478 (+0.098, sin saturar).
#
# A*Net no tiene tope por nodo: `neighbors()` (reasoning/data.py:276) devuelve TODAS las
# salientes de los nodos seleccionados y el unico recorte es GLOBAL, via `variadic_topks`.
# Para poder hacerlo sin padding concatenan el batch en un tensor plano con un id de muestra
# por elemento y convierten el top-k por muestra en un ORDENAMIENTO MULTI-CLAVE.
#
# ⚠️ Portado del CODIGO, no de la descripcion en prosa. Es la leccion del 2026-08-26 (tres
# intentos fallidos de poda por implementar su seleccion desde el texto del paper).

def multikey_argsort(inputs, descending=False, break_tie=False):
    """Orden lexicografico por varias claves. Verbatim de AStarNet/reasoning/functional.py:4.

    Ordena por la ULTIMA clave primero y sube; como cada `argsort` es ESTABLE, al terminar
    el resultado esta ordenado por la primera clave y, dentro de cada grupo, por la segunda.
    """
    if break_tie:
        order = torch.randperm(len(inputs[0]), device=inputs[0].device)
    else:
        order = torch.arange(len(inputs[0]), device=inputs[0].device)
    for key in inputs[::-1]:
        index = key[order].argsort(stable=True, descending=descending)
        order = order[index]
    return order


def variadic_topks(input, size, ks, largest=True, break_tie=False):
    """Top-k POR MUESTRA sobre un tensor plano de segmentos de largo variable.

    Verbatim de AStarNet/reasoning/functional.py:30.
      input: (total,) valores de todas las muestras concatenados.
      size:  (B,) largo del segmento de cada muestra.  ks: (B,) cuantos tomar de cada una.
    Devuelve (valores, indices) con `ks.sum()` elementos, agrupados por muestra.

    Como funciona el juego de offsets: tras el multikey sort, la muestra b ocupa el bloque
    [size[:b].sum(), size[:b+1].sum()) ya ordenado por valor descendente, y sus mejores ks[b]
    son los PRIMEROS de ese bloque. `offset` convierte un contador global 0..ks.sum() en la
    posicion absoluta del bloque correspondiente.
    """
    index2sample = torch.repeat_interleave(size)
    if largest:
        index2sample = -index2sample
    order = multikey_argsort((index2sample, input), descending=largest, break_tie=break_tie)

    range = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range = range + offset.repeat_interleave(ks)
    index = order[range]

    return input[index], index


def global_to_slot_flat(active, ids, qid, num_nodes):
    """Como `global_to_slot` pero para un tensor PLANO de ids con su query id por elemento.

    Lo necesita la ruta sin padding: ahi las candidatas no forman una matriz (B,M) sino un
    unico arreglo (total,) donde cada elemento sabe a que query pertenece.

    Truco: se codifica (query, nodo) en una sola clave `q*(N+1) + nodo`. Como `active` esta
    ordenado DENTRO de cada fila y el maximo de la fila b es b*(N+1)+N < (b+1)*(N+1), el
    aplanado queda GLOBALMENTE ordenado y basta un `searchsorted` 1-D.
    ⚠️ Se sigue sin materializar ningun (B,N): la clave es (total,) y el buscado (B*S,).
    """
    B, S = active.shape
    base = torch.arange(B, device=active.device).unsqueeze(1) * (num_nodes + 1)
    flat_sorted = (active + base).reshape(-1)
    key = ids + qid * (num_nodes + 1)
    pos = torch.searchsorted(flat_sorted, key).clamp(max=B * S - 1)
    found = flat_sorted[pos] == key
    slot = (pos - qid * S).clamp(0, S - 1)
    return slot, found


@torch.no_grad()
def personalized_pagerank(edge_index, num_nodes, src, alpha=0.8, num_iteration=20):
    """PPR desde `src` (B,), portado de `AStarNet/reasoning/data.py:293`.

    Devuelve (B, N) float. NO lleva gradiente: lo aprendido es el embedding del BIN, no el PPR.

    Fiel a su implementacion:
      - peso de arista 1/grado_saliente(src)  (`edge_weight / degree_in[node_in]`; en torchdrug
        `node_in` es la FUENTE y `degree_in` cuenta aristas donde el nodo es fuente);
      - `spmm` con index [node_out, node_in] => propaga de la fuente al destino;
      - `ppr = ppr*alpha + init*(1-alpha)` con alpha 0.8, 20 iteraciones. Ojo: aca `alpha` pesa
        la PROPAGACION, no el reinicio (al reves de la convencion habitual de PPR).

    ⚠️ (B, N) es el UNICO tensor denso sobre N que se materializa. A N=2.5 M y B=12 son 120 MB,
    una sola vez por batch (no por capa), asi que no rompe el diseno de estado disperso.
    """
    dev = edge_index.device
    s, d = edge_index[:, 0], edge_index[:, 2]
    outdeg = torch.bincount(s, minlength=num_nodes).clamp(min=1).float()
    w = (1.0 / outdeg[s]).unsqueeze(1)                       # (E,1)
    B = src.numel()
    init = torch.zeros(num_nodes, B, device=dev)
    init[src, torch.arange(B, device=dev)] = 1.0
    ppr = init
    for _ in range(num_iteration):
        nxt = torch.zeros_like(ppr)
        nxt.index_add_(0, d, ppr[s] * w)
        ppr = nxt * alpha + init * (1 - alpha)
    return ppr.t().contiguous()                              # (B, N)


def ppr_bins(ppr, num_nodes, num_bin=10):
    """Discretiza el PPR en `num_bin` bins logaritmicos, como ellos (`model.py:356`).

    `torch.logspace(-1, 0, num_bin, base=N)` = N^linspace(-1,0), o sea de 1/N a 1.
    ⚠️ `bucketize` puede devolver `num_bin` si el valor supera todas las cotas; se clampa
    porque el `nn.Embedding` tiene solo `num_bin` filas.
    """
    b = torch.logspace(-1, 0, num_bin, base=num_nodes, device=ppr.device)
    return torch.bucketize(ppr, b).clamp(max=num_bin - 1).to(torch.uint8)
