"""ogbl-wikikg2 con el PROTOCOLO DE EVALUACION OFICIAL DE OGB.

Por que un modulo aparte y no `TransductiveKnowledgeGraph`: en wikikg2 el ranking NO es
full-filtered sobre las 2.5 M entidades. OGB entrega **500 negativos fijos** por triple
(`head_neg` / `tail_neg`) y el MRR se calcula solo contra esos, con su evaluador oficial.
Son metricas distintas: mezclarlas invalida cualquier comparacion contra su tabla o contra
A*Net. Ver `AStarNet/reasoning/task.py:110` (KnowledgeGraphCompletionOGB) y
`AStarNet/reasoning/dataset.py:274` (OGBLKGTest).

Protocolo, portado de su codigo:
  - cada triple de test da DOS items de evaluacion: uno con negativos de COLA y otro con
    negativos de CABEZA (`OGBLKGTest.__len__` = 2 x #triples). Las metricas promedian los dos.
  - en cada item el positivo va en la posicion 0 y los 500 negativos despues.
  - rank = 0.5*(optimista + pesimista) + 1, donde optimista cuenta los negativos con score
    ESTRICTAMENTE mayor y pesimista los con score >=. Es el desempate de OGB (`_eval_mrr`),
    distinto del nuestro (`sum(score >= answer) + 1`): con empates dan valores distintos.

Convencion de relaciones: la del harness, INTERCALADA -- 2r es la relacion r y 2r+1 su
inversa (`train.py:166` hace `where(r % 2 == 0, r+1, r-1)`). num_relation = 2 x 535 = 1070.
"""
import torch
from src.data import Graph


class WikiKG2:
    """N=2 500 604, R=535 (1070 con inversas), train 16.1 M triples (32.2 M aristas)."""

    def __init__(self, path='/home/jreutter/datasets/ogb/wikikg2.pt'):
        D = torch.load(path, map_location='cpu')
        self.num_entity = D['num_nodes']
        R = D['num_relation']
        self.num_relation = 2 * R
        tr = D['train']
        # aristas: (h, 2r, t) y la inversa (t, 2r+1, h)
        self.edge_index = torch.cat([
            torch.stack([tr[:, 0], 2 * tr[:, 1], tr[:, 2]], 1),
            torch.stack([tr[:, 2], 2 * tr[:, 1] + 1, tr[:, 0]], 1)])
        self.train_triplets = torch.stack([tr[:, 0], 2 * tr[:, 1], tr[:, 2]], 1)
        for sp in ('valid', 'test'):
            s = D[sp]
            setattr(self, sp + '_triplets', torch.stack([s[:, 0], 2 * s[:, 1], s[:, 2]], 1))
            setattr(self, sp + '_head_neg', D[sp + '_head_neg'])
            setattr(self, sp + '_tail_neg', D[sp + '_tail_neg'])
        g = Graph(self.edge_index, self.num_entity, self.num_relation)
        self.train_graph = self.valid_graph = self.test_graph = g

        # Clave ordenada para el muestreo de negativos ESTRICTOS sin diccionario Python.
        # ⚠️ Un defaultdict(set) sobre 32.2 M aristas son varios GB de RAM y minutos de
        # construccion. Aca la pertenencia de (h, r, c) es una busqueda binaria sobre un
        # unico int64: (h*R2 + r)*N + t. Cotas: (2.5e6*1070)*2.5e6 = 6.7e15 < 9.2e18. OK.
        N, R2 = self.num_entity, self.num_relation
        k = (self.edge_index[:, 0] * R2 + self.edge_index[:, 1]) * N + self.edge_index[:, 2]
        self.edge_key, _ = k.sort()

    def _is_true(self, h, r, cand):
        """(B,) (B,) (B,k) -> (B,k) bool: cand[i,j] es cola verdadera de (h_i, r_i)?"""
        N, R2 = self.num_entity, self.num_relation
        q = ((h * R2 + r) * N).unsqueeze(1) + cand
        pos = torch.searchsorted(self.edge_key, q.reshape(-1)).clamp(max=self.edge_key.numel() - 1)
        return (self.edge_key[pos] == q.reshape(-1)).view(cand.shape)

    def sample_strict_negatives(self, h, r, k, max_tries=4):
        """Negativos uniformes que NO son colas verdaderas de (h,r), por rechazo.

        ⚠️ NO se construye una mascara (B, N): con N=2.5 M son 80 MB por batch y un
        `multinomial` sobre 2.5 M categorias. Con ~1e-5 de probabilidad de colision, el
        rechazo converge en una o dos pasadas.
        """
        B = h.size(0)
        # ⚠️ A*Net usa num_negative = 1 048 576 en wikikg2 (contra 32 en YAGO/FB15k-237), o sea
        # el 42 % de las 2.5 M entidades como negativos POR POSITIVO. A B=12 eso son 12.6 M
        # indices; se genera en CHUNKS para no picar 100 MB de golpe en el rechazo.
        chunk = max(1, 2_000_000 // max(B, 1))
        out = []
        for c0 in range(0, k, chunk):
            kc = min(chunk, k - c0)
            neg = torch.randint(self.num_entity, (B, kc))
            for _ in range(max_tries):
                bad = self._is_true(h, r, neg)
                if not bad.any():
                    break
                neg = torch.where(bad, torch.randint(self.num_entity, (B, kc)), neg)
            out.append(neg)
        return torch.cat(out, 1) if len(out) > 1 else out[0]

    def train_collate_fn(self, batch):
        """Mitad prediccion de COLA (h, 2r) y mitad de CABEZA (t, 2r+1), como el harness."""
        b = torch.stack(batch, 0)
        n = len(batch) // 2
        h = torch.cat([b[:n, 0], b[n:, 2]])
        r = torch.cat([b[:n, 1], b[n:, 1] + 1])       # 2r par -> 2r+1 impar
        t = torch.cat([b[:n, 2], b[n:, 0]])
        return {'h_index': h, 'r_index': r, 't_index': t,
                'graph': self.train_graph, 'ogb': True}

    def ogb_collate_fn(self, split):
        """Items de evaluacion OGB. Devuelve un collate para `OGBEvalSet`.

        ⚠️ Devuelve una INSTANCIA, no una funcion anidada: bajo DDP los workers del DataLoader
        serializan el collate con pickle, y una closure local no es serializable
        ("Can't pickle local object", job 92691 -- entrenaba bien y moria al validar).
        """
        return _OGBCollate(self, split)


class OGBEvalSet(torch.utils.data.Dataset):
    """2 items por triple: el primero con negativos de cola, el segundo de cabeza.

    Replica `OGBLKGTest` (AStarNet/reasoning/dataset.py:274), incluido el orden: la primera
    mitad del indice son los de cola.

    `subsample`: si es > 0, se evaluan solo esos items, elegidos con una PERMUTACION ALEATORIA
    (semilla `seed`), que es lo que hace A*Net con `fast_test` (util.py:95-101, `random_split`
    con generator 1024). ⚠️ Importa: el test de OGB son 1 197 086 items y evaluarlos todos
    cuesta ~52 h a batch 8, asi que el submuestreo es OBLIGATORIO -- y ellos publican con 5 000.
    Tomar un PREFIJO en vez de una muestra aleatoria sesga: medido sobre los triples de test,
    el prefijo de 2 000 tiene grado mediano de cola 1 939 contra 1 230 del total y le faltan
    dos tercios de las relaciones. A 64 000 items el prefijo ya es representativo (L1 del
    histograma de relaciones 0.030), pero la muestra aleatoria lo es a cualquier tamano.
    """

    def __init__(self, n, subsample=0, seed=1024):
        self.n = n
        self.perm = None
        if subsample and subsample < 2 * n:
            g = torch.Generator().manual_seed(seed)
            self.perm = torch.randperm(2 * n, generator=g)[:subsample]

    def __len__(self):
        return self.perm.numel() if self.perm is not None else 2 * self.n

    def __getitem__(self, i):
        if self.perm is not None:
            i = int(self.perm[i])
        # ⚠️ INTERCALADO (i par = cola, impar = cabeza), NO la primera mitad / segunda mitad
        # que usa `OGBLKGTest`. Con su orden, CUALQUIER PREFIJO del dataset son puros items de
        # COLA: al subsamplear con `limit_val_batches` / `limit_test_batches` se mide solo la
        # mitad facil del problema. Paso de verdad: el job 92695 reporto valid_mrr 0.732 (por
        # ENCIMA del 0.6767 de A*Net) porque sus 40 batches de validacion eran 100 % cola.
        # Ellos evaluan el conjunto COMPLETO, asi que el orden no les importa; a nosotros si.
        # Intercalado, cualquier prefijo queda balanceado 50/50.
        return (i // 2, i % 2 == 0)


def ogb_ranks(score_cand):
    """Rank oficial de OGB: 0.5*(optimista + pesimista) + 1. `score_cand` (B, 1+k), gold en 0.

    Verbatim de ogb.linkproppred.Evaluator._eval_mrr (rama torch):
      optimista = #negativos con score ESTRICTAMENTE mayor  (el positivo va primero al empatar)
      pesimista = #negativos con score >=                   (el positivo va ultimo al empatar)
    ⚠️ Distinto del ranking del harness (`sum(score >= answer) + 1`), que es el pesimista.
    Con empates dan numeros distintos, asi que NO son intercambiables.
    """
    pos = score_cand[:, :1]
    neg = score_cand[:, 1:]
    optimistic = (neg > pos).sum(dim=1)
    pessimistic = (neg >= pos).sum(dim=1)
    return 0.5 * (optimistic + pessimistic).float() + 1


class _OGBCollate:
    """Collate de evaluacion OGB, picklable (ver `WikiKG2.ogb_collate_fn`)."""

    def __init__(self, data, split):
        self.graph = data.train_graph
        self.trip = getattr(data, split + '_triplets')
        self.hneg = getattr(data, split + '_head_neg')
        self.tneg = getattr(data, split + '_tail_neg')

    def __call__(self, batch):
        idx = torch.tensor([i for i, _ in batch])
        is_tail = torch.tensor([b for _, b in batch])
        tr = self.trip[idx]
        h = torch.where(is_tail, tr[:, 0], tr[:, 2])
        r = torch.where(is_tail, tr[:, 1], tr[:, 1] + 1)   # cabeza => relacion inversa (impar)
        gold = torch.where(is_tail, tr[:, 2], tr[:, 0])
        neg = torch.where(is_tail.unsqueeze(1), self.tneg[idx], self.hneg[idx])
        cand = torch.cat([gold.unsqueeze(1), neg], 1)      # positivo en la columna 0
        return {'h_index': h, 'r_index': r, 't_index': gold, 'cand': cand,
                'graph': self.graph, 'ogb': True}
