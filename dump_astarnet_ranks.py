"""Vuelca, por query de test, el RANK full-filtered y la RELACION de A*Net.

Para que: gate barato de la OPCION B (reranking sobre A*Net, SESSION_NOTES 2026-08-22).
Con rank+relacion por query se calcula, SIN ENTRENAR NADA:
  (1) el techo oraculo de cualquier reranker sobre top-K (si el rank verdadero es <= K,
      un reranker perfecto lo lleva a 1; si no, no lo puede tocar), y
  (2) ese techo desglosado por CARDINALIDAD de la relacion (1-1 / 1-N / N-1 / N-N),
      que es donde la hipotesis de exclusion mutua predice que esta el margen.

Si el margen es plano entre cardinalidades, la opcion B queda muerta y la unica ruta a
wikikg2 es la opcion A (poda aprendida dentro del GT).

Mismo wrapper que run_astarnet.py (PyG antes que torchdrug + parche is_meta); lo unico
extra es un monkey-patch de la task para capturar `pos_r_index` y el `ranking`.
NO toca el repo AStarNet.
"""
import os
import runpy
import sys
import pickle

import torch_geometric.data  # noqa: F401  PyG ANTES que torchdrug (ver run_astarnet.py)

from torchdrug.data import Graph as _TDGraph  # noqa: E402
if not isinstance(getattr(_TDGraph, 'is_meta', None), bool):
    _TDGraph.is_meta = False

import torch  # noqa: E402
from torchdrug import tasks  # noqa: E402

OUT = os.environ.get('DUMP_OUT', '/home/jreutter/Attention/astarnet_ranks.pkl')
KGC = tasks.KnowledgeGraphCompletion

_rels = []          # pos_r_index por batch, en orden del dataloader
_state = {}

_orig_predict = KGC.predict
_orig_evaluate = KGC.evaluate


def predict(self, batch, all_loss=None, metric=None):
    # all_loss is None <=> estamos en evaluacion (ver reasoning.py:44)
    if all_loss is None:
        h, t, r = batch.t()
        _rels.append(torch.stack([h, t, r], dim=1).detach().cpu())   # (B,3)
    return _orig_predict(self, batch, all_loss, metric)


def evaluate(self, pred, target):
    mask, tgt = target
    # Replica exacta de reasoning.py:197-201 (filtered ranking).
    pos_pred = pred.gather(-1, tgt.unsqueeze(-1))
    ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1     # (Q, 2) = [tail, head]

    htr = torch.cat(_rels) if _rels else None
    rel = htr[:, 2] if htr is not None else None

    # Top-10 candidatos que el modelo pone arriba EXCLUYENDO todas las respuestas
    # verdaderas (mask=0 en ellas) => son los DISTRACTORES mejor rankeados. Con eso se
    # separa "escasez de evidencia" de "sesgo de popularidad": si el grado de esos
    # distractores es sistematicamente mayor que el de la respuesta correcta, el modelo
    # prefiere hubs mas alla de lo que la evidencia justifica.
    # Por chunks: pred es (Q,2,N) y un masked_fill completo duplicaria ~2.4 GB.
    tops = []
    for i in range(0, len(pred), 2048):
        pc = pred[i:i + 2048].clone()
        pc[~mask[i:i + 2048]] = float('-inf')
        tops.append(pc.topk(10, dim=-1).indices)
        del pc
    top10 = torch.cat(tops)                        # (Q,2,10)
    split = getattr(self, 'split', 'unknown')
    _rels.clear()                      # el engine evalua valid y DESPUES test: no mezclar
    if rel is not None and len(rel) == len(ranking):
        # Cardinalidad por relacion, calculada del grafo de TRAIN (fact_graph), que es
        # el unico que el modelo vio. Convencion estandar (TransE): tph = colas medias
        # por (h,r); hpt = cabezas medias por (r,t); umbral 1.5.
        el = self.fact_graph.edge_list.cpu()                        # (E,3) = (h,t,r)
        h, t, r = el[:, 0], el[:, 1], el[:, 2]
        R = int(self.num_relation)
        n_triples = torch.zeros(R)
        n_hr = torch.zeros(R)
        n_tr = torch.zeros(R)
        for rr in range(R):
            m = r == rr
            if m.sum() == 0:
                continue
            n_triples[rr] = m.sum()
            n_hr[rr] = len(torch.unique(h[m]))
            n_tr[rr] = len(torch.unique(t[m]))
        tph = n_triples / n_hr.clamp(min=1)    # colas por cabeza
        hpt = n_triples / n_tr.clamp(min=1)    # cabezas por cola

        out = OUT.replace('.pkl', f'_{split}.pkl')
        with open(out, 'wb') as f:
            pickle.dump({
                'ranking': ranking.cpu(),      # (Q,2): col 0 = predecir cola, col 1 = cabeza
                'relation': rel,               # (Q,)
                'htr': htr,                    # (Q,3) = (h, t, r) de cada triple de test
                # Grafo de TRAIN que el modelo ve en test (fact_graph), para poder cruzar
                # los fallos con estructura: distancia h-t, grados, frecuencia de relacion.
                'train_edges': self.fact_graph.edge_list.cpu(),
                'top10_distractors': top10.cpu(),   # (Q,2,10) sin respuestas verdaderas
                'tph': tph, 'hpt': hpt,
                'n_triples': n_triples,
                'num_entity': int(self.num_entity),
                'num_relation': R,
                'split': split,
            }, f)
        print(f'[dump] {out}  split={split}  Q={len(ranking)}  R={R}', flush=True)
    return _orig_evaluate(self, pred, target)


KGC.predict = predict
KGC.evaluate = evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'AStarNet')
if 'CUDA_HOME' not in os.environ and 'CONDA_PREFIX' in os.environ:
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
sys.path.insert(0, REPO)
os.chdir(REPO)
sys.argv[0] = os.path.join(REPO, 'script', 'run.py')
runpy.run_path(sys.argv[0], run_name='__main__')
