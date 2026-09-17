"""Que comparten las queries donde A*Net falla?

Cruza el rank de cada query de test con la ESTRUCTURA del grafo de train que el modelo
ve. Pregunta del usuario (2026-08-22): el 42 % de las queries no tiene la respuesta en
top-10 y el 68 % no la tiene en el puesto 1 -- que tienen en comun?

Features por query, todas calculadas del grafo de train (fact_graph):
  - dist(h,t)  : distancia mas corta NO dirigida (el modelo agrega relaciones inversas,
                 asi que propaga en ambos sentidos). ES LA CLAVE: con num_layer=6, un par
                 a distancia > 6 es INALCANZABLE por composicion de caminos => ningun
                 modelo path-based con L=6 puede resolverlo, y ningun reranker tampoco.
  - grado de la FUENTE de la query y grado de la RESPUESTA correcta
  - n_vecinos_comunes(h,t) : caminos de largo 2, evidencia composicional mas barata
  - frecuencia de la relacion en train
"""
import pickle
import sys
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

d = pickle.load(open(sys.argv[1] if len(sys.argv) > 1
                     else 'astarnet_ranks_full_s1024_test.pkl', 'rb'))
rank = d['ranking'].numpy()                 # (T,2)
htr = d['htr'].numpy()                      # (T,3)
E = d['train_edges'].numpy()                # (M,3) = (h,t,r)
N = int(d['num_entity'])
T = len(htr)
print(f'test: {T} triples ({2*T} queries) | grafo de train: {len(E)} aristas, N={N}\n')

# --- grafo no dirigido de train (el modelo agrega inversas) ---
rows = np.concatenate([E[:, 0], E[:, 1]])
cols = np.concatenate([E[:, 1], E[:, 0]])
A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(N, N))
A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel()
rel_freq = np.bincount(E[:, 2], minlength=int(d['num_relation']))

# --- distancia h-t, BFS por chunks de fuentes ---
h_all, t_all = htr[:, 0], htr[:, 1]
srcs = np.unique(h_all)
pos = {s: i for i, s in enumerate(srcs)}
dist = np.full(T, np.inf)
CH = 400
for i in range(0, len(srcs), CH):
    chunk = srcs[i:i + CH]
    D = shortest_path(A, method='D', unweighted=True, indices=chunk)
    m = np.isin(h_all, chunk)
    idx = np.array([np.where(chunk == x)[0][0] for x in h_all[m]])
    dist[m] = D[idx, t_all[m]]
    print(f'  BFS {min(i+CH, len(srcs))}/{len(srcs)} fuentes', end='\r', flush=True)
print(' ' * 40, end='\r')

# vecinos comunes (caminos de largo 2)
common = np.asarray([A[h].multiply(A[t]).sum() for h, t in zip(h_all, t_all)])

# --- por direccion: col 0 = predecir cola (fuente h), col 1 = predecir cabeza (fuente t) ---
rk = rank.reshape(-1)
dist2 = np.repeat(dist, 2)
common2 = np.repeat(common, 2)
relf2 = np.repeat(rel_freq[htr[:, 2]], 2)
src_deg = np.stack([deg[h_all], deg[t_all]], 1).reshape(-1)     # grado de la fuente
ans_deg = np.stack([deg[t_all], deg[h_all]], 1).reshape(-1)     # grado de la respuesta

buckets = [('rank 1', rk == 1), ('rank 2-10', (rk >= 2) & (rk <= 10)),
           ('rank 11-100', (rk >= 11) & (rk <= 100)), ('rank >100', rk > 100)]

print(f"{'bucket':>12} {'queries':>8} {'%':>6} {'dist mediana':>13} {'%d>6':>7} "
      f"{'%inalc':>7} {'gr.fuente':>10} {'gr.resp':>9} {'vec.com':>9} {'frec.rel':>9}")
for name, m in buckets:
    n = int(m.sum())
    dd = dist2[m]
    fin = dd[np.isfinite(dd)]
    print(f'{name:>12} {n:>8} {100*n/len(rk):>5.1f}% '
          f'{np.median(fin) if len(fin) else float("nan"):>13.1f} '
          f'{100*np.mean(fin > 6) if len(fin) else 0:>6.1f}% '
          f'{100*np.mean(~np.isfinite(dd)):>6.1f}% '
          f'{np.median(src_deg[m]):>10.0f} {np.median(ans_deg[m]):>9.0f} '
          f'{np.median(common2[m]):>9.0f} {np.median(relf2[m]):>9.0f}')

print('\nDISTRIBUCION DE DISTANCIA (todas las queries) y MRR condicional')
for lo, hi, lbl in [(1, 1, 'd=1 (arista directa)'), (2, 2, 'd=2'), (3, 3, 'd=3'),
                    (4, 6, 'd=4-6'), (7, 99, 'd>6 (fuera del horizonte L=6)')]:
    m = np.isfinite(dist2) & (dist2 >= lo) & (dist2 <= hi)
    n = int(m.sum())
    if n:
        print(f'  {lbl:>32}: {n:>6} ({100*n/len(rk):>4.1f}%)  '
              f'MRR {np.mean(1/rk[m]):.4f}  H@1 {np.mean(rk[m]<=1):.4f}  '
              f'H@10 {np.mean(rk[m]<=10):.4f}')
m = ~np.isfinite(dist2)
if m.sum():
    print(f'  {"inalcanzable":>32}: {int(m.sum()):>6} ({100*m.mean():>4.1f}%)  '
          f'MRR {np.mean(1/rk[m]):.4f}  H@1 {np.mean(rk[m]<=1):.4f}  '
          f'H@10 {np.mean(rk[m]<=10):.4f}')

print('\nCUANTO DEL FALLO ES ESTRUCTURAL (fuera del horizonte o inalcanzable)')
fail10 = rk > 10
oob = (~np.isfinite(dist2)) | (dist2 > 6)
print(f'  queries con la respuesta FUERA de top-10: {int(fail10.sum())} ({100*fail10.mean():.1f}%)')
print(f'  de esas, cuantas son d>6 o inalcanzables: {100*np.mean(oob[fail10]):.1f}%')
print(f'  (base: en el total de queries, d>6/inalcanzable es {100*oob.mean():.1f}%)')
