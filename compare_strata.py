"""Compara modelos ESTRATIFICADO por grado de la respuesta correcta.

Eje del paper (SESSION_NOTES 2026-08-22 b): el modo de fallo dominante es la cola larga
(MRR 0.23 en el decil de grado mas bajo vs 0.92 en el mas alto), y el MRR agregado lo
esconde. La pregunta que decide si hay reclamo arquitectonico:

  ¿la atencion difiere de la agregacion fija EN ALGUN ESTRATO, o es uniformemente peor?

Solo el primer caso es accionable: si gana en un estrato y pierde en otro, un hibrido con
compuerta condicionada al grado tiene motivo medido. Si es uniformemente peor, no hay nada
que conservar ni que agregar.

Estratifica ademas por grado de la FUENTE, que es la otra mitad del perfil de fallo
(fuente hub -> respuesta long-tail).

Uso:
  python compare_strata.py astarnet:astarnet_ranks_bias_s1024_test.pkl \
                           gt:gt_ranks_fb237_s42_partial.pt
"""
import pickle
import sys
import numpy as np
import torch
from scipy.sparse import csr_matrix


def load(spec):
    name, path = spec.split(':', 1)
    if path.endswith('.pkl'):                      # volcado de A*Net (torchdrug)
        d = pickle.load(open(path, 'rb'))
        htr = d['htr'].numpy()
        rk = d['ranking'].numpy()                  # (T,2) = [predecir cola, cabeza]
        E = d['train_edges'].numpy()[:, [0, 1]]
        # fuente/respuesta por direccion
        src = np.stack([htr[:, 0], htr[:, 1]], 1).reshape(-1)
        ans = np.stack([htr[:, 1], htr[:, 0]], 1).reshape(-1)
        rk = rk.reshape(-1)
        N = int(d['num_entity'])
    else:                                          # volcado de nuestro harness
        d = torch.load(path)
        rows = d['rows'].numpy()                   # (Q,4) = (h, r, t, rank)
        src, ans, rk = rows[:, 0], rows[:, 2], rows[:, 3]
        E = d['edge_index'].numpy()[:, [0, 2]]     # (h, r, t) -> (h, t)
        N = int(max(E.max(), src.max(), ans.max()) + 1)
    rows_ = np.concatenate([E[:, 0], E[:, 1]])
    cols_ = np.concatenate([E[:, 1], E[:, 0]])
    A = csr_matrix((np.ones(len(rows_), np.int8), (rows_, cols_)), shape=(N, N))
    A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel()
    return name, dict(rk=rk.astype(float), deg_src=deg[src], deg_ans=deg[ans])


runs = dict(load(a) for a in sys.argv[1:])
if not runs:
    sys.exit('uso: compare_strata.py nombre:archivo [nombre:archivo ...]')

for n, r in runs.items():
    print(f'{n}: {len(r["rk"])} queries | MRR {np.mean(1/r["rk"]):.4f} '
          f'H@10 {np.mean(r["rk"]<=10):.4f}')

# Cortes FIJOS (no cuantiles) para que los estratos sean los mismos entre modelos
CUTS = [(0, 14), (15, 27), (28, 43), (44, 99), (100, 249), (250, 10**9)]
for key, lbl in (('deg_ans', 'GRADO DE LA RESPUESTA'), ('deg_src', 'GRADO DE LA FUENTE')):
    print(f'\n{lbl} -- MRR por estrato')
    hdr = f"{'estrato':>14} {'n':>7}" + ''.join(f'{n:>12}' for n in runs)
    print(hdr)
    for lo, hi in CUTS:
        lbl_s = f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'
        line = f'{lbl_s:>14}'
        n_shown = None
        vals = []
        for n, r in runs.items():
            m = (r[key] >= lo) & (r[key] <= hi)
            n_shown = int(m.sum()) if n_shown is None else n_shown
            vals.append(np.mean(1 / r['rk'][m]) if m.sum() else float('nan'))
        print(line + f'{n_shown:>7}' + ''.join(f'{v:>12.4f}' for v in vals))
    if len(runs) == 2:
        a, b = list(runs)
        print(f'  delta ({b} - {a}) por estrato:')
        for lo, hi in CUTS:
            ma = (runs[a][key] >= lo) & (runs[a][key] <= hi)
            mb = (runs[b][key] >= lo) & (runs[b][key] <= hi)
            if ma.sum() and mb.sum():
                d = np.mean(1/runs[b]['rk'][mb]) - np.mean(1/runs[a]['rk'][ma])
                l2 = f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'
                print(f'    {l2:>12}: {d:+.4f}')
