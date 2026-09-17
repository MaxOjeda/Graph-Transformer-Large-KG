"""Escasez de evidencia o SESGO DE POPULARIDAD?

El analisis de fallos (2026-08-22) mostro que el grado de la respuesta correcta cae de 156
(rank 1) a 26 (rank >100) mientras el de la fuente sube de 33 a 102. Eso es ASOCIACION.
Hay dos lecturas incompatibles y este script las separa:

  (a) ESCASEZ DE EVIDENCIA: las entidades de bajo grado son genuinamente mas dificiles
      (menos caminos que las conecten) => limite de informacion, poca palanca.
  (b) SESGO DE POPULARIDAD: el modelo pone hubs arriba MAS ALLA de lo que la evidencia
      justifica => hay margen real y es atacable.

Test: comparar el grado de los DISTRACTORES que el modelo rankea por encima de la
respuesta correcta contra el grado de la respuesta correcta. Bajo (a) los distractores
deberian tener grado parecido al de la respuesta; bajo (b), sistematicamente mayor.
"""
import pickle
import sys
import numpy as np
from scipy.sparse import csr_matrix

d = pickle.load(open(sys.argv[1] if len(sys.argv) > 1
                     else 'astarnet_ranks_bias_s1024_test.pkl', 'rb'))
rank = d['ranking'].numpy()
htr = d['htr'].numpy()
E = d['train_edges'].numpy()
top10 = d['top10_distractors'].numpy()          # (T,2,10)
N = int(d['num_entity'])

rows = np.concatenate([E[:, 0], E[:, 1]])
cols = np.concatenate([E[:, 1], E[:, 0]])
A = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(N, N))
A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel()

# respuesta correcta por direccion: col 0 predice cola (t), col 1 predice cabeza (h)
ans = np.stack([htr[:, 1], htr[:, 0]], 1).reshape(-1)
rk = rank.reshape(-1)
top1 = top10[:, :, 0].reshape(-1)               # mejor distractor
deg_ans, deg_top1 = deg[ans], deg[top1]
deg_top10 = deg[top10.reshape(-1, 10)]          # (Q,10)

print(f'Q = {len(rk)} queries\n')
print('GRADO: respuesta correcta vs distractores mejor rankeados')
print(f"{'':>26} {'mediana':>9} {'media':>9}")
print(f"{'respuesta correcta':>26} {np.median(deg_ans):>9.0f} {deg_ans.mean():>9.1f}")
print(f"{'mejor distractor':>26} {np.median(deg_top1):>9.0f} {deg_top1.mean():>9.1f}")
print(f"{'top-10 distractores':>26} {np.median(deg_top10):>9.0f} {deg_top10.mean():>9.1f}")
print(f"{'entidad promedio del KG':>26} {np.median(deg):>9.0f} {deg.mean():>9.1f}")

print('\nEl distractor #1 tiene MAS grado que la respuesta correcta?')
for name, m in [('todas las queries', np.ones(len(rk), bool)),
                ('donde el modelo ACIERTA (rank 1)', rk == 1),
                ('donde FALLA (rank 2-10)', (rk >= 2) & (rk <= 10)),
                ('donde FALLA (rank >10)', rk > 10)]:
    n = int(m.sum())
    if not n:
        continue
    frac = np.mean(deg_top1[m] > deg_ans[m])
    ratio = np.median(deg_top1[m] / np.maximum(deg_ans[m], 1))
    print(f'  {name:>34}: {100*frac:>5.1f}% de las veces | '
          f'ratio mediano grado(distr)/grado(resp) = {ratio:.2f}')

print('\nMRR por grado de la RESPUESTA correcta (decilas)')
qs = np.quantile(deg_ans, np.linspace(0, 1, 11))
for i in range(10):
    lo, hi = qs[i], qs[i + 1]
    m = (deg_ans >= lo) & (deg_ans <= hi if i == 9 else deg_ans < hi)
    if m.sum():
        print(f'  decil {i+1:>2} (grado {lo:>5.0f}-{hi:>6.0f}): n={int(m.sum()):>6}  '
              f'MRR {np.mean(1/rk[m]):.4f}  H@10 {np.mean(rk[m]<=10):.4f}')

print('\nLECTURA')
frac_all = np.mean(deg_top1 > deg_ans)
ratio_all = np.median(deg_top1 / np.maximum(deg_ans, 1))
if frac_all > 0.6 and ratio_all > 1.3:
    print(f'  => SESGO DE POPULARIDAD: el distractor top-1 tiene mas grado que la respuesta')
    print(f'     el {100*frac_all:.0f}% de las veces (ratio mediano {ratio_all:.2f}).')
    print('     Hay palanca: debiasing de grado es atacable arquitecturalmente.')
elif frac_all < 0.55:
    print(f'  => NO hay sesgo de popularidad claro ({100*frac_all:.0f}%, ratio {ratio_all:.2f}).')
    print('     El fallo en la cola larga es ESCASEZ DE EVIDENCIA => limite de informacion,')
    print('     no un defecto del modelo que una arquitectura distinta pueda corregir.')
else:
    print(f'  => AMBIGUO: {100*frac_all:.0f}% de las veces, ratio mediano {ratio_all:.2f}.')
