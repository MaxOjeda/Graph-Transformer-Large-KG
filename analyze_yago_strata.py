"""ANALISIS ESTRATIFICADO en YAGO: donde gana la atencion contra A*Net?

Replica en YAGO el hallazgo H6, medido una vez en FB15k-237 con n=1: "la ventaja del GT esta
LOCALIZADA en el estrato de fuentes de grado >=250; el 91 % de la ventaja agregada viene de
ahi". Si se replica en un dataset 8.5x mayor, deja de ser un dato suelto y pasa a ser el
argumento mecanico del paper.

HIPOTESIS (escrita antes de mirar): la ventaja se concentra donde la FUENTE tiene grado alto.
Razon estructural: A*Net pesa NODOS (`layer_input = sigmoid(score) * hidden`, model.py:319),
asi que todas las aristas salientes de un nodo reciben el MISMO peso; no puede decir "este hub
es relevante por la relacion r1 pero no por r2". Nuestra atencion pesa ARISTAS. Esa libertad
solo importa cuando hay muchas aristas que discriminar, o sea en hubs.

⚠️ Si la ventaja resulta PLANA en grado, la explicacion mecanica se cae y hay que buscar otra.
"""
import pickle, sys, torch, numpy as np

ours = torch.load(sys.argv[1] if len(sys.argv) > 1 else 'yago_nopad_ranks.pt', map_location='cpu')
rows = ours['rows']                      # (Q,4) = (h, r, t, rank)
ei = ours['edge_index']                  # (E,3) del grafo de TRAIN
N = 123182
with open(sys.argv[2] if len(sys.argv) > 2 else 'astarnet_yago_ranks_test.pkl', 'rb') as f:
    astar = pickle.load(f)

deg = np.bincount(ei[:, 0].numpy(), minlength=N) + np.bincount(ei[:, 2].numpy(), minlength=N)

# --- alinear las dos corridas por (h, r, t) ---
# ⚠️ NO se pueden comparar por posicion: los dos harness ordenan el test distinto y nuestra
# convencion de relaciones es intercalada (2r/2r+1) contra su offset (r/r+R).
htr = astar['htr']                       # (Q,3) = (h, t, r) -- OJO al orden de columnas
rk_a = astar['ranking']                  # (Q,2): col 0 = predecir cola, col 1 = cabeza
key_a = {}
for i in range(len(htr)):
    h, t, r = int(htr[i, 0]), int(htr[i, 1]), int(htr[i, 2])
    key_a[(h, 2 * r, t)] = float(rk_a[i, 0])      # cola:   (h, 2r) -> t
    key_a[(t, 2 * r + 1, h)] = float(rk_a[i, 1])  # cabeza: (t, 2r+1) -> h

ok, ra, ro, src = [], [], [], []
for i in range(len(rows)):
    h, r, t, rk = (int(rows[i, 0]), int(rows[i, 1]), int(rows[i, 2]), float(rows[i, 3]))
    if (h, r, t) in key_a:
        ra.append(key_a[(h, r, t)]); ro.append(rk); src.append(h)
ra, ro, src = np.array(ra), np.array(ro), np.array(src)
print(f'queries alineadas: {len(ra):,} de {len(rows):,} nuestras y {2*len(htr):,} suyas')
if len(ra) < 1000:
    print('⚠️ MUY POCAS: revisar el orden de columnas de `htr` y la convencion de relaciones')
    sys.exit(1)
print(f'MRR global   nuestro {(1/ro).mean():.4f}   A*Net {(1/ra).mean():.4f}   '
      f'ventaja {(1/ro).mean()-(1/ra).mean():+.4f}')

# --- estratificado por grado de la FUENTE ---
edges = [0, 28, 100, 250, 1000, 10**9]
print(f"\n{'grado fuente':<16}{'n':>8}{'nuestro':>10}{'A*Net':>10}{'ventaja':>10}{'% del total':>12}")
d = deg[src]
tot = (1/ro).sum() - (1/ra).sum()
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (d >= lo) & (d < hi)
    if m.sum() == 0: continue
    adv = (1/ro[m]).mean() - (1/ra[m]).mean()
    share = 100 * ((1/ro[m]).sum() - (1/ra[m]).sum()) / tot if tot else float('nan')
    lab = f'{lo}-{hi-1}' if hi < 10**9 else f'{lo}+'
    print(f'{lab:<16}{int(m.sum()):>8}{(1/ro[m]).mean():>10.4f}{(1/ra[m]).mean():>10.4f}'
          f'{adv:>+10.4f}{share:>11.1f}%')
print(f"\nMR   nuestro {ro.mean():.1f}   A*Net {ra.mean():.1f}")

# --- test pareado por estrato ---
# Cada query se evalua con AMBOS modelos, asi que el contraste correcto es PAREADO (elimina
# la varianza de dificultad entre queries, que es enorme: MRR va de 0.29 a 0.73 entre estratos).
from math import sqrt
print(f"\n{'grado fuente':<16}{'n':>7}{'delta MRR':>11}{'se':>9}{'t':>8}{'gana/pierde/empata':>22}")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (d >= lo) & (d < hi)
    if m.sum() < 30: continue
    dif = (1/ro[m]) - (1/ra[m])
    se = dif.std(ddof=1) / sqrt(len(dif))
    w = int((ro[m] < ra[m]).sum()); l = int((ro[m] > ra[m]).sum()); e = int((ro[m] == ra[m]).sum())
    lab = f'{lo}-{hi-1}' if hi < 10**9 else f'{lo}+'
    print(f'{lab:<16}{len(dif):>7}{dif.mean():>+11.4f}{se:>9.4f}{dif.mean()/se:>8.2f}'
          f'{f"{w}/{l}/{e}":>22}')
dif = (1/ro) - (1/ra); se = dif.std(ddof=1)/sqrt(len(dif))
print(f'{"GLOBAL":<16}{len(dif):>7}{dif.mean():>+11.4f}{se:>9.4f}{dif.mean()/se:>8.2f}')
print('\n⚠️ n=1 por brazo: esto mide la diferencia ENTRE ESTAS DOS CORRIDAS, no entre metodos.')
