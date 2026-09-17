"""Solapamiento de fallos NBFNet / A*Net / GT en WN18RR, alineando por NOMBRE de entidad.
Replica analyze_overlap_fb237.py en WN18RR con 3 semillas por lado (control de ruido entre
semillas del MISMO modelo). Requiere los volcados de sbatch_dump_wn_ranks.sh.
Uso: python analyze_overlap_wn18rr.py   (env attention). Ver OBJETIVOS_Y_PLAN_WWW.md §3.3."""
import pickle, torch, numpy as np, csv, os, itertools, sys
from collections import Counter

RAW = os.path.expanduser('~/datasets/knowledge_graphs/torchdrug')
inv_e, inv_r = {}, {}
for s in ('train', 'valid', 'test'):
    for h, r, t in csv.reader(open(f'{RAW}/wn18rr_{s}.txt'), delimiter='\t'):
        inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
o_e = {int(i): e for e, i in (l.rstrip('\n').split('\t') for l in open('data/wn18rr/entities.txt'))}
o_r = {int(i): r for r, i in (l.rstrip('\n').split('\t') for l in open('data/wn18rr/relations.txt'))}
print('vocab torchdrug', len(inv_e), len(inv_r), ' nuestro', len(o_e), len(o_r))

G = {f'GT s{s}': torch.load(f, map_location='cpu') for s, f in
     ((42, 'wn_b8lr1_b100_ranks.pt'), (43, 'wn_b8lr1_b100_s43_ranks.pt'), (44, 'wn_b8lr1_b100_s44_ranks.pt'))}
rows = G['GT s42']['rows'].numpy(); ei = G['GT s42']['edge_index'].numpy()
for n in G: assert (G[n]['rows'][:, :3].numpy() == rows[:, :3]).all(), n

ext = {}
for tag, pat in (('A*', 'astarnet_wn_ranks_s%d_test.pkl'), ('NBF', 'nbfnet_wn_ranks_s%d_test.pkl')):
    for s in (1024, 1025, 1026):
        f = pat % s
        if os.path.exists(f):
            ext[f'{tag} s{s}'] = pickle.load(open(f, 'rb'))
if not ext:
    sys.exit('faltan los volcados de A*Net / NBFNet (sbatch_dump_wn_ranks.sh)')
ref = next(iter(ext.values()))
htr = np.asarray(ref['htr'])
for n, d in ext.items():
    assert (np.asarray(d['htr']) == htr).all(), f'{n} no comparte el orden de test'
key = {}
for i in range(len(htr)):
    h, t, r = map(int, htr[i]); key[(h, r, t, 0)] = (i, 0); key[(t, r, h, 1)] = (i, 1)
idx, ok = [], []
for i in range(len(rows)):
    h, r, t = map(int, rows[i, :3]); rn = o_r[r]; rev = rn.startswith('-'); base = rn[1:] if rev else rn
    k = (inv_e.get(o_e[h], -1), inv_r.get(base, -1), inv_e.get(o_e[t], -1), int(rev))
    if k in key: idx.append(key[k]); ok.append(i)
idx, ok = np.array(idx), np.array(ok)
print(f'alineadas {len(ok)} de {len(rows)}')

rk = {n: G[n]['rows'][:, 3].numpy().astype(float)[ok] for n in G}
for n, d in ext.items(): rk[n] = np.asarray(d['ranking'], float)[idx[:, 0], idx[:, 1]]
names = list(rk)
print('MRR :', {k: round(float(np.mean(1 / v)), 4) for k, v in rk.items()})
print('H@10:', {k: round(float(np.mean(v <= 10)), 3) for k, v in rk.items()})
F = {k: v > 10 for k, v in rk.items()}
J = lambda x, y: (x & y).sum() / (x | y).sum()
print('\nJaccard de fallos (rank>10):')
print(' ' * 10 + ''.join(f'{n:>10}' for n in names))
for a in names: print(f'{a:>10}' + ''.join(f'{J(F[a], F[b]):>10.3f}' for b in names))
grp = lambda ps: np.mean([J(F[a], F[b]) for a, b in ps]) if ps else float('nan')
fam = {p: [n for n in names if n.startswith(p)] for p in ('GT', 'A*', 'NBF')}
for p in fam: print(f'Jaccard medio {p}-{p} (semillas): {grp(list(itertools.combinations(fam[p], 2))):.3f}')
for p, q in (('GT', 'A*'), ('GT', 'NBF'), ('A*', 'NBF')):
    print(f'Jaccard medio {p}-{q}: {grp(list(itertools.product(fam[p], fam[q]))):.3f}')
one = {p: fam[p][0] for p in fam if fam[p]}
allf = np.all(np.stack([F[one[p]] for p in one]), 0); anyf = np.any(np.stack([F[one[p]] for p in one]), 0)
print(f'\n(una semilla por modelo) fallan todos: {allf.mean()*100:.1f}%  falla alguno: {anyf.mean()*100:.1f}%')
m = np.min(np.stack([rk[one[p]] for p in one]), 0)
print(f'oraculo entre modelos: MRR {np.mean(1/m):.4f} H@10 {np.mean(m<=10):.3f}')
mg = np.min(np.stack([rk[n] for n in fam['GT']]), 0)
print(f'oraculo 3 semillas GT : MRR {np.mean(1/mg):.4f} H@10 {np.mean(mg<=10):.3f}')

# --- cardinalidad y grado ---
N = len(o_e)
deg = np.bincount(ei[:, 0], minlength=N) + np.bincount(ei[:, 2], minlength=N)
cnt = Counter(zip(ei[:, 0].tolist(), ei[:, 1].tolist()))
r_ = rows[ok]; na = np.array([cnt.get((int(h), int(r)), 0) for h, r in zip(r_[:, 0], r_[:, 1])])
dh, da = deg[r_[:, 0]], deg[r_[:, 2]]
rG = rk['GT s42']; rX = rk[one['A*']] if 'A*' in one else rk[one['NBF']]
both = F['GT s42'] & (rX > 10)
print('\n=== WN18RR: fallo y MRR por cardinalidad de (h,r) en train ===')
print(f"{'#resp train':>12}{'n':>7}{'% test':>8}{'fallan ambos':>13}{'MRR ext':>8}{'MRR GT':>8}{'deg h med':>10}{'deg t med':>10}")
for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 10), (11, 30), (31, 10**9)):
    mm = (na >= lo) & (na <= hi)
    if mm.sum() == 0: continue
    lab = f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'
    print(f"{lab:>12}{mm.sum():>7}{100*mm.mean():>7.1f}%{100*both[mm].mean():>12.1f}%{np.mean(1/rX[mm]):>8.3f}{np.mean(1/rG[mm]):>8.3f}{np.median(dh[mm]):>10.0f}{np.median(da[mm]):>10.0f}")
print('\n=== WN18RR: MRR por grado de la RESPUESTA ===')
for lo, hi in ((0, 1), (2, 3), (4, 7), (8, 15), (16, 50), (51, 10**9)):
    mm = (da >= lo) & (da <= hi)
    if mm.sum() == 0: continue
    lab = f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'
    print(f"  {lab:>6}: n={mm.sum():5d} ({100*mm.mean():4.1f}%)  ext {np.mean(1/rX[mm]):.3f}  GT {np.mean(1/rG[mm]):.3f}  fallan ambos {100*both[mm].mean():.1f}%")
print('\nfallos comunes vs aciertos comunes (GT s42 y ext):')
for lab, mm in (('fallan ambos', both), ('aciertan ambos', ~(F['GT s42'] | (rX > 10)))):
    print(f'  {lab:15s} n={mm.sum():5d} deg resp med {np.median(da[mm]):4.0f}  deg fuente med {np.median(dh[mm]):4.0f}  '
          f'#resp(h,r) med {np.median(na[mm]):2.0f}  resp deg<=3: {np.mean(da[mm]<=3)*100:4.1f}%  card>=4: {np.mean(na[mm]>=4)*100:4.1f}%')
print('rank en fallos comunes p25/p50/p75/p90: ext', np.percentile(rX[both], [25, 50, 75, 90]).round(0), ' GT', np.percentile(rG[both], [25, 50, 75, 90]).round(0))
print('\nfallos comunes por relacion (fraccion del total de fallos comunes):')
rel_names = Counter(o_r[int(r)] for r in r_[both, 1])
for rname, c in rel_names.most_common(8):
    mm = both & np.array([o_r[int(r)] == rname for r in r_[:, 1]])
    tot = np.array([o_r[int(r)] == rname for r in r_[:, 1]])
    print(f'  {rname:40s} {100*c/both.sum():5.1f}% de los fallos | falla {100*mm.sum()/tot.sum():5.1f}% de sus {tot.sum()} queries')
