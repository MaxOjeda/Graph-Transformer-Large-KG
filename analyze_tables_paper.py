"""Reproduce las tablas de fallo/solapamiento del paper con el MODELO NUEVO (GT + bloque global).

Genera, para FB15k-237 y YAGO3-10, exactamente las tablas que estaban calculadas con el GT viejo:
  (1) metricas agregadas + % de fallo por modelo
  (2) solapamiento de fallos: fallan todos / falla alguno / Jaccard por pares / discrepancias /
      correlacion de Pearson del log-rank
  (3) donde esta la respuesta en los fallos comunes: percentiles y % en top-20/50/100
  (4) fallo y MRR por cardinalidad de (h,r) en train
  (5) oraculos

Uso (env attention, CPU, ~2 min):  python analyze_tables_paper.py --ds fb237
"""
import argparse, csv, itertools, os, pickle
from collections import Counter

import numpy as np
import torch

TD = os.path.expanduser('~/datasets/knowledge_graphs/torchdrug')
FB = os.path.expanduser('~/datasets/knowledge_graphs/FB15k-237/FB15k-237/raw')

CFG = {
    'fb237': dict(raw=[f'{FB}/{s}.txt' for s in ('train', 'valid', 'test')], data='data/fb15k-237',
                  ours=('GT+global', 'fb_global_s42_e30_ranks.pt'),
                  old=('GT viejo', 'fb_b8lr1_ranks.pt'),
                  ext=[('NBFNet', 'nbfnet_ranks_fb237_s1024_test.pkl'),
                       ('A*Net', 'astarnet_ranks_bias_s1024_test.pkl')],
                  card=((0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, 10**9))),
    'yago': dict(raw=[f'{TD}/yago310_{s}.txt' for s in ('train', 'valid', 'test')], data='data/yago3-10',
                 ours=('GT+global', 'yago_global_s42_e20_ranks.pt'),
                 old=('GT viejo', 'gt_b200_s42_ranks.pt'),
                 ext=[('A*Net', 'astarnet_yago_ranks_test.pkl')],
                 card=((0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, 10**9))),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ds', required=True, choices=list(CFG))
    args = ap.parse_args(); C = CFG[args.ds]

    inv_e, inv_r = {}, {}
    for f in C['raw']:
        for h, r, t in csv.reader(open(f), delimiter='\t'):
            inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
    o_e = {int(i): e for e, i in (l.rstrip('\n').split('\t') for l in open(f"{C['data']}/entities.txt"))}
    o_r = {int(i): r for r, i in (l.rstrip('\n').split('\t') for l in open(f"{C['data']}/relations.txt"))}

    g = torch.load(C['ours'][1], map_location='cpu')
    rows, ei = g['rows'].numpy(), g['edge_index'].numpy()
    ext = [(n, pickle.load(open(p, 'rb'))) for n, p in C['ext'] if os.path.exists(p)]
    htr = np.asarray(ext[0][1]['htr']); key = {}
    for i in range(len(htr)):
        h, t, r = map(int, htr[i]); key[(h, r, t, 0)] = (i, 0); key[(t, r, h, 1)] = (i, 1)
    idx, ok = [], []
    for i in range(len(rows)):
        h, r, t = map(int, rows[i, :3]); rn = o_r[r]; rev = rn.startswith('-'); base = rn[1:] if rev else rn
        k = (inv_e.get(o_e[h], -1), inv_r.get(base, -1), inv_e.get(o_e[t], -1), int(rev))
        if k in key: idx.append(key[k]); ok.append(i)
    idx, ok = np.array(idx), np.array(ok)
    rows = rows[ok]
    rk = {n: np.asarray(d['ranking'], float)[idx[:, 0], idx[:, 1]] for n, d in ext}
    rk[C['ours'][0]] = g['rows'].numpy()[ok, 3].astype(float)
    if os.path.exists(C['old'][1]):
        rk[C['old'][0]] = torch.load(C['old'][1], map_location='cpu')['rows'].numpy()[ok, 3].astype(float)
    names = [n for n, _ in ext] + [C['ours'][0]] + ([C['old'][0]] if C['old'][0] in rk else [])
    F = {n: rk[n] > 10 for n in names}
    OURS = C['ours'][0]
    print(f'=== {args.ds}: {len(ok):,} queries alineadas ===')

    print(f"\n(1) METRICAS\n{'modelo':>12}{'MRR':>9}{'Hits@1':>9}{'Hits@3':>9}{'Hits@10':>9}{'MR':>9}{'falla':>9}")
    for n in names:
        v = rk[n]
        print(f'{n:>12}{np.mean(1/v):>9.4f}{np.mean(v<=1):>9.4f}{np.mean(v<=3):>9.4f}'
              f'{np.mean(v<=10):>9.4f}{v.mean():>9.1f}{100*np.mean(v>10):>8.1f}%')

    core = [n for n, _ in ext] + [OURS]
    allf = np.all(np.stack([F[n] for n in core]), 0); anyf = np.any(np.stack([F[n] for n in core]), 0)
    J = lambda a, b: (F[a] & F[b]).sum() / (F[a] | F[b]).sum()
    print(f'\n(2) SOLAPAMIENTO DE FALLOS (rank > 10)')
    print(f'  fallan los {len(core)} a la vez            {100*allf.mean():>6.1f}%')
    print(f'  falla al menos uno              {100*anyf.mean():>6.1f}%')
    for a, b in itertools.combinations(core, 2):
        print(f'  Jaccard({a}, {b}){"":<{max(0,18-len(a)-len(b))}} {J(a,b):>6.3f}')
    for e, _ in ext:
        print(f'  falla {OURS} y acierta {e}{"":<{max(0,8-len(e))}} {100*np.mean(F[OURS]&~F[e]):>6.1f}%')
        print(f'  falla {e} y acierta {OURS}{"":<{max(0,8-len(e))}} {100*np.mean(F[e]&~F[OURS]):>6.1f}%')
    lr = {n: np.log(rk[n]) for n in core}
    for a, b in itertools.combinations(core, 2):
        print(f'  Pearson log-rank({a}, {b}){"":<{max(0,10-len(a)-len(b))}} {np.corrcoef(lr[a],lr[b])[0,1]:>6.3f}')

    print(f"\n(3) DONDE ESTA LA RESPUESTA EN LOS FALLOS COMUNES (n={allf.sum():,})")
    print(f"{'modelo':>12}{'p25':>7}{'p50':>7}{'p75':>8}{'p90':>9}{'top-20':>9}{'top-50':>9}{'top-100':>9}")
    for n in core:
        v = rk[n][allf]; p = np.percentile(v, [25, 50, 75, 90])
        print(f'{n:>12}{p[0]:>7.0f}{p[1]:>7.0f}{p[2]:>8.0f}{p[3]:>9.0f}'
              f'{100*np.mean(v<=20):>8.1f}%{100*np.mean(v<=50):>8.1f}%{100*np.mean(v<=100):>8.1f}%')

    N = max(int(ei[:, [0, 2]].max()), int(rows[:, [0, 2]].max())) + 1
    deg = np.bincount(ei[:, 0], minlength=N) + np.bincount(ei[:, 2], minlength=N)
    cnt = Counter(zip(ei[:, 0].tolist(), ei[:, 1].tolist()))
    card = np.array([cnt.get((int(h), int(r)), 0) for h, r in zip(rows[:, 0], rows[:, 1])])
    print(f"\n(4) FALLO Y MRR POR CARDINALIDAD DE (h,r) EN TRAIN")
    hdr = f"{'#resp train':>12}{'n':>7}{'%test':>7}{'fallan todos':>14}"
    for n in core: hdr += f'{"MRR "+n:>15}'
    print(hdr + f"{'H@10 '+core[0]:>15}{'grado h med':>13}")
    for lo, hi in C['card']:
        m = (card >= lo) & (card <= hi)
        if m.sum() < 20: continue
        lab = f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'
        line = f'{lab:>12}{m.sum():>7}{100*m.mean():>6.1f}%{100*allf[m].mean():>13.1f}%'
        for n in core: line += f'{np.mean(1/rk[n][m]):>15.3f}'
        print(line + f'{np.mean(rk[core[0]][m]<=10):>15.3f}{np.median(deg[rows[m,0]]):>13.0f}')

    print(f"\n(5) CARACTERIZACION Y ORACULOS")
    da = deg[rows[:, 2]]
    for lab, m in (('fallan todos', allf), ('aciertan todos', ~anyf)):
        print(f'  {lab:15s} n={m.sum():6d}  grado resp med {np.median(da[m]):5.0f}  '
              f'grado fuente med {np.median(deg[rows[m,0]]):5.0f}  #resp(h,r) med {np.median(card[m]):4.0f}  '
              f'card>=31 {100*np.mean(card[m]>=31):4.1f}%')
    for ks in list(itertools.combinations(core, 2)) + [tuple(core)]:
        mm = np.min(np.stack([rk[k] for k in ks]), 0)
        print(f'  oraculo {" + ".join(ks):28s} MRR {np.mean(1/mm):.4f}  H@10 {np.mean(mm<=10):.4f}')


if __name__ == '__main__':
    main()
