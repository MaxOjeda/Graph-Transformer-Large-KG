"""Contraste PAREADO brazo vs baseline, estratificado (OBJETIVOS_Y_PLAN_WWW.md §3.4, §4).

Compara dos grupos de volcados de nuestro harness (`--dump_ranks`, formato rows=(h,r,t,rank)),
que comparten el orden de queries de test. Reporta:
  - MRR / H@10 / MR por semilla y media +- sd de cada grupo, y Welch entre grupos;
  - delta PAREADO por query (media sobre semillas de 1/rank), global y por estrato:
      cardinalidad de (h,r) en train, grado de la fuente, grado de la respuesta,
      y (opcional, --distance) distancia h-t en train, que es el eje de WN18RR;
  - fraccion de respuestas SIN score (rank >= N-10) por grupo.
La prediccion falsable del bloque global (§3.6): mueve cardinalidad 4-30, NO mueve 101+.

Uso (env attention):
  python analyze_arm_vs_base.py --data_path ./data/fb15k-237 \
      --base fb_b8lr1_ranks.pt fb_b8lr1_s43_ranks.pt fb_b8lr1_s44_ranks.pt \
      --arm  fb_global_s42_ranks.pt fb_global_s43_ranks.pt fb_global_s44_ranks.pt [--distance]
"""
import argparse
from collections import Counter
from math import sqrt

import numpy as np
import torch


def load_group(paths):
    rows, ranks = None, []
    ei = None
    for p in paths:
        d = torch.load(p, map_location='cpu')
        r = d['rows'].numpy()
        if rows is None:
            rows, ei = r[:, :3], d['edge_index'].numpy()
        assert (r[:, :3] == rows).all(), f'{p}: orden de queries distinto'
        ranks.append(r[:, 3].astype(float))
    return rows, ei, np.stack(ranks)          # (n_seeds, Q)


def summary(R, N):
    mrr = (1 / R).mean(1); h10 = (R <= 10).mean(1); mr = R.mean(1); uns = (R >= N - 10).mean(1)
    return mrr, h10, mr, uns


def welch(a, b):
    d = b.mean() - a.mean()
    se = sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 and len(b) > 1 else float('nan')
    return d, se, (d / se if se == se and se > 0 else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--base', nargs='+', required=True)
    ap.add_argument('--arm', nargs='+', required=True)
    ap.add_argument('--distance', action='store_true', help='estratifica ademas por distancia h-t (BFS, ~2 min)')
    args = ap.parse_args()

    rows, ei, RB = load_group(args.base)
    rows2, _, RA = load_group(args.arm)
    assert (rows == rows2).all(), 'base y brazo no comparten el orden de queries'
    N = max(int(ei[:, [0, 2]].max()), int(rows[:, [0, 2]].max())) + 1
    Q = rows.shape[0]

    print(f'queries {Q}  N {N}  semillas base {RB.shape[0]}  brazo {RA.shape[0]}')
    for name, R in (('BASE', RB), ('BRAZO', RA)):
        mrr, h10, mr, uns = summary(R, N)
        print(f'{name:6s} MRR {mrr.mean():.4f} +- {mrr.std(ddof=1) if len(mrr) > 1 else 0:.4f} '
              f'[{", ".join(f"{m:.4f}" for m in mrr)}]  H@10 {h10.mean():.4f}  MR {mr.mean():.1f}  '
              f'sin score {100 * uns.mean():.1f}%')
    d, se, t = welch(summary(RB, N)[0], summary(RA, N)[0])
    print(f'Welch brazo - base: {d:+.4f} +- {se:.4f}  t = {t:.2f}')

    # pareado por query: media sobre semillas del reciproco del rank
    qb, qa = (1 / RB).mean(0), (1 / RA).mean(0)
    diff = qa - qb
    print(f'pareado global: {diff.mean():+.4f}  se {diff.std(ddof=1) / sqrt(Q):.4f}  '
          f't {diff.mean() / (diff.std(ddof=1) / sqrt(Q)):.2f}   gana/pierde/empata '
          f'{(diff > 1e-9).sum()}/{(diff < -1e-9).sum()}/{(np.abs(diff) <= 1e-9).sum()}')

    deg = np.bincount(ei[:, 0], minlength=N) + np.bincount(ei[:, 2], minlength=N)
    cnt = Counter(zip(ei[:, 0].tolist(), ei[:, 1].tolist()))
    card = np.array([cnt.get((int(h), int(r)), 0) for h, r in zip(rows[:, 0], rows[:, 1])])
    dh, dt = deg[rows[:, 0]], deg[rows[:, 2]]

    def strata(label, values, cuts):
        print(f'\n{label}')
        print(f"{'estrato':>12}{'n':>7}{'% test':>8}{'MRR base':>10}{'MRR brazo':>10}{'delta':>9}{'se':>8}{'t':>7}{'gana/pierde':>14}")
        for lo, hi in cuts:
            m = (values >= lo) & (values <= hi)
            if m.sum() < 20:
                continue
            dd = diff[m]; se_ = dd.std(ddof=1) / sqrt(m.sum())
            lab = f'{lo}-{hi}' if hi < 10**9 else f'{lo}+'
            print(f'{lab:>12}{m.sum():>7}{100 * m.mean():>7.1f}%{qb[m].mean():>10.4f}{qa[m].mean():>10.4f}'
                  f'{dd.mean():>+9.4f}{se_:>8.4f}{dd.mean() / se_ if se_ > 0 else 0:>7.2f}'
                  f'{f"{(dd > 1e-9).sum()}/{(dd < -1e-9).sum()}":>14}')

    strata('CARDINALIDAD de (h,r) en train (eje de FB15k-237 / YAGO)', card,
           ((0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, 10**9)))
    strata('GRADO DE LA FUENTE', dh, ((0, 27), (28, 99), (100, 249), (250, 999), (1000, 10**9)))
    strata('GRADO DE LA RESPUESTA', dt, ((0, 3), (4, 14), (15, 43), (44, 99), (100, 249), (250, 10**9)))

    if args.distance:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        A = csr_matrix((np.ones(len(ei)), (ei[:, 0], ei[:, 2])), shape=(N, N))
        srcs = np.unique(rows[:, 0])
        D = shortest_path(A, directed=False, unweighted=True, indices=srcs)
        pos = {s: i for i, s in enumerate(srcs)}
        dist = np.array([D[pos[h], t] for h, t in zip(rows[:, 0], rows[:, 2])])
        dist = np.where(np.isfinite(dist), dist, 99).astype(int)
        strata('DISTANCIA h-t en train (eje de WN18RR; 99 = inalcanzable)', dist,
               ((1, 2), (3, 3), (4, 4), (5, 6), (7, 98), (99, 99)))


if __name__ == '__main__':
    main()
