"""WN18RR: respuestas aisladas, ranks sin score y DISTANCIA h-t en los fallos comunes (reusa analyze_overlap_wn18rr.py).
Uso: python analyze_wn18rr_distance.py   (env attention, ~2 min). Ver OBJETIVOS_Y_PLAN_WWW.md §3.4.6."""
import runpy, numpy as np, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path('analyze_overlap_wn18rr.py')
rk, F, r_, deg, da, dh, na, N, o_r = (g[k] for k in ('rk','F','r_','deg','da','dh','na','N','o_r'))
ei = g['ei']
und = np.bincount(np.concatenate([ei[:,0], ei[:,2]]), minlength=N) // 2   # grado no dirigido (train)
dat = und[r_[:,2]]
iso = dat == 0
print(f'respuestas SIN ninguna arista en el grafo de train: {iso.sum()} queries = {100*iso.mean():.1f}% del test')
for n in ('GT s42','A* s1024','NBF s1024'):
    print(f'  {n:9s} MRR en esas queries {np.mean(1/rk[n][iso]):.4f}  rank medio {rk[n][iso].mean():.0f}')
print(f'techo de MRR de cualquier metodo de caminos si acertara TODO lo demas: {1-iso.mean():.3f}')
both = F['GT s42'] & F['A* s1024'] & F['NBF s1024']
print(f'\nfallan los tres: {both.sum()} ({100*both.mean():.1f}%)  de los cuales respuesta aislada: {100*iso[both].mean():.1f}%')
for n in ('GT s42','A* s1024','NBF s1024'):
    r = rk[n][both]
    print(f'  {n:9s} p25/p50/p75/p90 {np.percentile(r,[25,50,75,90]).round(0)}  rank>=N-10 (sin score / ultimo): {100*np.mean(r>=N-10):.1f}%  rank>1000: {100*np.mean(r>1000):.1f}%')
# distancia h-t en el grafo de train para los fallos comunes (BFS no dirigido, hasta 6)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
A = csr_matrix((np.ones(len(ei)), (ei[:,0], ei[:,2])), shape=(N,N))
srcs = np.unique(r_[both,0])
D = shortest_path(A, directed=False, unweighted=True, indices=srcs)
pos = {s:i for i,s in enumerate(srcs)}
dist = np.array([D[pos[h], t] for h,t in zip(r_[both,0], r_[both,2])])
print('\ndistancia h-t (no dirigida, train) en los fallos comunes:')
for d in (1,2,3,4,5,6): print(f'  d={d}: {100*np.mean(dist==d):.1f}%')
print(f'  >6 o inalcanzable: {100*np.mean(~np.isfinite(dist) | (dist>6)):.1f}%')
# lo mismo en los aciertos comunes
ok = ~(F['GT s42']|F['A* s1024']|F['NBF s1024'])
srcs2 = np.unique(r_[ok,0]); D2 = shortest_path(A, directed=False, unweighted=True, indices=srcs2); pos2={s:i for i,s in enumerate(srcs2)}
dist2 = np.array([D2[pos2[h], t] for h,t in zip(r_[ok,0], r_[ok,2])])
print('en los aciertos comunes: d<=2: %.1f%%  d=3: %.1f%%  d>=4 o inalc.: %.1f%%' % (100*np.mean(dist2<=2), 100*np.mean(dist2==3), 100*np.mean(~np.isfinite(dist2)|(dist2>=4))))
print('en los fallos comunes:   d<=2: %.1f%%  d=3: %.1f%%  d>=4 o inalc.: %.1f%%' % (100*np.mean(dist<=2), 100*np.mean(dist==3), 100*np.mean(~np.isfinite(dist)|(dist>=4))))
