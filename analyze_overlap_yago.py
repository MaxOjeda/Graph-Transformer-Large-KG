"""Solapamiento de fallos A*Net / GT en YAGO3-10 (alineacion por nombre; 6 semillas del GT como control de ruido) + fallo por cardinalidad y ejemplos.
Uso: python analyze_overlap_yago.py   (env attention). Ver OBJETIVOS_Y_PLAN_WWW.md §3.3."""
import pickle, torch, numpy as np, csv, os, itertools
from collections import Counter
RAW = os.path.expanduser('~/datasets/knowledge_graphs/torchdrug')
inv_e, inv_r = {}, {}
for s in ('train','valid','test'):
    for h,r,t in csv.reader(open(f'{RAW}/yago310_{s}.txt'), delimiter='\t'):
        inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
o_e = {int(i):e for e,i in (l.rstrip('\n').split('\t') for l in open('data/yago3-10/entities.txt'))}
o_r = {int(i):r for r,i in (l.rstrip('\n').split('\t') for l in open('data/yago3-10/relations.txt'))}
print('vocab torchdrug', len(inv_e), len(inv_r), ' nuestro', len(o_e), len(o_r))
A = pickle.load(open('astarnet_yago_ranks_test.pkl','rb'))
htr = np.asarray(A['htr']); rA = np.asarray(A['ranking'],float)
key = {}
for i in range(len(htr)):
    h,t,r = map(int, htr[i]); key[(h,r,t,0)]=(i,0); key[(t,r,h,1)]=(i,1)
G = {n: torch.load(f, map_location='cpu') for n,f in (('GT200 s42','gt_b200_s42_ranks.pt'),('GT200 s43','gt_b200_s43_ranks.pt'),('GT200 s44','gt_b200_s44_ranks.pt'),
                                                       ('GT100 s42','yago_beta100_ranks.pt'),('GT100 s43','gt_b100_s43_ranks.pt'),('GT100 s44','gt_b100_s44_ranks.pt'))}
rows = G['GT200 s42']['rows'].numpy(); ei = G['GT200 s42']['edge_index'].numpy()
for n in G: assert (G[n]['rows'][:,:3].numpy()==rows[:,:3]).all(), n
idx=[]; ok=[]
for i in range(len(rows)):
    h,r,t = map(int, rows[i,:3]); rn=o_r[r]; rev=rn.startswith('-'); base=rn[1:] if rev else rn
    k=(inv_e.get(o_e[h],-1), inv_r.get(base,-1), inv_e.get(o_e[t],-1), int(rev))
    if k in key: idx.append(key[k]); ok.append(i)
idx=np.array(idx); ok=np.array(ok)
print(f'alineadas {len(ok)} de {len(rows)}')
rk = {n: G[n]['rows'][:,3].numpy().astype(float)[ok] for n in G}
rk['A* s1024'] = rA[idx[:,0], idx[:,1]]
print('MRR:', {k: round(float(np.mean(1/v)),4) for k,v in rk.items()}, '\nH@10:', {k: round(float(np.mean(v<=10)),3) for k,v in rk.items()})
F = {k: v>10 for k,v in rk.items()}; J = lambda x,y: (x&y).sum()/(x|y).sum()
names=list(rk)
print('\nJaccard de fallos:'); print(' '*11+''.join(f'{n:>11}' for n in names))
for a in names: print(f'{a:>11}'+''.join(f'{J(F[a],F[b]):>11.3f}' for b in names))
g2=[n for n in names if n.startswith('GT200')]; g1=[n for n in names if n.startswith('GT100')]
mean=lambda ps: np.mean([J(F[a],F[b]) for a,b in ps])
print(f'\nJaccard medio GT200-GT200 {mean(list(itertools.combinations(g2,2))):.3f} | GT100-GT100 {mean(list(itertools.combinations(g1,2))):.3f} | GT200-GT100 {mean(list(itertools.product(g2,g1))):.3f} | GT-A* {mean([(g,"A* s1024") for g in g2+g1]):.3f}')
allf = np.all(np.stack([F[n] for n in names]),0); anyf=np.any(np.stack([F[n] for n in names]),0)
f3 = F['GT200 s42']&F['A* s1024']
print(f'fallan GT(s42) y A*: {f3.mean()*100:.1f}%  fallan los 7: {allf.mean()*100:.1f}%  falla alguno: {anyf.mean()*100:.1f}%')
m=np.min(np.stack([rk['GT200 s42'],rk['A* s1024']]),0); print(f'oraculo GT+A*: MRR {np.mean(1/m):.4f} H@10 {np.mean(m<=10):.3f}   oraculo 3 semillas GT200: {np.mean(1/np.min(np.stack([rk[n] for n in g2]),0)):.4f}')
# cardinalidad
N=123182; deg=np.bincount(ei[:,0],minlength=N)+np.bincount(ei[:,2],minlength=N)
cnt=Counter(zip(ei[:,0].tolist(), ei[:,1].tolist()))
r_=rows[ok]; na=np.array([cnt.get((int(h),int(r)),0) for h,r in zip(r_[:,0],r_[:,1])]); dh=deg[r_[:,0]]; da=deg[r_[:,2]]
rG, rAa = rk['GT200 s42'], rk['A* s1024']
print('\n=== YAGO: fallo y MRR por cardinalidad de (h,r) en train ===')
print(f"{'#resp train':>12}{'n':>7}{'% test':>8}{'fallan ambos':>13}{'MRR A*':>8}{'MRR GT':>8}{'H@10 A*':>9}{'deg h med':>10}{'deg t med':>10}")
for lo,hi in ((0,0),(1,3),(4,10),(11,30),(31,100),(101,10**9)):
    m=(na>=lo)&(na<=hi)
    if m.sum()==0: continue
    print(f"{(f'{lo}-{hi}' if hi<10**9 else f'{lo}+'):>12}{m.sum():>7}{100*m.mean():>7.1f}%{100*f3[m].mean():>12.1f}%{np.mean(1/rAa[m]):>8.3f}{np.mean(1/rG[m]):>8.3f}{np.mean(rAa[m]<=10):>9.3f}{np.median(dh[m]):>10.0f}{np.median(da[m]):>10.0f}")
print('\nfallos comunes vs aciertos comunes (GT s42 y A*):')
for lab,m in (('fallan ambos',f3),('aciertan ambos',~(F['GT200 s42']|F['A* s1024']))):
    print(f'  {lab:15s} n={m.sum():5d} deg resp med {np.median(da[m]):5.0f}  deg fuente med {np.median(dh[m]):5.0f}  #resp(h,r) med {np.median(na[m]):3.0f}  resp deg<=14: {np.mean(da[m]<=14)*100:4.1f}%  card>=31: {np.mean(na[m]>=31)*100:4.1f}%')
print('\nrank en fallos comunes p25/p50/p75/p90: A*', np.percentile(rAa[f3],[25,50,75,90]).round(0), ' GT', np.percentile(rG[f3],[25,50,75,90]).round(0))
# ejemplos
e_name={v:k for k,v in inv_e.items()}
ii=np.where(f3&(na>=30))[0]; ii=ii[np.argsort(-na[ii])]; seen=set(); shown=0
print('\n=== ejemplos YAGO (fallan ambos, alta cardinalidad) ===')
tcnt=Counter(zip(r_[:,0].tolist(), r_[:,1].tolist()))
for i in ii:
    h,r,t=map(int,r_[i,:3])
    if (h,r) in seen: continue
    seen.add((h,r)); shown+=1; rn=o_r[r]
    print(f"- fuente {o_e[h]} (grado {dh[i]})  rel {rn}  resp {o_e[t]} (grado {da[i]}) | resp en train {na[i]}, en test {tcnt[(h,r)]} | rank A* {rAa[i]:.0f} GT {rG[i]:.0f}")
    if shown>=6: break
