"""Fallo por CARDINALIDAD de (h,r) y ejemplos reales de fallos comunes en FB15k-237.
Uso: python analyze_cardinality_fb237.py   (env attention). Ver OBJETIVOS_Y_PLAN_WWW.md §3.4."""
import pickle, torch, numpy as np, csv, os, glob
from collections import Counter, defaultdict
RAW = os.path.expanduser('~/datasets/knowledge_graphs/FB15k-237/FB15k-237/raw')
files = [glob.glob(f'{RAW}/*{s}*')[0] for s in ('train','valid','test')]
inv_e, inv_r = {}, {}
for f in files:
    for h,r,t in csv.reader(open(f), delimiter='\t'):
        inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
e_name = {v:k for k,v in inv_e.items()}
o_e = {int(i):e for e,i in (l.rstrip('\n').split('\t') for l in open('data/fb15k-237/entities.txt'))}
o_r = {int(i):r for r,i in (l.rstrip('\n').split('\t') for l in open('data/fb15k-237/relations.txt'))}
a = pickle.load(open('astarnet_ranks_fb237_s1024_test.pkl','rb'))
n = pickle.load(open('nbfnet_ranks_fb237_s1024_test.pkl','rb'))
g = torch.load('fb_b8lr1_ranks.pt', map_location='cpu')
rows = g['rows'].numpy(); ei = g['edge_index'].numpy()
htr = np.asarray(n['htr']); rn = np.asarray(n['ranking'],float); ra = np.asarray(a['ranking'],float)
dis = np.asarray(n['top10_distractors'])
key = {}
for i in range(len(htr)):
    h,t,r = map(int, htr[i]); key[(h,r,t,0)] = i; key[(t,r,h,1)] = i
N=14541
deg = np.bincount(ei[:,0],minlength=N)+np.bincount(ei[:,2],minlength=N)
cnt = Counter(zip(ei[:,0].tolist(), ei[:,1].tolist()))
# respuestas en test por (h,r) nuestras
tcnt = Counter(zip(rows[:,0].tolist(), rows[:,1].tolist()))
rec=[]
for i in range(len(rows)):
    h,r,t,rkg = map(int, rows[i]); rname=o_r[r]; rev=rname.startswith('-'); base=rname[1:] if rev else rname
    k=(inv_e[o_e[h]], inv_r[base], inv_e[o_e[t]], int(rev)); j=key[k]; d=k[3]
    rec.append((h,r,t,rkg,rn[j,d],ra[j,d],cnt.get((h,r),0),tcnt[(h,r)],deg[h],deg[t],base,rev,j,d))
R = np.array([x[:10] for x in rec], float)
rkG,rkN,rkA,na,nt,dh,dt = R[:,3],R[:,4],R[:,5],R[:,6],R[:,7],R[:,8],R[:,9]
print('=== fallo (rank>10) y MRR por CARDINALIDAD de (h,r) en train (respuestas ya conocidas) ===')
print(f"{'#resp train':>12}{'n':>7}{'% test':>8}{'fallan 3':>10}{'MRR NBF':>9}{'MRR A*':>8}{'MRR GT':>8}{'H@10 NBF':>10}{'deg h med':>10}")
for lo,hi in ((0,0),(1,3),(4,10),(11,30),(31,100),(101,10**9)):
    m=(na>=lo)&(na<=hi); f3=((rkN>10)&(rkA>10)&(rkG>10))[m].mean()
    lab=f'{lo}-{hi}' if hi<10**9 else f'{lo}+'
    print(f"{lab:>12}{m.sum():>7}{100*m.mean():>7.1f}%{100*f3:>9.1f}%{np.mean(1/rkN[m]):>9.3f}{np.mean(1/rkA[m]):>8.3f}{np.mean(1/rkG[m]):>8.3f}{np.mean(rkN[m]<=10):>10.3f}{np.median(dh[m]):>10.0f}")
print('\n=== cruce: cardinalidad x grado de la fuente, MRR de NBFNet (n) ===')
cb=[(0,3),(4,30),(31,10**9)]; sb=[(0,27),(28,99),(100,249),(250,10**9)]
print(f"{'card x deg h':>14}"+''.join(f"{f'{lo}-{hi}' if hi<10**9 else f'{lo}+':>16}" for lo,hi in sb))
for clo,chi in cb:
    line=f"{(f'{clo}-{chi}' if chi<10**9 else f'{clo}+'):>14}"
    for slo,shi in sb:
        m=(na>=clo)&(na<=chi)&(dh>=slo)&(dh<=shi)
        line+=f"{np.mean(1/rkN[m]):>9.3f} ({m.sum():>5})"
    print(line)
# fraccion de fallos comunes que son card>=31
f3=(rkN>10)&(rkA>10)&(rkG>10)
print(f'\nfallos comunes con cardinalidad >=31: {np.mean(na[f3]>=31)*100:.1f}%  >=11: {np.mean(na[f3]>=11)*100:.1f}%   (en aciertos comunes >=31: {np.mean(na[~((rkN>10)|(rkA>10)|(rkG>10))]>=31)*100:.1f}%)')
# ejemplos: fallos comunes, cardinalidad alta, ordenar por cardinalidad
idx=np.where(f3&(na>=50))[0]
idx=idx[np.argsort(-na[idx])]
print('\n=== ejemplos de fallos comunes con alta cardinalidad ===')
seen=set(); shown=0
for i in idx:
    h,r,t,rkg,rN,rA,nA,nT,dH,dT,base,rev,j,d = rec[i]
    if (h,r) in seen: continue
    seen.add((h,r)); shown+=1
    dl=[e_name[x] for x in dis[j,d]]
    print(f'\n- fuente {o_e[h]} (grado {dH})  relacion {"INV " if rev else ""}{base}\n  respuesta {o_e[t]} (grado {dT})  | respuestas conocidas en train: {nA}, en test: {nT}\n  rank NBFNet {rN:.0f}  A*Net {rA:.0f}  GT {rkg}\n  top-10 de NBFNet: {dl}')
    if shown>=6: break
# para el ejemplo del USA: cuantas entidades del grafo son "film" (aparecen como cabeza de /film/film/...)?


