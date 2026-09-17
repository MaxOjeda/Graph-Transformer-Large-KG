"""Jaccard de fallos ENTRE SEMILLAS del mismo modelo vs entre modelos, FB15k-237 (3 GT, 3 A*Net, 1 NBFNet).
Uso: python analyze_seeds_fb237.py   (env attention). Ver OBJETIVOS_Y_PLAN_WWW.md §3.3."""
import pickle, torch, numpy as np, csv, os, glob, itertools
RAW = os.path.expanduser('~/datasets/knowledge_graphs/FB15k-237/FB15k-237/raw')
files = [glob.glob(f'{RAW}/*{s}*')[0] for s in ('train','valid','test')]
inv_e, inv_r = {}, {}
for f in files:
    for h,r,t in csv.reader(open(f), delimiter='\t'):
        inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
o_e = {int(i):e for e,i in (l.rstrip('\n').split('\t') for l in open('data/fb15k-237/entities.txt'))}
o_r = {int(i):r for r,i in (l.rstrip('\n').split('\t') for l in open('data/fb15k-237/relations.txt'))}
A = {s: pickle.load(open(f'astarnet_ranks_bias_s{s}_test.pkl','rb')) for s in (1024,1025,1026)}
NB = pickle.load(open('nbfnet_ranks_fb237_s1024_test.pkl','rb'))
htr = np.asarray(NB['htr'])
for s in A: assert (np.asarray(A[s]['htr'])==htr).all()
key = {}
for i in range(len(htr)):
    h,t,r = map(int, htr[i]); key[(h,r,t,0)] = (i,0); key[(t,r,h,1)] = (i,1)
G = {s: torch.load(f, map_location='cpu')['rows'].numpy() for s,f in ((42,'fb_b8lr1_ranks.pt'),(43,'fb_b8lr1_s43_ranks.pt'),(44,'fb_b8lr1_s44_ranks.pt'))}
rows = G[42]
idx = []
for i in range(len(rows)):
    h,r,t = map(int, rows[i,:3]); rname=o_r[r]; rev=rname.startswith('-'); base=rname[1:] if rev else rname
    idx.append(key[(inv_e[o_e[h]], inv_r[base], inv_e[o_e[t]], int(rev))])
idx = np.array(idx)
# comprobar que las 3 semillas del GT comparten el orden de filas
for s in (43,44): assert (G[s][:,:3]==rows[:,:3]).all()
rk = {f'GT s{s}': G[s][:,3].astype(float) for s in G}
for s in A: rk[f'A* s{s}'] = np.asarray(A[s]['ranking'],float)[idx[:,0], idx[:,1]]
rk['NBF s1024'] = np.asarray(NB['ranking'],float)[idx[:,0], idx[:,1]]
F = {k: v>10 for k,v in rk.items()}
J = lambda x,y: (x&y).sum()/(x|y).sum()
names = list(rk)
print('MRR:', {k: round(float(np.mean(1/v)),4) for k,v in rk.items()})
print('\nJaccard de fallos (rank>10), matriz:')
print(' '*11 + ''.join(f'{n:>11}' for n in names))
for a in names: print(f'{a:>11}' + ''.join(f'{J(F[a],F[b]):>11.3f}' for b in names))
def grp(pairs): return np.mean([J(F[a],F[b]) for a,b in pairs])
gt=[n for n in names if n.startswith('GT')]; ast=[n for n in names if n.startswith('A*')]
print(f'\nJaccard medio  GT-GT (semillas): {grp(list(itertools.combinations(gt,2))):.3f}')
print(f'Jaccard medio  A*-A* (semillas): {grp(list(itertools.combinations(ast,2))):.3f}')
print(f'Jaccard medio  GT-A*            : {grp(list(itertools.product(gt,ast))):.3f}')
print(f'Jaccard medio  GT-NBF           : {grp([(g,"NBF s1024") for g in gt]):.3f}')
print(f'Jaccard medio  A*-NBF           : {grp([(a,"NBF s1024") for a in ast]):.3f}')
allf = np.all(np.stack([F[n] for n in names]),0)
print(f'\nfallan los 7 (3 GT + 3 A* + NBF): {allf.mean()*100:.1f}%   falla alguno: {np.any(np.stack([F[n] for n in names]),0).mean()*100:.1f}%')
m = np.min(np.stack([rk[n] for n in names]),0); print(f'oraculo de los 7: MRR {np.mean(1/m):.4f} H@10 {np.mean(m<=10):.4f}')
mg = np.min(np.stack([rk[n] for n in gt]),0); ma = np.min(np.stack([rk[n] for n in ast]),0)
print(f'oraculo 3 semillas GT: {np.mean(1/mg):.4f}   oraculo 3 semillas A*: {np.mean(1/ma):.4f}')
