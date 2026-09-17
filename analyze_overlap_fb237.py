"""Solapamiento de fallos NBFNet / A*Net / GT en FB15k-237, alineando por NOMBRE de entidad.
Uso: python analyze_overlap_fb237.py   (env attention). Ver OBJETIVOS_Y_PLAN_WWW.md §3.3."""
import pickle, torch, numpy as np, csv, os, glob
from collections import Counter
RAW = os.path.expanduser('~/datasets/knowledge_graphs/FB15k-237/FB15k-237/raw')
files = [glob.glob(f'{RAW}/*{s}*')[0] for s in ('train','valid','test')]
inv_e, inv_r = {}, {}
for f in files:
    for h,r,t in csv.reader(open(f), delimiter='\t'):
        inv_e.setdefault(h, len(inv_e)); inv_r.setdefault(r, len(inv_r)); inv_e.setdefault(t, len(inv_e))
print('torchdrug vocab', len(inv_e), len(inv_r))
o_e = {int(i):e for e,i in (l.rstrip('\n').split('\t') for l in open('data/fb15k-237/entities.txt'))}
o_r = {int(i):r for r,i in (l.rstrip('\n').split('\t') for l in open('data/fb15k-237/relations.txt'))}
a = pickle.load(open('astarnet_ranks_fb237_s1024_test.pkl','rb'))
n = pickle.load(open('nbfnet_ranks_fb237_s1024_test.pkl','rb'))
g = torch.load('fb_b8lr1_ranks.pt', map_location='cpu')
rows = g['rows'].numpy(); ei = g['edge_index'].numpy()
htr = np.asarray(n['htr']); rn = np.asarray(n['ranking'],float); ra = np.asarray(a['ranking'],float)
assert (np.asarray(n['relation'])==np.asarray(a['relation'])).all()
key = {}
for i in range(len(htr)):
    h,t,r = map(int, htr[i])
    key[(h,r,t,0)] = i; key[(t,r,h,1)] = i
rk = {'NBFNet':[], 'A*Net':[], 'GT':[]}; keep=[]
for i in range(len(rows)):
    h,r,t,rkg = map(int, rows[i])
    rname = o_r[r]; rev = rname.startswith('-'); base = rname[1:] if rev else rname
    th, tt, tr = inv_e[o_e[h]], inv_e[o_e[t]], inv_r[base]
    k = (th,tr,tt,0) if not rev else (th,tr,tt,1)   # (fuente, r, respuesta, dir)
    if k not in key: continue
    j = key[k]; d = k[3]
    rk['NBFNet'].append(rn[j,d]); rk['A*Net'].append(ra[j,d]); rk['GT'].append(float(rkg)); keep.append(i)
rk = {k:np.array(v) for k,v in rk.items()}; keep=np.array(keep)
print(f'alineadas {len(keep)} de {len(rows)}')
for k,v in rk.items(): print(f'{k:7s} MRR {np.mean(1/v):.4f}  H@1 {np.mean(v<=1):.4f}  H@10 {np.mean(v<=10):.4f}  MR {v.mean():.1f}')
F = {k: v>10 for k,v in rk.items()}
allf = F['NBFNet']&F['A*Net']&F['GT']; anyf = F['NBFNet']|F['A*Net']|F['GT']
J = lambda x,y: (x&y).sum()/(x|y).sum()
print(f'\nfallan los TRES: {allf.mean()*100:.1f}%  falla alguno: {anyf.mean()*100:.1f}%')
print(f'Jaccard fallos  NBF-A*: {J(F["NBFNet"],F["A*Net"]):.3f}  NBF-GT: {J(F["NBFNet"],F["GT"]):.3f}  A*-GT: {J(F["A*Net"],F["GT"]):.3f}')
print(f'falla GT y no NBFNet: {(F["GT"]&~F["NBFNet"]).mean()*100:.1f}%   falla NBFNet y no GT: {(F["NBFNet"]&~F["GT"]).mean()*100:.1f}%')
def orc(keys):
    m = np.min(np.stack([rk[k] for k in keys]),0); return np.mean(1/m), np.mean(m<=10)
print('\noraculos (mejor rank por query):')
for keys in (['NBFNet','A*Net'],['NBFNet','GT'],['A*Net','GT'],['NBFNet','A*Net','GT']):
    m,h = orc(keys); print(f'  {"+".join(keys):20s} MRR {m:.4f}  H@10 {h:.4f}')
lr = {k: np.log(v) for k,v in rk.items()}
print('\npearson log-rank:', {f'{x}-{y}': round(float(np.corrcoef(lr[x],lr[y])[0,1]),3) for x,y in (('NBFNet','A*Net'),('NBFNet','GT'),('A*Net','GT'))})
# ensemble simple: media de log-rank no es un modelo; pero "rank reciproco medio" tampoco. Solo oraculo.
N=14541
deg = np.bincount(ei[:,0],minlength=N)+np.bincount(ei[:,2],minlength=N)
r_ = rows[keep]; da = deg[r_[:,2]]; ds = deg[r_[:,0]]
cnt = Counter(zip(ei[:,0].tolist(), ei[:,1].tolist()))
nans = np.array([cnt.get((int(h),int(r)),0) for h,r in zip(r_[:,0],r_[:,1])])
print('\ncaracterizacion:')
for lab,m in (('fallan los 3',allf),('aciertan los 3',~anyf),('falla NBF, acierta GT',F['NBFNet']&~F['GT']),('falla GT, acierta NBF',F['GT']&~F['NBFNet'])):
    print(f'  {lab:24s} n={m.sum():6d}  deg resp med {np.median(da[m]):4.0f}  deg fuente med {np.median(ds[m]):4.0f}  #resp(h,r) train med {np.median(nans[m]):3.0f}  resp deg<=14: {np.mean(da[m]<=14)*100:4.1f}%  sin resp en train: {np.mean(nans[m]==0)*100:4.1f}%')
m=allf
print('\nfallos comunes: percentiles de rank p25/p50/p75/p90')
for k in rk: print(f'  {k:7s}', np.percentile(rk[k][m],[25,50,75,90]).round(0))
for K in (20,50,100): print(f'  en top-{K}: NBF {np.mean(rk["NBFNet"][m]<=K)*100:.1f}%  A* {np.mean(rk["A*Net"][m]<=K)*100:.1f}%  GT {np.mean(rk["GT"][m]<=K)*100:.1f}%')
# MRR por estrato de grado de respuesta, los tres
print('\nMRR por grado de la RESPUESTA:')
for lo,hi in ((0,14),(15,27),(28,43),(44,99),(100,249),(250,10**9)):
    m=(da>=lo)&(da<=hi); print(f'  {lo}-{hi if hi<10**9 else "+"}: n={m.sum():6d} ' + '  '.join(f'{k} {np.mean(1/rk[k][m]):.3f}' for k in rk))
