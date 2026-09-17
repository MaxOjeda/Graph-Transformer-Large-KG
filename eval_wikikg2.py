"""Metricas de TEST del mejor checkpoint de wikikg2, con el protocolo oficial de OGB.

Uso: python eval_wikikg2.py <ckpt> [n_items] [test_top_nodes] [batch]

⚠️ Con test_top_nodes alto hay que BAJAR el batch: S crece a 6*L y el estado (B,S,d) domina.
   test_node_ratio 0.01 => S = 1 933 098 => a batch 32 son 7.9 GB por tensor (OOM).

⚠️ Se evalua sobre un SUBCONJUNTO BALANCEADO (mitad prediccion de cola, mitad de cabeza).
El test completo son 1 197 086 items y a batch 32 cuesta ~8 h; con 64 000 items el error
estandar del MRR es ~0.002, de sobra para decidir.
"""
import sys, torch
from torch.utils.data import DataLoader, Subset
from src.data_ogb import WikiKG2, OGBEvalSet, ogb_ranks
from src.model import SparseStateGraphTransformer

ck = torch.load(sys.argv[1], map_location='cpu')
hp = ck['hyper_parameters']
n_items = int(sys.argv[2]) if len(sys.argv) > 2 else 64000
data = WikiKG2()
sd = {k.replace('model.', '', 1): v for k, v in ck['state_dict'].items() if k.startswith('model.')}
m = SparseStateGraphTransformer(
        data.num_relation, hp['num_layer'], hp['hidden_dim'], hp['num_heads'], 0.0,
        node_slots=hp['node_slots'], top_nodes=hp['top_nodes'], edge_budget=hp['edge_budget'],
        edge_cap=hp['edge_cap'], dependent=hp['dependent'],
        prune_attn=hp['prune_attn'],
        # test_node_ratio 0.01 = lo que usa A*Net al evaluar wikikg2 (5x el de train)
        test_top_nodes=int(sys.argv[3]) if len(sys.argv) > 3 else 0).cuda().eval()
m.load_state_dict(sd)
graph = data.train_graph; graph.to('cuda')

full = OGBEvalSet(data.test_triplets.size(0))
ds = Subset(full, range(min(n_items, len(full))))       # prefijo: ya viene balanceado
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 32
ld = DataLoader(ds, shuffle=False, collate_fn=data.ogb_collate_fn('test'),
                batch_size=bs, num_workers=0)   # 0: los workers heredan CUDA ya inicializado   # 0: los workers heredan el contexto CUDA ya inicializado y fallan
print(f'ckpt {sys.argv[1]}\n items {len(ds):,} de {len(full):,}  '
      f'(el prefijo esta INTERCALADO => 50 % cola / 50 % cabeza)', flush=True)

rk_tail, rk_head = [], []
with torch.no_grad():
    for bi, b in enumerate(ld):
        bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        bb['graph'] = graph
        r = ogb_ranks(m(bb).gather(1, bb['cand']))
        is_tail = bb['r_index'] % 2 == 0
        rk_tail.append(r[is_tail].cpu()); rk_head.append(r[~is_tail].cpu())
        if bi % 250 == 0:
            print(f'  {bi}/{len(ld)}', flush=True)
t, h = torch.cat(rk_tail), torch.cat(rk_head)
def rep(nm, r):
    print(f'{nm:<10} n={len(r):>6,}  MRR {(1/r).mean():.4f}  H@1 {(r<=1).float().mean():.4f}  '
          f'H@3 {(r<=3).float().mean():.4f}  H@10 {(r<=10).float().mean():.4f}  MR {r.mean():.1f}')
rep('COLA', t); rep('CABEZA', h); rep('AMBAS', torch.cat([t, h]))
print(f'\nse = {(1/torch.cat([t,h])).std().item()/len(torch.cat([t,h]))**0.5:.4f}')
print('A*Net publicado en wikikg2: valid 0.6767  test 0.6851   (NBFNet: OOM)')
