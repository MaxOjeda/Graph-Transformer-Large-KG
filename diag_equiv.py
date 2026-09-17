"""Es mi capa podada una reimplementacion FIEL de la sparse, o tiene un bug?

Los dos sospechosos (fallback, gate) quedaron descartados. Antes de culpar a la poda hay que
descartar que `PrunedSparseAttentionLayer` compute algo distinto de
`SparseRelationalAttentionLayer`: es codigo NUEVO, no el mismo.

TEST: cargar los pesos YA ENTRENADOS del GT completo (test 0.41827) en el modelo podado,
correrlo con edge_ratio=1.0 (selecciona TODAS las aristas) y el gate desactivado. Si mi capa
es fiel, los scores deben coincidir. Si no, el bug es mio y no tiene que ver con podar.
"""
import sys, torch
sys.path.insert(0, '/home/jreutter/Attention')
from torch.utils.data import DataLoader
from src.data import TransductiveKnowledgeGraph
from src.model import SparseGraphTransformer, PrunedSparseGraphTransformer

ck = torch.load(sys.argv[1], map_location='cpu')
sd = {k.replace('model.', '', 1): v for k, v in ck['state_dict'].items() if k.startswith('model.')}
hp = ck['hyper_parameters']
data = TransductiveKnowledgeGraph('./data/fb15k-237')
kw = dict(dependent=hp.get('dependent', False))
full = SparseGraphTransformer(data.num_relation, hp['num_layer'], hp['hidden_dim'],
                              hp['num_heads'], 0.0, **kw).cuda().eval()
full.load_state_dict(sd)
pr = PrunedSparseGraphTransformer(data.num_relation, hp['num_layer'], hp['hidden_dim'],
                                  hp['num_heads'], 0.0, edge_ratio=1.0, **kw).cuda().eval()
missing, unexpected = pr.load_state_dict(sd, strict=False)
print('faltan en el podado :', [k for k in missing])
print('sobran del completo:', [k for k in unexpected][:5])

# desactivar el gate para aislar SOLO la diferencia de implementacion
for l in pr.layers:
    f = l.forward
    l.forward = (lambda f: lambda x, edges, q_emb=None, gate=None: f(x, edges, q_emb=q_emb,
                                                                     gate=None))(f)

graph = data.train_graph; graph.to('cuda')
loader = DataLoader(data.test_triplets.clone(), shuffle=False,
                    collate_fn=data.test_collate_fn, batch_size=4, num_workers=0)
b = next(iter(loader))
bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
bb['graph'] = graph
with torch.no_grad():
    s_full = full(bb)
    s_pr = pr(bb)
d = (s_full - s_pr).abs()
print(f'\nmax|diff| = {d.max().item():.3e}   media|diff| = {d.mean().item():.3e}')
print(f'corr = {torch.corrcoef(torch.stack([s_full.flatten(), s_pr.flatten()]))[0,1].item():.6f}')
print('=> capa FIEL (la brecha es la poda)' if d.max().item() < 1e-3
      else '=> ⚠️ MI CAPA COMPUTA ALGO DISTINTO: el bug es de implementacion, no de la poda')
