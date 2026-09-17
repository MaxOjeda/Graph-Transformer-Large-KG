"""Verifica el pipeline OGB antes de entrenar: formato, protocolo y equivalencia del rank.

Tres cosas:
 1. el rank de OGB coincide con su evaluador OFICIAL sobre scores aleatorios;
 2. el dataset arma los items como `OGBLKGTest` (2 por triple, positivo en la columna 0);
 3. el muestreo de negativos estrictos no devuelve colas verdaderas.
"""
import torch
from torch.utils.data import DataLoader
from src.data_ogb import WikiKG2, OGBEvalSet, ogb_ranks

torch.manual_seed(0)
# (1) rank contra el evaluador oficial de OGB
B, K = 64, 500
sc = torch.randn(B, 1 + K)
mine = ogb_ranks(sc)
opt = (sc[:, 1:] > sc[:, :1]).sum(1); pes = (sc[:, 1:] >= sc[:, :1]).sum(1)
ref = 0.5 * (opt + pes).float() + 1          # formula literal de ogb _eval_mrr
print(f'(1) rank vs formula oficial de OGB: max|dif| {(mine-ref).abs().max():.3e}')
print(f'    MRR de scores aleatorios: {(1/mine).mean():.4f}  (esperado ~1/500 x ln => bajo)')

d = WikiKG2()
print(f'\n(2) N={d.num_entity:,}  R={d.num_relation}  E={d.edge_index.size(0):,}')
ld = DataLoader(OGBEvalSet(d.test_triplets.size(0)), shuffle=False,
                collate_fn=d.ogb_collate_fn('test'), batch_size=4)
b = next(iter(ld))
print(f'    cand {tuple(b["cand"].shape)}   gold == cand[:,0]: '
      f'{bool((b["t_index"] == b["cand"][:,0]).all())}')
print(f'    r par (prediccion de COLA en la primera mitad del indice): '
      f'{(b["r_index"] % 2 == 0).all().item()}')
n = d.test_triplets.size(0)
it = iter(DataLoader(OGBEvalSet(n), shuffle=False, collate_fn=d.ogb_collate_fn('test'),
                     batch_size=4, sampler=torch.utils.data.SubsetRandomSampler([n, n+1, n+2, n+3])))
b2 = next(it)
print(f'    segunda mitad => r IMPAR (prediccion de CABEZA): {(b2["r_index"] % 2 == 1).all().item()}')
print(f'    items totales de test: {len(OGBEvalSet(n)):,}  (2 x {n:,})')

# (3) negativos estrictos
h, r = d.train_triplets[:32, 0], d.train_triplets[:32, 1]
neg = d.sample_strict_negatives(h, r, 32)
print(f'\n(3) negativos que SI son colas verdaderas: {int(d._is_true(h, r, neg).sum())} de {neg.numel()}')
gold = d.train_triplets[:32, 2].unsqueeze(1)
print(f'    control (el gold DEBE dar verdadero): {int(d._is_true(h, r, gold).sum())} de 32')
