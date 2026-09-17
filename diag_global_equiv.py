"""Test de EQUIVALENCIA AL INIT de los flags de la semana 2 (OBJETIVOS_Y_PLAN_WWW.md §4).

Todos los flags nuevos (--global_tokens, --hop_pe, --rel_frontier, --fallback ppr) y el
--indicator ppr se inicializan en CERO, asi que con los MISMOS pesos compartidos el forward
tiene que ser BIT A BIT identico al del modelo base. Si no lo es, hay un bug y no se entrena.

Ademas: --indicator ppr con `distance` en cero deberia ser un no-op exacto al init. Los jobs
93091-93093 (init cero) arrancaron en valid 0.13 / 0.41 contra 0.40 / 0.53 del base, o sea
NO son un no-op => este test es el que localiza el bug.

Uso (env attention, CPU basta):
  python diag_global_equiv.py [--data_path ./data/fb15k-237] [--batch 4] [--gpu]
"""
import argparse
import torch
import pytorch_lightning as pl

from train import GTLightningModule, KGDataModule, is_inductive


def build(dm, args, **extra):
    pl.seed_everything(0)
    base = dict(model='sparse_state', node_slots=args.node_slots,
                top_nodes=args.top_nodes, edge_budget=args.edge_budget, edge_cap=0,
                dependent=True, prune_attn='sigmoid', loss='bce', remove_one_hop=True)
    base.update(extra)
    m = GTLightningModule(dm.num_relation, 6, 32, 8, 0.0, 1e-3, 1e-4, **base)
    return m.eval()


def run(m, batch, train_mode=False):
    torch.manual_seed(0)
    m.train(train_mode)
    with torch.no_grad():
        return m.model(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='./data/fb15k-237')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--node_slots', type=int, default=14541)
    ap.add_argument('--top_nodes', type=int, default=1454)
    ap.add_argument('--edge_budget', type=int, default=54419)
    ap.add_argument('--gpu', action='store_true')
    args = ap.parse_args()
    dev = 'cuda' if args.gpu else 'cpu'

    dm = KGDataModule(args.data_path, is_inductive(args.data_path), 0, args.batch, args.batch)
    batch = next(iter(dm.train_dataloader()))
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if hasattr(batch['graph'], 'to'):
        batch['graph'] = batch['graph'].to(dev)

    ref = build(dm, args).to(dev)
    sd = ref.state_dict()
    out_ref = run(ref, batch)
    print(f'base: scores {tuple(out_ref.shape)}  finitos={bool(torch.isfinite(out_ref).all())}')

    variants = {
        'global_tokens=32 (topk)': dict(global_tokens=32, global_from=2, global_pool='topk'),
        'global_tokens=32 (active)': dict(global_tokens=32, global_from=2, global_pool='active'),
        'hop_pe': dict(hop_pe=True),
        'rel_frontier=1.0': dict(rel_frontier=1.0),
        'fallback=ppr': dict(fallback='ppr'),
        'indicator=ppr': dict(indicator='ppr'),
        'TODO junto': dict(global_tokens=32, hop_pe=True, rel_frontier=1.0, fallback='ppr'),
    }
    ok_all = True
    for name, kw in variants.items():
        m = build(dm, args, **kw).to(dev)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        out = run(m, batch)
        d = (out - out_ref).abs()
        exact = bool(torch.equal(out, out_ref))
        print(f'{name:28s} max|dif| {d.max().item():.3e}  exacto={exact}  '
              f'params nuevos={len(missing)}  finitos={bool(torch.isfinite(out).all())}')
        ok_all &= exact
        # y que los parametros nuevos RECIBAN gradiente. Con la salida en CERO, el resto del
        # bloque global tiene gradiente cero en el paso 0 (esperado); lo que NO puede pasar es
        # que siga en cero DESPUES de un paso de optimizador (el bug de `lowrank`, 2026-08-08 e).
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        report = {}
        for step in range(2):
            opt.zero_grad()
            torch.manual_seed(step)
            sc = m.model(batch)
            loss = m._bce_loss(sc, batch['t_index'], batch['filter_mask'])
            loss.backward()
            dead = [n for n in missing if m.get_parameter(n).grad is None or
                    not torch.isfinite(m.get_parameter(n).grad).all()]
            zero = [n for n in missing if m.get_parameter(n).grad is not None and
                    m.get_parameter(n).grad.abs().sum() == 0]
            report[step] = (len(dead), len(zero), float(loss))
            opt.step()
        print(f'{"":28s} paso 0: sin grad {report[0][0]}, grad cero {report[0][1]} (esperado en el bloque global) | '
              f'paso 1: sin grad {report[1][0]}, grad cero {report[1][1]}  loss {report[0][2]:.4f} -> {report[1][2]:.4f}')
        if report[1][1]:
            print(f'{"":28s} ⚠️ SIGUEN EN CERO tras un paso: {[n for n in missing if m.get_parameter(n).grad.abs().sum()==0]}')
    print('\nRESULTADO:', 'TODOS EXACTOS' if ok_all else '⚠️ ALGUNA VARIANTE NO ES UN NO-OP AL INIT')


if __name__ == '__main__':
    main()
