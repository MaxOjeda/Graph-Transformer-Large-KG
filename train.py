"""Entry point limpio del proyecto "Attention".

Entrena/evalua el Relational Full-Attention Graph Transformer (src/model.py) en
KGC con ranking FULL-FILTERED (todos los nodos candidatos), mismo protocolo que se
uso para reproducir KnowFormer/NBFNet. Sin nada de KnowFormer (model.py es nuevo).

Uso tipico (FB15k-237 inductivo v1):
  source env.sh && PY=$(which python)
  $PY train.py --data_path ./data/inductive/fb15k-237_v1 \
      --num_layer 6 --hidden_dim 32 --num_heads 8 \
      --batch_size 16 --test_batch_size 16 --max_epochs 20 \
      --learning_rate 5e-3 --weight_decay 1e-4 --seed 42
"""

import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import (GraphTransformer, NBFNet, SparseGraphTransformer,
                       SparseExpanderGraphTransformer, SparseNBFValueTransformer,
                       PrunedSparseGraphTransformer, SparseStateGraphTransformer)
from src.data import TransductiveKnowledgeGraph, InductiveKnowledgeGraph
from src.data_ogb import WikiKG2, OGBEvalSet, ogb_ranks
from src.metric import MRMetric, MRRMetric, HitsMetric


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class KGDataModule(pl.LightningDataModule):
    def __init__(self, data_path, inductive, num_workers, batch_size, test_batch_size,
                 eval_subsample=0, eval_seed=1024):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # Submuestreo ALEATORIO del set de evaluacion de OGB (su `fast_test`). Ver src/data_ogb.py.
        self.eval_subsample = eval_subsample
        self.eval_seed = eval_seed
        # `--data_path ogbl-wikikg2` usa el PROTOCOLO DE EVALUACION DE OGB (500 negativos
        # fijos + su desempate), no el full-filtered del resto del harness. Ver src/data_ogb.py.
        self.is_ogb = 'wikikg2' in data_path
        if self.is_ogb:
            self.data = WikiKG2()
        else:
            cls = InductiveKnowledgeGraph if inductive else TransductiveKnowledgeGraph
            self.data = cls(data_path)
        self.num_relation = self.data.num_relation

    def train_dataloader(self):
        # Solo relaciones forward (par); el collate genera las reversas (impar).
        triplets = self.data.train_triplets.clone()
        triplets = triplets[triplets[:, 1] % 2 == 0]
        return DataLoader(triplets, shuffle=True, collate_fn=self.data.train_collate_fn,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def _ogb_loader(self, split):
        n = getattr(self.data, split + '_triplets').size(0)
        return DataLoader(OGBEvalSet(n, self.eval_subsample, self.eval_seed), shuffle=False,
                          collate_fn=self.data.ogb_collate_fn(split),
                          batch_size=self.test_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.is_ogb:
            return self._ogb_loader('valid')
        return DataLoader(self.data.valid_triplets.clone(), shuffle=False,
                          collate_fn=self.data.valid_collate_fn,
                          batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.is_ogb:
            return self._ogb_loader('test')
        return DataLoader(self.data.test_triplets.clone(), shuffle=False,
                          collate_fn=self.data.test_collate_fn,
                          batch_size=self.test_batch_size, num_workers=self.num_workers)


# ----------------------------------------------------------------------------
# Lightning module
# ----------------------------------------------------------------------------
class GTLightningModule(pl.LightningModule):
    def __init__(self, num_relation, num_layer, hidden_dim, num_heads, drop,
                 learning_rate, weight_decay, model='rfat', aggregate='pna',
                 use_rwse=False, rwse_dim=16, use_lappe=False, lappe_dim=16,
                 use_source_rw=False, source_rw_dim=8,
                 use_rpb=False, rpb_hops=4, rpb_dim=16, exp_degree=4,
                 exp_typing='single', exp_path_len=3,
                 attn='softmax', filtered_ce=False, edge_drop=0.0,
                 remove_one_hop=False, rel_param='diag', rel_rank=4, dependent=False,
                 edge_ratio=0.1, prune_attn='softmax', node_ratio=0.1,
                 rel_readout=False, node_slots=4096, top_nodes=1024,
                 edge_budget=8192, edge_cap=64, evict='id', grad_ckpt=False,
                 test_top_nodes=0, test_edge_budget=0,
                 indicator='onehot', num_indicator_bin=10, edge_dropout_p=0.0,
                 break_tie=False, exp_degree_state=0,
                 global_tokens=0, global_from=2, global_pool='topk', hop_pe=False,
                 rel_frontier=0.0, fallback='scalar',
                 loss='ce', num_negative=32, adversarial_temperature=0.5, dump_ranks=''):
        super().__init__()
        self.save_hyperparameters()
        self.remove_one_hop = remove_one_hop
        # Volcado de ranks POR QUERY en test (--dump_ranks). Las metricas de src/metric.py
        # acumulan sumas, asi que sin esto no se puede estratificar el resultado por
        # ninguna covariable. Motivo: evaluacion estratificada por grado de la respuesta
        # (SESSION_NOTES 2026-08-22 b) -- el modo de fallo dominante es la cola larga, y
        # el MRR agregado la esconde.
        self.dump_ranks = dump_ranks
        self._rank_rows = []
        relp = dict(rel_param=rel_param, rel_rank=rel_rank, dependent=dependent)
        sparse_kw = dict(rel_readout=rel_readout)
        rwse = dict(use_rwse=use_rwse, rwse_dim=rwse_dim,
                    use_lappe=use_lappe, lappe_dim=lappe_dim,
                    use_source_rw=use_source_rw, source_rw_dim=source_rw_dim,
                    edge_drop=edge_drop)
        if model == 'nbfnet':
            self.model = NBFNet(num_relation, num_layer, hidden_dim, aggregate=aggregate,
                                edge_drop=edge_drop)
        elif model == 'sparse':
            self.model = SparseGraphTransformer(num_relation, num_layer, hidden_dim,
                                                num_heads, drop, attn=attn, **rwse, **relp,
                                                **sparse_kw)
        elif model == 'sparse_exp':
            self.model = SparseExpanderGraphTransformer(num_relation, num_layer, hidden_dim,
                                                        num_heads, drop,
                                                        exp_degree=exp_degree, attn=attn,
                                                        exp_typing=exp_typing,
                                                        exp_path_len=exp_path_len,
                                                        **rwse, **relp)
        elif model == 'sparse_pruned':
            self.model = PrunedSparseGraphTransformer(
                num_relation, num_layer, hidden_dim, num_heads, drop,
                edge_ratio=edge_ratio, dependent=dependent, edge_drop=edge_drop,
                prune_attn=prune_attn, node_ratio=node_ratio,
                rel_readout=rel_readout)
        elif model == 'sparse_state':
            self.model = SparseStateGraphTransformer(
                num_relation, num_layer, hidden_dim, num_heads, drop,
                node_slots=node_slots, top_nodes=top_nodes, edge_budget=edge_budget,
                edge_cap=edge_cap, dependent=dependent, prune_attn=prune_attn,
                rel_readout=rel_readout, evict=evict, grad_ckpt=grad_ckpt,
                test_top_nodes=test_top_nodes, test_edge_budget=test_edge_budget,
                indicator=indicator, num_indicator_bin=num_indicator_bin,
                edge_dropout=edge_dropout_p, break_tie=break_tie,
                exp_degree=exp_degree_state,
                global_tokens=global_tokens, global_from=global_from, global_pool=global_pool,
                hop_pe=hop_pe, rel_frontier=rel_frontier, fallback=fallback)
        elif model == 'sparse_nbfv':
            self.model = SparseNBFValueTransformer(num_relation, num_layer, hidden_dim,
                                                   num_heads, drop, aggregate=aggregate, **rwse)
        else:
            self.model = GraphTransformer(num_relation, num_layer, hidden_dim,
                                          num_heads, drop, **rwse, **relp,
                                          use_rpb=use_rpb, rpb_hops=rpb_hops, rpb_dim=rpb_dim)
        self.mr = MRMetric()
        self.mrr = MRRMetric()
        self.h1 = HitsMetric(topk=1)
        self.h3 = HitsMetric(topk=3)
        self.h10 = HitsMetric(topk=10)

    def remove_edge(self, batched_data):
        """Quita del grafo las aristas que trivializan la query. Devuelve graph_mask (E,).

        Dos modos, que son exactamente las dos ramas de `remove_easy_edges` de NBFNet
        (NBFNet-PyG/nbfnet/models.py:48-71):

        - `remove_one_hop=False` (default, comportamiento historico del harness): quita solo
          (h,r,t) y su reversa (t,r^-1,h), o sea la arista con la RELACION EXACTA de la query.
          Equivale a la rama `else` de ellos.
        - `remove_one_hop=True`: quita TODAS las aristas directas entre h y t (en ambas
          direcciones), sin mirar la relacion. Es la rama `if self.remove_one_hop` de ellos,
          y es la que activa su config inductivo (`remove_one_hop: yes`). Su comentario la
          describe como "dynamic edge dropout": impide resolver la query por el atajo de 1
          salto cuando el par ya esta conectado por OTRA relacion, y fuerza caminos largos.

        Por que importa (medido 2026-08-08 sobre nuestros datos): la fraccion de triples de
        train que conservan una arista directa h-t bajo el modo debil es 19.6 % en FB15k-237
        ind v1, 23.1 % en ind v2 y **31.6 % en FB15k-237 transductivo**; en WN18RR es ~1 %.

        En ambos modos el enmascarado es batch-wide y solo se aplica en train (este metodo
        solo se llama desde `training_step`), igual que en NBFNet (`if self.training:`).
        """
        h, r, t = batched_data['h_index'], batched_data['r_index'], batched_data['t_index']
        graph = batched_data['graph']
        ei = graph.edge_index.to(h.device)
        h_rm = torch.cat([h, t], 0)
        t_rm = torch.cat([t, h], 0)
        if self.remove_one_hop:
            # match solo por par (src,dst): cae cualquier relacion entre h y t.
            encode2 = lambda a, c: c + a * graph.num_nodes
            src_hash = encode2(ei[:, 0], ei[:, 2])
            tgt_hash = encode2(h_rm, t_rm)
        else:
            rev_r = torch.where(r % 2 == 0, r + 1, r - 1)
            r_rm = torch.cat([r, rev_r], 0)
            encode = lambda a, b, c: c + (a + b * graph.num_nodes) * graph.num_nodes
            src_hash = encode(ei[:, 0], ei[:, 1], ei[:, 2])
            tgt_hash = encode(h_rm, r_rm, t_rm)
        mask = ~torch.isin(src_hash, tgt_hash)
        batched_data['graph_mask'] = mask
        return batched_data

    def _bce_loss(self, score, t_index, filter_mask):
        """BCE con k negativos muestreados, FIEL a la receta de NBFNet/ULTRA/A*Net.

        Referencia: NBFNet-PyG/script/run.py:57-68 (loss) y nbfnet/tasks.py:41-91
        (muestreo). Reproducida operacion por operacion:
          - `strict negative sampling`: los negativos NUNCA son colas verdaderas de
            (h, r). NBFNet construye ese mask desde el grafo de TRAIN
            (`strict_negative_mask`); aca `filter_mask` ya es exactamente eso — viene
            de `data.train_filters`, que se arma solo con triplets de train — asi que
            el complemento es su mask (que tambien excluye el propio gold).
          - muestreo uniforme CON reemplazo sobre los candidatos (su `rand *
            num_candidate` lo es; `multinomial(replacement=True)` es equivalente).
          - self-adversarial weighting de RotatE: los negativos se pesan con
            softmax(score_neg / T) SIN gradiente, T = adversarial_temperature.
            Con T <= 0 el peso es 1/num_negative (su rama `else`).

        Nota de costo: el modelo ya calcula los N scores (la propagacion no depende
        de la loss), asi que muestrear negativos NO ahorra computo en este harness —
        solo indexa. Ver la entrada 2026-08-19 de SESSION_NOTES.
        """
        num_neg = self.hparams.num_negative
        cand = 1.0 - filter_mask.float()          # 1 en los candidatos legales
        empty = cand.sum(-1) == 0                 # guard: query sin candidatos
        if empty.any():
            cand = cand.clone()
            cand[empty] = 1.0
        neg_index = torch.multinomial(cand, num_neg, replacement=True)   # (B, k)
        return self._bce_loss_neg(score, t_index, neg_index)

    def _bce_loss_neg(self, score, t_index, neg_index):
        """BCE-k con los negativos YA muestreados. Es el cuerpo de `_bce_loss`; se separo
        para que wikikg2 pueda muestrear por rechazo sin construir la mascara (B, N)."""
        pos = score.gather(1, t_index.unsqueeze(1))                      # (B, 1)
        neg = score.gather(1, neg_index)                                 # (B, k)
        pred = torch.cat([pos, neg], dim=1)                              # (B, 1+k)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        neg_weight = torch.ones_like(pred)
        if self.hparams.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(
                    pred[:, 1:] / self.hparams.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / num_neg
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        return loss.mean()

    def training_step(self, batched_data, batch_idx):
        batched_data = self.remove_edge(batched_data)
        score = self.model(batched_data)                          # (B, N)
        t_index = batched_data['t_index']
        if self.hparams.loss == 'bce':
            if batched_data.get('ogb'):
                # ⚠️ En wikikg2 NO se materializa el `filter_mask` (B, N): con N=2.5 M son
                # 80 MB por batch y un `multinomial` sobre 2.5 M categorias. Los negativos
                # estrictos salen por RECHAZO con busqueda binaria (src/data_ogb.py).
                neg = self.trainer.datamodule.data.sample_strict_negatives(
                    batched_data['h_index'].cpu(), batched_data['r_index'].cpu(),
                    self.hparams.num_negative).to(score.device)
                loss = self._bce_loss_neg(score, t_index, neg)
            else:
                loss = self._bce_loss(score, t_index, batched_data['filter_mask'])
            self.log('train_loss', loss, prog_bar=True)
            self.log('mem_gb', torch.cuda.max_memory_allocated() / 1024**3, prog_bar=True)
            return loss
        # CE de grafo completo: el gold es t_index; las demas entidades son negativos.
        if self.hparams.filtered_ce:
            # CE filtrado SOLO-train: las otras respuestas conocidas EN TRAIN de la
            # misma query (h, r) salen del denominador (multi-answer label handling).
            # 'filter_mask' del train_collate_fn viene de data.train_filters (solo
            # triplets de train) => NO hay fuga de val/test, a diferencia del pipeline
            # de Exphormer_Max que filtraba con train+val+test.
            filt = batched_data['filter_mask'].bool().clone()
            filt[torch.arange(filt.size(0), device=filt.device), t_index] = False
            score = score.masked_fill(filt, float('-inf'))
        loss = F.cross_entropy(score, t_index)
        self.log('train_loss', loss, prog_bar=True)
        self.log('mem_gb', torch.cuda.max_memory_allocated() / 1024**3, prog_bar=True)
        return loss

    def _eval_step(self, batched_data, collect=False):
        score = self.model(batched_data)
        if 'cand' in batched_data:
            # PROTOCOLO OGB: ranking contra los 500 negativos FIJOS que entrega el dataset,
            # no full-filtered sobre las N entidades. Rank = 0.5*(optimista+pesimista)+1,
            # que es el desempate de su evaluador oficial y NO el nuestro. Ver src/data_ogb.py.
            if not torch.isfinite(score).all():
                raise FloatingPointError('scores no finitos: el modelo divergio.')
            sc = score.gather(1, batched_data['cand'])                   # (B, 1+k)
            ranks = ogb_ranks(sc)
            for m in (self.mr, self.mrr, self.h1, self.h3, self.h10):
                m.update(ranks)
            return
        answer = score.gather(1, batched_data['t_index'].unsqueeze(1))
        filt = batched_data['filter_mask'].bool()
        # ⚠️ GUARDA CONTRA NaN. `NaN >= NaN` es False, asi que un modelo que diverge da
        # suma 0 => rank 1 para TODA query => **valid_mrr = 1.000 exacto**, y Lightning lo
        # guarda feliz como "mejor checkpoint" porque selecciona por max(valid_mrr). Paso de
        # verdad: job 92462 (--evict keep) reporto 1.000 con train_loss=nan. Un NaN tiene que
        # ser ruidoso, no verse como un resultado perfecto.
        if not torch.isfinite(score).all():
            bad = (~torch.isfinite(score)).any(1).sum().item()
            raise FloatingPointError(
                f'scores no finitos en {bad}/{score.size(0)} queries: el modelo divergio. '
                f'Sin esta guarda el rank seria 1 para todas y valid_mrr saldria 1.000.')
        ranks = torch.sum((score >= answer) & (~filt), dim=1) + 1
        for m in (self.mr, self.mrr, self.h1, self.h3, self.h10):
            m.update(ranks)
        if collect and self.dump_ranks:
            self._rank_rows.append(torch.stack([
                batched_data['h_index'], batched_data['r_index'],
                batched_data['t_index'], ranks], dim=1).detach().cpu())

    def validation_step(self, batched_data, batch_idx):
        self._eval_step(batched_data)

    def test_step(self, batched_data, batch_idx):
        self._eval_step(batched_data, collect=True)

    def _save_ranks(self):
        if not (self.dump_ranks and self._rank_rows):
            return
        rows = torch.cat(self._rank_rows)          # (Q,4) = (h, r, t, rank)
        torch.save({'rows': rows,
                    'edge_index': self.trainer.datamodule.data.edge_index.cpu(),
                    'num_relation': self.trainer.datamodule.data.num_relation},
                   self.dump_ranks)
        print(f'[dump_ranks] {self.dump_ranks}  Q={len(rows)}', flush=True)
        self._rank_rows = []

    def _log_epoch(self, split):
        vals = {f'{split}_mr': self.mr.compute(), f'{split}_mrr': self.mrr.compute(),
                f'{split}_hits1': self.h1.compute(), f'{split}_hits3': self.h3.compute(),
                f'{split}_hits10': self.h10.compute()}
        for m in (self.mr, self.mrr, self.h1, self.h3, self.h10):
            m.reset()
        for k, v in vals.items():
            self.log(k, v, prog_bar=k.endswith(('mrr', 'hits10')), sync_dist=True)

    def validation_epoch_end(self, outputs):
        self._log_epoch('valid')

    def test_epoch_end(self, outputs):
        self._save_ranks()
        self._log_epoch('test')

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'norm']
        groups = [
            {'params': [p for n, p in self.named_parameters()
                        if any(d in n for d in no_decay) and p.requires_grad],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.named_parameters()
                        if not any(d in n for d in no_decay) and p.requires_grad],
             'weight_decay': self.hparams.weight_decay},
        ]
        opt = torch.optim.Adam(groups, lr=self.hparams.learning_rate)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [10, 15], 0.1)
        return [opt], [{'scheduler': sched, 'interval': 'epoch'}]


INDUCTIVE_DATASETS = {'fb15k-237', 'wn18rr', 'nell-995'}


def is_inductive(data_path):
    name = os.path.basename(os.path.normpath(data_path))
    return '_v' in name and name.split('_v')[0] in INDUCTIVE_DATASETS


def main():
    p = ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--model', type=str, default='rfat',
                   choices=['rfat', 'nbfnet', 'sparse', 'sparse_nbfv', 'sparse_exp', 'sparse_pruned', 'sparse_state'])
    p.add_argument('--exp_typing', type=str, default='single',
                   choices=['single', 'ultra', 'path'],
                   help='Tipado de las aristas del expander (--model sparse_exp). '
                        '"single": una relacion nueva generica R_exp compartida (Exphormer). '
                        '"ultra": relacion PRESTADA de las representaciones relacionales de '
                        'ULTRA (grafo de relaciones + GNN condicionado a r_q). '
                        '"path": composicion de las relaciones del camino mas corto real '
                        'entre los dos nodos (fallback a R_exp si no hay camino).')
    p.add_argument('--exp_path_len', type=int, default=3,
                   help='Largo maximo del camino para --exp_typing path.')
    p.add_argument('--exp_degree', type=int, default=4,
                   help='Grado del grafo expander (aristas por nodo) para --model sparse_exp.')
    p.add_argument('--attn', type=str, default='softmax',
                   choices=['softmax', 'sigmoid', 'degree', 'anchor', 'rel', 'qc'],
                   help='Agregacion/parametrizacion de la atencion sparse (sparse/sparse_exp). '
                        'softmax: segment-softmax (promedio, pierde conteo de caminos); '
                        'sigmoid: gates sin normalizar => suma ponderada que conserva conteo de '
                        'caminos y grado (opcion A); degree: softmax x log(1+grado_in); '
                        'anchor: interpola softmax con la media uniforme via lambda aprendido '
                        'por cabeza => regulariza hacia la agregacion fija (opcion C); '
                        'rel: ABLATION sin termino q.k => logit puramente relacional '
                        '(b[head,rel] + compatibilidad r_q x rel), transferible por construccion; '
                        'qc: paquete QC-Exphormer (Q anclado a x0, logit trilineal con relacion '
                        'vectorial, conditioning c_q en q/k/e, exp-suma sin normalizar + residual BF).')
    p.add_argument('--loss', type=str, default='ce', choices=['ce', 'bce'],
                   help='Objetivo de entrenamiento. bce: BCE con --num_negative negativos '
                        'muestreados (strict) + self-adversarial weighting => LA RECETA DE '
                        'NBFNet/ULTRA/A*Net, y por lo tanto la unica comparable con los numeros '
                        'publicados. ce: cross-entropy de grafo completo (1vsAll) sobre las N '
                        'entidades. El default sigue siendo ce SOLO por compatibilidad con los '
                        'sbatch_*.sh existentes; para el paper WWW hay que pasar --loss bce '
                        'EXPLICITAMENTE en todos los brazos, baselines incluidos.')
    p.add_argument('--num_negative', type=int, default=32,
                   help='Negativos por positivo con --loss bce (NBFNet usa 32).')
    p.add_argument('--adversarial_temperature', type=float, default=0.5,
                   help='Temperatura del self-adversarial weighting de RotatE con --loss bce '
                        '(NBFNet usa 0.5). <= 0 => peso uniforme 1/num_negative.')
    p.add_argument('--filtered_ce', action='store_true',
                   help='CE filtrado solo-train: excluye del denominador las otras respuestas '
                        'conocidas EN TRAIN de la query (h, r). Sin fuga de val/test. Aplica a '
                        'todos los modelos; para comparar hay que re-correr el baseline con el flag.')
    p.add_argument('--aggregate', type=str, default='pna', choices=['pna', 'sum'],
                   help='NBFNet y sparse_nbfv: funcion de agregacion del message passing.')
    p.add_argument('--use_rwse', action='store_true',
                   help='Sumar RWSE (random-walk structural encoding) a x^0. Solo modelos '
                        'de atencion (rfat/sparse/sparse_nbfv); nbfnet lo ignora.')
    p.add_argument('--rwse_dim', type=int, default=16,
                   help='Largo del random walk para RWSE (k=1..rwse_dim).')
    p.add_argument('--use_lappe', action='store_true',
                   help='Anade Laplacian positional encoding (autovectores del Laplaciano) a x^0.')
    p.add_argument('--use_source_rw', action='store_true',
                   help='Labeling condicionado a la query: landing probs de random walk '
                        'desde el head por nodo (rompe la simetria inicial del transformer).')
    p.add_argument('--source_rw_dim', type=int, default=8,
                   help='Largo del random walk para source_rw (k=1..source_rw_dim).')
    p.add_argument('--lappe_dim', type=int, default=16,
                   help='Numero de autovectores no triviales para LapPE.')
    p.add_argument('--use_rpb', action='store_true',
                   help='Relational Path Bias (opcion B): bias par de camino relacional '
                        'query-conditioned en el logit de atencion (solo modelo rfat).')
    p.add_argument('--rpb_hops', type=int, default=4,
                   help='Profundidad K del camino relacional para RPB (fuera del horizonte).')
    p.add_argument('--rpb_dim', type=int, default=16,
                   help='Dim de los embeddings de relacion para el score de compatibilidad RPB.')
    p.add_argument('--num_layer', type=int, default=6)
    p.add_argument('--hidden_dim', type=int, default=32)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--drop', type=float, default=0.1)
    p.add_argument('--rel_param', type=str, default='diag',
                   choices=['diag', 'lowrank'],
                   help="Parametrizacion relacional del VALOR en los graph transformers. "
                        "'diag' (default) = historico: escalar por cabeza + vector DistMult "
                        "=> R*(H+d) params por capa. 'lowrank' suma una correccion "
                        "(v@U[rel])@V[rel]^T de rango --rel_rank, init en CERO (identico a "
                        "'diag' en la inicializacion). NBFNet usa R*d^2 por capa: 14x mas "
                        "capacidad relacional. Solo con --exp_typing single.")
    p.add_argument('--edge_ratio', type=float, default=0.1,
                   help='Solo con --model sparse_pruned: fraccion de aristas que se '
                        'seleccionan POR QUERY en cada capa (top-L de tamano fijo). Es el '
                        'analogo de `node_ratio` de A*Net y lo que hace que la memoria sea '
                        'O(B*L*d) en vez de O(B*E*d) => la ruta a ogbl-wikikg2. '
                        'Ver DISENO_GT_PODA.md.')
    p.add_argument('--node_slots', type=int, default=4096,
                   help='Solo --model sparse_state: cupo S de SLOTS de estado por query. El '
                        'estado es (B,S,d) en vez de (B,N,d) -- a 2.5 M nodos eso son 30 MB '
                        'contra 15.6 GB. Ver DISENO_GT_PODA.md (adenda 2026-08-26).')
    p.add_argument('--top_nodes', type=int, default=1024,
                   help='Solo --model sparse_state: nodos K que se expanden por capa.')
    p.add_argument('--edge_budget', type=int, default=8192,
                   help='Solo --model sparse_state: aristas L que se propagan por capa.')
    p.add_argument('--edge_cap', type=int, default=64,
                   help='Solo --model sparse_state: tope de salientes POR NODO. '
                        '**0 = SIN TOPE (ruta sin padding, Alg. 2 de A*Net)**, que es lo que '
                        'hacen ellos. Con tope el pool es una matriz (B,K*cap) y se toman las '
                        'PRIMERAS cap aristas de la fila CSR -- orden del archivo del dataset, '
                        'no de relevancia: descarta evidencia. Medido en YAGO: cap 64 -> MRR '
                        '0.4498, cap 1024 -> 0.5478 (+0.098, sin saturar).')
    p.add_argument('--exp_degree_state', type=int, default=0,
                   help='Solo --model sparse_state: grado del grafo EXPANDER (0 = sin '
                        'expander). Las aristas expander se agregan al grafo con una relacion '
                        'RESERVADA propia R_exp = num_relation+1, nunca una relacion real del '
                        'KG (una arista aleatoria no es un hecho). Se agregan antes del CSR, '
                        'asi que la busqueda las trata como aristas mas. ⚠️ Lista negra #2 y '
                        '#8 de CLAUDE.md: el expander ya se refuto en inductivo (metia ruido) '
                        'y en FB15k-237 transductivo (Delta +0.0008, dentro del ruido). Se '
                        're-testea sobre el modelo NUEVO, que es otro punto de partida.')
    p.add_argument('--indicator', choices=['onehot', 'ppr'], default='onehot',
                   help="Condicion de borde de los nodos NO fuente. 'onehot' = cero (labeling "
                        "trick de NBFNet). 'ppr' = embedding del bin de PageRank personalizado, "
                        "que es lo que A*Net usa SOLO en wikikg2 (`indicator_func: ppr`): "
                        '"instead of using a boundary condition of mostly zeros, we find it is '
                        'better to incorporate distance information". Importa a escala: con '
                        'alpha=0.2 %% la busqueda alcanza <=7.7 %% de N y el resto llegaria al '
                        'readout sin informacion alguna.')
    p.add_argument('--num_indicator_bin', type=int, default=10,
                   help='Bins logaritmicos del PPR (su `num_indicator_bin`, default 10).')
    p.add_argument('--edge_dropout_p', type=float, default=0.0,
                   help='Elimina al azar esta fraccion de candidatas en TRAIN. A*Net usa 0.2 '
                        'en wikikg2 (`edge_dropout: 0.2`), aplicado como Dropout sobre '
                        'edge_weight; sin edge_weight el equivalente es sacarlas del pool.')
    p.add_argument('--break_tie', action='store_true',
                   help='Rompe los EMPATES del top-k al azar en vez de por posicion. A*Net lo '
                        'activa en wikikg2 (`break_tie: yes`). ⚠️ Sin esto los empates se '
                        'rompen por orden del CSR = orden del archivo del dataset, que es el '
                        'MISMO sesgo arbitrario que ya costo caro dos veces (desalojo por id '
                        'global y edge_cap). En las primeras capas casi todos los candidatos '
                        'empatan en la prioridad `base`, asi que el sesgo es masivo.')
    p.add_argument('--test_top_nodes', type=int, default=0,
                   help='Solo --model sparse_state: K en EVALUACION (0 = el mismo de train). '
                        'A*Net usa presupuestos distintos en wikikg2: node_ratio 0.002 al '
                        'entrenar y test_node_ratio 0.01 al evaluar (5x mas nodos). En sus '
                        'configs de FB15k-237/WN18RR/YAGO los dos coinciden.')
    p.add_argument('--test_edge_budget', type=int, default=0,
                   help='Solo --model sparse_state: L en EVALUACION (0 = escala con '
                        '--test_top_nodes para mantener beta constante).')
    # ---- semana 2 del plan (OBJETIVOS_Y_PLAN_WWW.md §4), solo --model sparse_state ----
    # Todos con init CERO: activarlos deja el forward inicial bit a bit igual (diag_global_equiv.py).
    p.add_argument('--global_tokens', type=int, default=0,
                   help='Canal GLOBAL del GT: M tokens inductores por query (Set Transformer / '
                        'Perceiver) que atienden sobre el conjunto activo y sobre los que luego '
                        'atienden los nodos. 0 = sin canal global (modelo actual). Es lo unico que '
                        'saca al modelo de la clase C-MPNN/rawl2 sin embeddings de entidad.')
    p.add_argument('--global_from', type=int, default=2,
                   help='Primera capa (0-based) con canal global; las capas 0-1 son degeneradas.')
    p.add_argument('--global_pool', choices=['topk', 'active'], default='topk',
                   help="Sobre que nodos hacen POOL los tokens: los K expandidos en la capa "
                        "('topk', escala a wikikg2) o todos los S activos ('active').")
    p.add_argument('--hop_pe', action='store_true',
                   help='PE relativa de SALTO: cada nodo recibe al activarse el embedding de la '
                        'capa en la que entro (su distancia al head). Sin identidad de entidad.')
    p.add_argument('--rel_frontier', type=float, default=0.0,
                   help='lambda de la FRONTERA CONSCIENTE DE LA RELACION: suma lambda * '
                        '<q_emb, rel_compat[r]> a la prioridad del destino al elegir aristas, y el '
                        'mismo termino al logit de la atencion (weight sharing). 0 = apagado.')
    p.add_argument('--fallback', choices=['scalar', 'ppr'], default='scalar',
                   help="Score de los nodos NO visitados: 'scalar' (historico, todos iguales) o "
                        "'ppr' (+ termino aprendido por bin de PageRank personalizado desde el head; "
                        "ordena la cola del ranking por posicion global, ataca el MR en WN18RR).")
    p.add_argument('--grad_ckpt', action='store_true',
                   help='Solo --model sparse_state: recomputa las activaciones de cada capa en '
                        'el backward en vez de retenerlas. Medido (92610): la memoria crece '
                        '3.5 GB POR CAPA de forma lineal y son todas activaciones por arista '
                        '=> deberia bajar de ~22.8 GB a ~4-5 GB, a cambio de ~30 %% de tiempo. '
                        'Es lo que habilita subir el batch de 8 hacia los 40 de A*Net.')
    p.add_argument('--evict', choices=['id', 'prio', 'keep'], default='id',
                   help="Solo --model sparse_state: a quien se DESALOJA cuando el activo "
                        "desborda --node_slots. 'id' (historico) se queda con los ids globales "
                        "mas chicos -- arbitrario, no tiene que ver con relevancia; en YAGO "
                        "deja el 34.6 %% de las respuestas sin score. 'prio' se queda con los "
                        "S de mayor prioridad -- MEDIDO Y ROTO (92457, valid_mrr 0.073): compara un score "
                        "aprendido contra la constante `base` y borra el estado cada capa. "
                        "'keep' NUNCA desaloja a un incumbente; los nuevos llenan lo que sobra "
                        "por orden de la heuristica A*. A*Net no desaloja nunca (VirtualTensor sobre N).")
    p.add_argument('--rel_readout', action='store_true',
                   help='Readout global ESPECIFICO POR RELACION. Es la unica salida PROBADA '
                        'de la clase de NBFNet: Huang et al. (NeurIPS 2023) muestran que las '
                        'C-MPNN estan exactamente caracterizadas por rawl2, y su Teorema 5.3 '
                        'que el readout global aumenta ESTRICTAMENTE el poder expresivo. '
                        'OJO: el readout global PLANO les DEGRADO el resultado; el que '
                        'funciona (SOTA en FB15k-237) es el especifico por relacion, que es '
                        'el implementado. Cuesta O(B*N*d) y NO toca aristas => compatible '
                        'con la poda. Init en CERO: arranca identico al baseline.')
    p.add_argument('--node_ratio', type=float, default=0.1,
                   help='Solo con --model sparse_pruned: fraccion de NODOS que se '
                        'seleccionan por query en cada capa (primera etapa de la seleccion '
                        'de A*Net). Las aristas candidatas son las SALIENTES de esos nodos, '
                        'y se ordenan por la prioridad del nodo DESTINO. A*Net usa 0.1 en '
                        'FB15k-237 (§3.2 de A_star_Net_resumen.md).')
    p.add_argument('--prune_attn', type=str, default='softmax', choices=['softmax','sigmoid'],
                   help='Solo con --model sparse_pruned. `sigmoid` agrega SIN normalizar: '
                        'bajo poda el segment-softmax normaliza sobre las aristas elegidas y '
                        'esconde que se descarto evidencia (un nodo con 100 entrantes de las '
                        'que se eligen 5 queda como si tuviera 5 vecinos). NBFNet/A*Net suman '
                        'y por eso su poda es sin perdida. Ver SESSION_NOTES 2026-08-25.')
    p.add_argument('--dependent', action='store_true',
                   help='Representacion relacional GENERADA DESDE LA QUERY, una vez por '
                        'batch: rel = Linear(d, R*d)(q) -> (B,R,d), usada como modulacion '
                        'DistMult del valor. Es el mecanismo de NBFNet/A*Net '
                        '(`dependent: yes`, que sus configs usan en FB15k-237, YAGO3-10 y '
                        'ogbl-wikikg2; WN18RR es el unico con `no`). Sin el flag el modelo '
                        'usa la tabla estatica rel_value, que equivale a su `dependent: no`. '
                        'Se aplica por QUERY, no por arista => O(B*R*d^2), a diferencia de '
                        '--rel_param lowrank, que es por arista y cuesta 2.3x. Init peso 0 / '
                        'bias 1 => forward identico al modo estatico al arranque. '
                        'INCOMPATIBLE con --rel_param lowrank y con --exp_typing ultra/path.')
    p.add_argument('--rel_rank', type=int, default=4,
                   help='Rango k de la correccion de --rel_param lowrank. Anade '
                        '2*H*R*hd*k params por capa. k=hd da rango completo.')
    p.add_argument('--remove_one_hop', action='store_true',
                   help='En train quita TODAS las aristas directas entre h y t de la query '
                        '(no solo la de la relacion exacta). Es `remove_one_hop: yes` de '
                        'NBFNet, que su config inductivo activa. Afecta al 19.6/23.1/31.6 %% '
                        'de los triples de train en FB15k-237 ind v1 / ind v2 / transductivo, '
                        'y a ~1 %% en WN18RR. Solo train; eval usa el grafo completo.')
    p.add_argument('--edge_drop', type=float, default=0.0,
                   help='DropEdge estructural: fraccion de aristas del grafo eliminadas al '
                        'azar en cada forward de TRAIN (0.0 = off, comportamiento previo). '
                        'En eval el grafo va completo. Regulariza contra el sobre-ajuste a '
                        'la topologia del train graph (firma valid^/testv del sparse).')
    p.add_argument('--learning_rate', type=float, default=5e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--test_batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--max_epochs', type=int, default=20)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--precision', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint_save_path', type=str, default='./experiments/gt')
    p.add_argument('--limit_train_batches', type=float, default=1.0,
                   help='PL Trainer limit_train_batches (smoke test: e.g. 20)')
    p.add_argument('--limit_val_batches', type=float, default=1.0,
                   help='PL Trainer limit_val_batches (smoke test: e.g. 0 to skip)')
    p.add_argument('--eval_subsample', type=int, default=0,
                   help='Solo wikikg2: evaluar sobre N items ALEATORIOS del set de OGB en vez de '
                        'los 1 197 086 completos (~52 h a batch 8). Es lo que hace A*Net con '
                        '`fast_test: 5000` (random_split con semilla 1024). Preferir esto a '
                        '--limit_test_batches, que toma un PREFIJO y sesga: el prefijo de 2 000 '
                        'triples tiene grado mediano de cola 1 939 contra 1 230 del total. '
                        'Error estandar del MRR: ~0.0064 con 5 000 items, ~0.0018 con 64 000.')
    p.add_argument('--eval_seed', type=int, default=1024,
                   help='Semilla del submuestreo de evaluacion (la de ellos es 1024).')
    p.add_argument('--limit_test_batches', type=float, default=1.0,
                   help='PL Trainer limit_test_batches. ⚠️ EN wikikg2 ES OBLIGATORIO MIRARLO: '
                        'el test de OGB son 1 197 086 items (598 543 triples x 2 direcciones) '
                        'y a batch 8 son 149 636 batches = 22 h. Su config usa `fast_test: 5000` '
                        'por el mismo motivo.')
    p.add_argument('--resume_from', type=str, default='',
                   help='Path a un checkpoint (last.ckpt) para reanudar entrenamiento')
    p.add_argument('--eval_only', action='store_true',
                   help='Saltar fit; solo correr test sobre --eval_ckpt (eval-only).')
    p.add_argument('--dump_ranks', type=str, default='',
                   help='Path donde guardar los ranks POR QUERY del test (h, r, t, rank) '
                        'mas el edge_index del grafo. Necesario para evaluar ESTRATIFICADO '
                        '(p.ej. por grado de la respuesta), que es donde vive el modo de '
                        'fallo dominante; el MRR agregado lo esconde. Ver SESSION_NOTES '
                        '2026-08-22 (b).')
    p.add_argument('--eval_ckpt', type=str, default='',
                   help='Path al checkpoint a testear cuando --eval_only.')
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    inductive = is_inductive(args.data_path)
    dm = KGDataModule(args.data_path, inductive, args.num_workers,
                      args.batch_size, args.test_batch_size,
                      eval_subsample=args.eval_subsample, eval_seed=args.eval_seed)
    model = GTLightningModule(dm.num_relation, args.num_layer, args.hidden_dim,
                              args.num_heads, args.drop, args.learning_rate, args.weight_decay,
                              model=args.model, aggregate=args.aggregate,
                              use_rwse=args.use_rwse, rwse_dim=args.rwse_dim,
                              use_lappe=args.use_lappe, lappe_dim=args.lappe_dim,
                              use_source_rw=args.use_source_rw, source_rw_dim=args.source_rw_dim,
                              use_rpb=args.use_rpb, rpb_hops=args.rpb_hops, rpb_dim=args.rpb_dim,
                              exp_degree=args.exp_degree, attn=args.attn,
                              exp_typing=args.exp_typing, exp_path_len=args.exp_path_len,
                              filtered_ce=args.filtered_ce, edge_drop=args.edge_drop,
                              remove_one_hop=args.remove_one_hop,
                              rel_param=args.rel_param, rel_rank=args.rel_rank,
                              dependent=args.dependent, edge_ratio=args.edge_ratio,
                              prune_attn=args.prune_attn, node_ratio=args.node_ratio,
                              rel_readout=args.rel_readout, node_slots=args.node_slots,
                              top_nodes=args.top_nodes, edge_budget=args.edge_budget,
                              edge_cap=args.edge_cap, evict=args.evict,
                              grad_ckpt=args.grad_ckpt,
                              test_top_nodes=args.test_top_nodes,
                              indicator=args.indicator,
                              num_indicator_bin=args.num_indicator_bin,
                              edge_dropout_p=args.edge_dropout_p,
                              break_tie=args.break_tie,
                              exp_degree_state=args.exp_degree_state,
                              test_edge_budget=args.test_edge_budget,
                              global_tokens=args.global_tokens, global_from=args.global_from,
                              global_pool=args.global_pool, hop_pe=args.hop_pe,
                              rel_frontier=args.rel_frontier, fallback=args.fallback,
                              loss=args.loss, num_negative=args.num_negative,
                              adversarial_temperature=args.adversarial_temperature,
                              dump_ranks=args.dump_ranks)

    ckpt = ModelCheckpoint(dirpath=args.checkpoint_save_path, monitor='valid_mrr',
                           mode='max', save_top_k=1, every_n_epochs=1, verbose=True,
                           save_last=True)
    trainer = pl.Trainer(accelerator='gpu', devices=args.devices, precision=args.precision,
                         max_epochs=args.max_epochs, callbacks=[ckpt],
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         num_sanity_val_steps=0, check_val_every_n_epoch=1, logger=False, limit_test_batches=args.limit_test_batches)
    if args.eval_only:
        print(f"[eval_only] testeando ckpt: {args.eval_ckpt}")
        trainer.test(model, datamodule=dm, ckpt_path=args.eval_ckpt)
        return
    trainer.fit(model, datamodule=dm, ckpt_path=(args.resume_from or None))
    print(f"[best ckpt] {ckpt.best_model_path}  valid_mrr={ckpt.best_model_score}")
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path)


if __name__ == '__main__':
    main()
