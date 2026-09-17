"""Wrapper para correr NBFNet-PyG (reimplementacion OFICIAL del mismo autor) sin modificarlo.

Por que este repo y no NBFNet/ (torchdrug): el cluster patagon no tiene CUDA toolkit y
torchdrug 0.2.1 arrastra cuatro incompatibilidades con torch 2.1 (metaclase PyG, header
ATen/SparseTensorUtils.h movido, ninja, y `Graph` sin `.is_meta` al cargar el checkpoint).
NBFNet-PyG corre sobre torch + PyG + torch-scatter, que es exactamente el stack instalado.
Misma config (input_dim 32, hidden_dims [32]x6, distmult, pna, short_cut, layer_norm,
dependent, remove_one_hop, lr 5e-3, batch 64, 20 epocas, BCE con 32 negativos).

Unico ajuste, y es de INFRAESTRUCTURA, no de modelo (`NBFNET_PYG_NO_RSPMM=1`):
NBFNet-PyG trae su propio kernel CUDA `nbfnet/rspmm/`, que se compila en JIT y necesita
nvcc — ausente en todo el cluster (verificado en el login y en nodeGPU01). Su propio codigo
ya define la salida en `nbfnet/layers.py:69-72`:

    def propagate(self, edge_index, size=None, **kwargs):
        if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
            # the rspmm cuda kernel only works for TransE and DistMult message functions
            # otherwise we invoke separate message & aggregate functions
            return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)

o sea la ruta `message()` + `aggregate()` estandar de PyG. El kernel es, en palabras del
propio comentario del repo (linea 154), la "fused computation of message and aggregate
steps": misma matematica, mas rapida y con menos memoria (O(|V|d) en vez de O(|E|d)).
Forzar el `super().propagate` es por lo tanto tomar SU camino de referencia, no reimplementar
nada. Cuesta velocidad y memoria; en FB15k-237 v2 (2608 nodos, 19478 aristas) eso es barato.

Uso identico al original:
  python run_nbfnet_pyg.py -c NBFNet-PyG/config/inductive/fb15k237.yaml --gpus [0] --version v2
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'NBFNet-PyG')
RUN_PY = os.path.join(REPO, 'script', 'run.py')

if not os.path.exists(RUN_PY):
    sys.exit(f'no encuentro {RUN_PY}')

sys.path.insert(0, REPO)

if os.environ.get('NBFNET_PYG_NO_RSPMM') == '1':
    from torch_geometric.nn.conv import MessagePassing
    from nbfnet import layers as _layers

    _layers.GeneralizedRelationalConv.propagate = MessagePassing.propagate
    print('[wrapper] NBFNET_PYG_NO_RSPMM=1 -> GeneralizedRelationalConv.propagate forzado a '
          'MessagePassing.propagate (ruta message+aggregate de PyG, sin kernel rspmm)',
          flush=True)

# run.py resuelve rutas relativas al cwd (working_dir.tmp, checkpoints).
os.chdir(REPO)
sys.argv[0] = RUN_PY
runpy.run_path(RUN_PY, run_name='__main__')
