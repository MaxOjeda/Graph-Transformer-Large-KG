"""Wrapper para correr el repo ORIGINAL de NBFNet (NBFNet/script/run.py) sin modificarlo.

Por que hace falta: torchdrug parchea `torch.utils.data.Dataset` con una metaclase propia
(su registro Configurable). Si torchdrug se importa ANTES que torch_geometric, la clase
`torch_geometric.data.Dataset` —que hereda de `torch.utils.data.Dataset` y de `ABC`— explota
con "TypeError: metaclass conflict". El repo original importa `torchdrug.core` (run.py) antes
de `ogb.linkproppred` (nbfnet/dataset.py), que a su vez arrastra PyG => conflicto.

Solucion: importar torch_geometric PRIMERO, de modo que sus clases queden creadas antes del
parche, y recien despues delegar en script/run.py tal cual esta. No se toca ni un byte del
repo original (CLAUDE.md: "NBFNet/ ... repo ORIGINAL. No modificar").

Uso identico al original, con los mismos flags:
  python run_nbfnet_orig.py -c NBFNet/config/inductive/fb15k237.yaml --gpus [0] --version v2 --seed 1024
"""
import os
import runpy
import sys

import torch_geometric  # noqa: F401  <- DEBE ir antes de cualquier import de torchdrug

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_PY = os.path.join(HERE, 'NBFNet', 'script', 'run.py')

if not os.path.exists(RUN_PY):
    sys.exit(f'no encuentro {RUN_PY}')

# --- NBFNET_NO_RSPMM=1: evita el kernel CUDA de torchdrug -------------------------------
# El cluster patagon NO tiene CUDA toolkit (no hay nvcc en el login ni en nodeGPU01, ni
# /usr/local/cuda). torchdrug compila spmm/rspmm en JIT: en CPU compila solo los .cpp (OK,
# verificado), pero en GPU necesita nvcc para los .cu => imposible.
#
# El propio codigo de NBFNet trae la salida. `GeneralizedRelationalConv.message_and_aggregate`
# (nbfnet/layer.py:108-110) arranca con:
#     if graph.requires_grad or self.message_func == "rotate":
#         return super(...).message_and_aggregate(graph, input)
# o sea YA usa la ruta generica de torchdrug (message + aggregate con scatter) cuando hay
# grafo diferenciable; el `generalized_rspmm` es la version FUSIONADA y optimizada de esa
# misma operacion, que solo se toma en el camino sin gradiente (eval). Forzar siempre la
# ruta del `super()` es por lo tanto **la misma matematica escrita por ellos**, no una
# reimplementacion nuestra: cambia velocidad y memoria, no el resultado.
#
# Se deja detras de un env var para que quede explicito en los logs y para no alterar el
# camino CPU, que si puede usar el kernel compilado.
if os.environ.get('NBFNET_NO_RSPMM') == '1':
    sys.path.insert(0, os.path.join(HERE, 'NBFNet'))
    from torchdrug import layers as _td_layers
    from nbfnet import layer as _nbf_layer

    _nbf_layer.GeneralizedRelationalConv.message_and_aggregate = \
        _td_layers.MessagePassingBase.message_and_aggregate
    print('[wrapper] NBFNET_NO_RSPMM=1 -> GeneralizedRelationalConv.message_and_aggregate '
          'forzado a la ruta generica de torchdrug (sin kernel rspmm)', flush=True)

# run.py hace sys.path.append(dirname(dirname(__file__))) para encontrar el paquete `nbfnet`,
# y torchdrug resuelve rutas relativas al cwd => se corre desde dentro de NBFNet/.
os.chdir(os.path.join(HERE, 'NBFNet'))
sys.argv[0] = RUN_PY
runpy.run_path(RUN_PY, run_name='__main__')
