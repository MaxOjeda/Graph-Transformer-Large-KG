"""Wrapper para correr A*Net (repo ORIGINAL, sin modificarlo).

A*Net: "A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs"
(Zhu et al., NeurIPS 2023). Repo `AStarNet/`, sobre torchdrug.

POR QUE HACE FALTA UN WRAPPER — un solo problema, de orden de imports:

  `AStarNet/reasoning/dataset.py:5` hace `from ogb import linkproppred`, que importa
  PyG. Si torchdrug ya se cargo (y `script/run.py` lo hace en su linea 6, ANTES de
  importar `reasoning`), PyG explota al definir su clase Dataset:

      torch_geometric/data/dataset.py:20, class Dataset(torch.utils.data.Dataset, ABC)
      TypeError: metaclass conflict

  Causa: torchdrug inyecta su metaclase en `torch.utils.data.Dataset`, y despues es
  incompatible con la ABCMeta de la clase de PyG. Si PyG se importa PRIMERO, la clase ya
  esta creada y el conflicto no ocurre. Verificado 2026-08-19: con torchdrug primero da
  metaclass conflict; con PyG primero, no.

  Es exactamente la "metaclase PyG/torchdrug" que SESSION_NOTES (2026-08-08 c) registro
  como una de las cuatro incompatibilidades que hacian inusable el repo `NBFNet/`. Al
  menos esta tiene arreglo y no requiere tocar el repo.

ENTORNO: env `astarnet` (ver `setup_astarnet_env.sh`), que es el unico con `nvcc`.
A*Net llama `functional.generalized_rspmm` de torchdrug en 9 sitios de
`reasoning/layer.py` (55, 163-175, 334) y **no tiene rama alternativa** como si la tiene
NBFNet-PyG (`layers.py:69-72`) => el kernel CUDA JIT es OBLIGATORIO, no opcional.

Uso identico al original:
  conda activate astarnet
  python run_astarnet.py -c config/transductive/fb15k237_astarnet.yaml --gpus [0] --seed 1024
"""
import os
import runpy
import sys

# ---- 1. PyG ANTES que torchdrug. Es la razon de ser de este wrapper: no mover. ----
import torch_geometric.data  # noqa: F401  (efecto secundario deliberado)

# ---- 2. Parche `is_meta` (2026-08-19, job 91869). ----
# torchdrug registra objetos `Graph` en el state_dict del modelo. Al cargar un checkpoint,
# `torch.nn.Module._load_from_state_dict` (torch 2.1, module.py:2024) evalua
# `param.is_meta`, atributo que existe en Tensor pero NO en el `Graph` de torchdrug 0.2.1
# (congelado en torch ~1.8) => AttributeError: 'Graph' object has no attribute 'is_meta'.
#
# Es la CUARTA incompatibilidad que SESSION_NOTES (2026-08-08 c) registro, y la unica que
# no aparece hasta el FINAL del entrenamiento: `AStarNet/script/run.py:36` hace
# `solver.load("model_epoch_%d.pth" % best_epoch)` justo antes de evaluar => el job 91869
# entreno las 20 epocas x 3 semillas y murio al cargar, sin llegar al test.
#
# `Graph` ya pasa el chequeo de `.shape` previo (module.py:2017); solo falta este flag, y
# False es el valor correcto: un Graph nunca esta en el meta device.
from torchdrug.data import Graph as _TDGraph  # noqa: E402
if not isinstance(getattr(_TDGraph, 'is_meta', None), bool):
    _TDGraph.is_meta = False

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'AStarNet')
RUN_PY = os.path.join(REPO, 'script', 'run.py')

if not os.path.exists(RUN_PY):
    sys.exit(f'no encuentro {RUN_PY}')

if 'CUDA_HOME' not in os.environ and 'CONDA_PREFIX' in os.environ:
    # torch.utils.cpp_extension busca nvcc via CUDA_HOME; en el env `astarnet` el
    # toolkit vive en el propio prefijo de conda.
    os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']

sys.path.insert(0, REPO)

# run.py resuelve rutas relativas al cwd (working_dir, checkpoints), igual que NBFNet.
os.chdir(REPO)
sys.argv[0] = RUN_PY
runpy.run_path(RUN_PY, run_name='__main__')
