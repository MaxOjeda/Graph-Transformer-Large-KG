#!/bin/bash
# Env AISLADO `astarnet` para correr los repos ORIGINALES (A*Net y NBFNet/torchdrug).
# Motivo: A*Net llama functional.generalized_rspmm de torchdrug en 9 sitios de
# reasoning/layer.py y NO tiene rama alternativa => exige el kernel CUDA JIT => nvcc.
# El cluster no trae nvcc ni module system (verificado 2026-08-19).
#
# NO toca `attention` (harness propio) ni `venv_nbfnet` (NBFNet-PyG).
# --override-channels -c conda-forge -c nvidia: evita los canales comerciales de Anaconda
# (su ToS no esta aceptado y su licencia para instituciones no es libre). Todo lo demas por pip.
# NOTA: `set -u` NO se puede usar: el activate.d de gcc_linux-64 (que arrastra el toolkit
# CUDA) referencia SYS_SYSROOT sin definir y aborta el script en `conda activate`.
set -ex
source /home/jreutter/miniconda3/etc/profile.d/conda.sh

conda create -n astarnet --override-channels -c conda-forge -y python=3.10
# CUDA 12.1 para que nvcc coincida con torch 2.1.0+cu121.
conda install -n astarnet --override-channels -c conda-forge -c nvidia -y \
    cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-cccl=12.1 cuda-nvrtc-dev=12.1 libcusparse-dev

conda activate astarnet
export CUDA_HOME="$CONDA_PREFIX"
python -m pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
python -m pip install --no-cache-dir torchdrug==0.2.1 ogb easydict pyyaml "numpy<2" "setuptools<81"

echo "=== LISTO ==="
which nvcc && nvcc --version | tail -2
python - <<'PY'
import torch, torch_scatter, torch_sparse, torchdrug
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| torchdrug", torchdrug.__version__)
PY
