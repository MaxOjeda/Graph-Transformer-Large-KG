#!/bin/bash
# Continuacion 2: torch, torch_scatter y torch_sparse ya quedaron instalados.
# Faltaba torch_cluster (dependencia de torchdrug): pip intentaba COMPILARLO en un env de
# build aislado que no ve torch => ModuleNotFoundError. Se instala desde el indice de wheels
# precompiladas de PyG, igual que scatter/sparse.
set -ex
source /home/jreutter/miniconda3/etc/profile.d/conda.sh
conda activate astarnet
export CUDA_HOME="$CONDA_PREFIX"
python -m pip install --no-cache-dir torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
python -m pip install --no-cache-dir --no-build-isolation torchdrug==0.2.1
python -m pip install --no-cache-dir ogb easydict pyyaml "numpy<2" "setuptools<81" torch_geometric==2.4.0
echo "=== LISTO ==="
python -c "import torch,torch_scatter,torch_sparse,torch_cluster,torchdrug,torch_geometric; print('torch',torch.__version__,'| torchdrug',torchdrug.__version__,'| PyG',torch_geometric.__version__)"
