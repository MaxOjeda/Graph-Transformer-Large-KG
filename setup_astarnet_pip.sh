#!/bin/bash
# Continuacion de setup_astarnet_env.sh (el toolkit CUDA ya quedo instalado).
set -ex
source /home/jreutter/miniconda3/etc/profile.d/conda.sh
conda activate astarnet
export CUDA_HOME="$CONDA_PREFIX"
python -m pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
python -m pip install --no-cache-dir torchdrug==0.2.1 ogb easydict pyyaml "numpy<2" "setuptools<81"
echo "=== LISTO ==="
which nvcc; nvcc --version | tail -2
python -c "import torch,torch_scatter,torch_sparse,torchdrug; print('torch',torch.__version__,'| cuda',torch.version.cuda,'| torchdrug',torchdrug.__version__)"
