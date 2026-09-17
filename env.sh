# Setup obligatorio para el cluster patagon (patagon-master). Uso: `source env.sh`
#
# Cluster nuevo (2026-07-31): sin conda/pip/módulos CUDA preinstalados. Se instaló
# Miniconda propio en $HOME y un env 'attention' con las versiones de requirements.txt
# (torch 2.1.0+cu121, PyG 2.4.0, pytorch_lightning 1.9.1, torchmetrics 0.11.4). El modelo
# no compila kernels (pure PyTorch, sin rspmm) -> no hace falta module load de CUDA/gcc,
# solo el runtime CUDA que trae la wheel de torch. numpy fijado <2 (torch 2.1.0 está
# compilado contra la ABI de numpy 1.x). setuptools fijado <81 (pkg_resources, requerido
# por pytorch_lightning 1.9.1, fue removido en setuptools 81+).
#
# Partición GPU: AI (nodeGPU01, 8x A100 40GB). Cuenta: puc. QOS: external (hasta 8 GPU,
# 4 jobs). Usar --gpus=N en sbatch/srun (NO --gres=gpu:...: el cluster lo rechaza).

export PYTHONNOUSERSITE=1

source /home/jreutter/miniconda3/etc/profile.d/conda.sh
conda activate attention
