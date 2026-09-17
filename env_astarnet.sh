# Setup del env AISLADO `astarnet` (repos ORIGINALES: A*Net y NBFNet/torchdrug). Uso:
#   source env_astarnet.sh
# NO reemplaza a env.sh (que es para nuestro harness, env `attention`).
export PYTHONNOUSERSITE=1
source /home/jreutter/miniconda3/etc/profile.d/conda.sh
conda activate astarnet
export CUDA_HOME="$CONDA_PREFIX"
# Los paquetes CUDA de conda-forge dejan los headers de las librerias de math
# (cublas_v2.h, cusparse.h, ...) en targets/<arch>/include, que torch.utils.cpp_extension
# NO agrega al include path (solo usa $CUDA_HOME/include). Sin esto, la compilacion JIT del
# kernel rspmm de torchdrug falla con "cublas_v2.h: No such file or directory".
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
