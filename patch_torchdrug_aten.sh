#!/bin/bash
# PARCHE al torchdrug INSTALADO en el env aislado `astarnet` (site-packages), NO a los repos.
# torchdrug 0.2.1 esta congelado en torch ~1.8 e incluye <ATen/SparseTensorUtils.h>; PyTorch
# movio ese header a <ATen/native/SparseTensorUtils.h> (esta en torch 2.1.0 con el MISMO
# namespace `at::sparse`, verificado) => es un cambio de RUTA, no de API.
# Es la 4a incompatibilidad que SESSION_NOTES (2026-08-08 c) registro como bloqueante.
set -e
E=/home/jreutter/miniconda3/envs/astarnet
D=$E/lib/python3.10/site-packages/torchdrug/layers/functional/extension
for f in spmm.h rspmm.h; do
  [ -f "$D/$f.orig" ] || cp "$D/$f" "$D/$f.orig"
  sed -i 's|#include <ATen/SparseTensorUtils.h>|#include <ATen/native/SparseTensorUtils.h>|' "$D/$f"
  grep -n "SparseTensorUtils" "$D/$f"
done
# El cache de la extension JIT hay que limpiarlo o reusa los .o rotos.
rm -rf ~/.cache/torch_extensions/py310_cu121/spmm ~/.cache/torch_extensions/py310_cu121/rspmm 2>/dev/null || true
echo "parche aplicado (backups .orig)"
