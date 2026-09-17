# CLAUDE.md — Attention (Graph Transformer para KGC, desde cero)

Briefing operacional para sesiones de Claude Code. Conciso a propósito.

---

> 🚩 **PRIORIDAD VIGENTE (acordada en la defensa de candidatura, 2026-08-18 — APROBADA).**
> **El régimen que manda ahora es el TRANSDUCTIVO**, con el objetivo de enviar una
> publicación a **ACM WWW, deadline la primera semana de octubre de 2026**.
> **El inductivo se retoma DESPUÉS del envío.**
>
> Consecuencia operativa: todo lo que sigue en este archivo y en `GOALS.md` está escrito
> alrededor del **sanity check inductivo** ("¿el full attention supera a NBFNet en FB15k-237
> ind v1?"), que es la etapa **anterior**. Sigue siendo válido como registro de lo medido,
> pero **ya no fija la prioridad**. Ningún experimento inductivo se lanza antes de octubre
> sin justificación explícita — la GPU es el recurso escaso y las corridas transductivas
> cuestan **días**, no horas.
>
> 🔁 **LOSS PRINCIPAL = BCE-k (2026-08-19).** Los cuatro papers de referencia (NBFNet, ULTRA,
> A\*Net, TRIX) entrenan con **BCE + 32 negativos muestreados + self-adversarial weighting**.
> Desde hoy ésa es la loss primaria del proyecto y la **CE de grafo completo queda como
> alternativa** que se reporta aparte. Flag: **`--loss bce`**, port bit a bit idéntico a
> `NBFNet-PyG/script/run.py:57-68`. ⚠️ El **default de `--loss` sigue siendo `ce`** por
> compatibilidad con los ~40 scripts viejos ⇒ **todo script nuevo debe pasar `--loss bce`
> explícitamente, baselines incluidos**. Todos los resultados de esta bitácora anteriores al
> 2026-08-19 son con **CE de grafo completo**. Ver entrada 2026-08-19.
>
> 🔒 **REGLA DEL USUARIO (2026-08-21): TODOS los experimentos van con `--loss bce`, salvo que
> se pida explícitamente lo contrario.** Incluye re-correr los experimentos de Graph
> Transformer, que hoy están todos medidos con CE. Como el default del flag sigue en `ce`,
> **la regla se cumple pasando `--loss bce` en cada script** — verificarlo antes de lanzar.
>
> **Baselines del paper: NBFNet y A\*Net** (transductivo: MRR 0.415 y 0.411 en FB15k-237;
> 0.551 y 0.549 en WN18RR).
>
> **Bloqueador #1 del paper transductivo — 3 de 4 brazos CERRADOS (2026-08-21).** Baselines
> externos medidos con los repos ORIGINALES (env `astarnet`, BCE-k, 20 ép, n=3, semillas
> 1024/1025/1026), **los tres reproducen la literatura dentro de 1.1 %**:
>
> | brazo | **nuestro (n=3)** | *paper* | Δ |
> |---|---:|---:|---:|
> | NBFNet WN18RR | **0.5466 ± 0.0015** (MR **652.9**) | *0.551* | −0.0044 |
> | A\*Net WN18RR | **0.5431 ± 0.0013** (MR 5853.7) | *0.549* | −0.0059 |
> | A\*Net FB15k-237 | **0.4084 ± 0.0014** (MR 497.8) | *0.411* | −0.0026 |
> | **NBFNet FB15k-237** | **PENDIENTE** (~18.4 h/semilla) | *0.415* | — |
>
> ⇒ **Falta solo NBFNet en FB15k-237**, que es justo el dataset donde vive el mejor número
> transductivo del proyecto (sparse + `--remove_one_hop`, test 0.42893, n=1) ⇒ ese número
> **sigue sin ser interpretable**. En WN18RR el baseline propio ya existe.
> ⚠️ **CORRECCIÓN**: el "este harness corre 0.011–0.016 **por encima** de las re-evaluaciones
> publicadas" es de **nuestro** harness con **CE de grafo completo en inductivo**; los repos
> originales con BCE-k en transductivo caen 0.003–0.006 **por debajo** del número publicado.
> No mezclar los dos hechos. El techo del paper es el **baseline propio medido**, no la tabla
> publicada.
> ⚠️ **NBFNet WN18RR es un piso**: 2 de 3 semillas eligen la época 20 de 20 (subentrenado a las
> 20 épocas de su propia receta). Declararlo al citar el número absoluto.
> Detalle y contrastes: entrada **2026-08-21** de `SESSION_NOTES.md`; cola: entrada 2026-08-18.

---

## Qué es este proyecto (reinicio 2026-06-21)

Proyecto nuevo, **desde cero**. Dos proyectos previos quedaron botados y NO se
construye sobre ellos:

- **Exphormer-Max**: adaptar Exphormer (expander graphs) a KGC. Bien en transductivo,
  **pésimo en inductivo** (los expander graphs metían ruido). Línea descartada.
- **Knowformer-Expander**: construir sobre KnowFormer. Tras meses se concluyó que su
  atención es **decorativa** (todo el peso está en el V-RMPNN stream ≈ NBFNet).
  Confuso e improductivo. Descartado.

**Pregunta de esta etapa (sanity check antes de invertir en sparse/lineal):**

> Un **Graph Transformer con atención FULL (densa, all-pairs O(N²))** —el techo de
> expresividad de cualquier atención sparse— ¿supera a **NBFNet** en **FB15k-237
> inductivo v1**? Si la versión densa **no** gana, una versión sparse (que es una
> restricción de la densa) no tiene caso, y habría que repensar todo el enfoque de
> "Graph Transformer para KGC".

Marco teórico que justifica el diseño: `transformer_vs_nbfnet.tex`. Resumen:
- **Caso (a)** transformer ciego a la estructura (atención sobre embeddings de nodo sin
  aristas): **rompe la inductividad** (no hay embeddings de entidad que transferir al
  grafo de test disjunto). Inviable → NO hacer esto.
- **Caso (b)** *graph* transformer con labeling trick + adyacencia relacional: **puede
  igualar a NBFNet en principio**, pero al usar aristas+relaciones+fuente está
  "reimplementando message passing con agregación aprendida". Este proyecto mide
  empíricamente si esa flexibilidad extra **mejora** o no.

---

## Arquitectura actual: RFAT (Relational Full-Attention Graph Transformer)

`src/model.py` (NUEVO, escrito desde cero — NO es KnowFormer). Una sola torre de
atención. **Sin V-RMPNN, sin QK-RMPNN, sin RSPMM kernel.**

- **Inductivo puro**: sin embeddings de entidad, sin positional encoding de nodo. Lo
  único compartido train/test son embeddings de **relación** (las relaciones sí se ven).
- **Labeling trick** (NBFNet): `x⁰_v = emb(r_q)` si `v == head`, si no `0`.
- **Cada capa** (multi-head, pre-LN), atención **densa all-pairs**
  `softmax(QKᵀ/√d + b_rel)` con:
  - (i) **bias escalar relacional** `b[head, rel]` en pares conectados por arista
    (estilo Graphormer: marca quién es vecino y por qué relación).
  - (ii) **corrección de valor relacional** `Σ_edges α·(V_w ⊙ g[rel])` (composición
    estilo DistMult; aporta el razonamiento por caminos).
  - residual + FFN.
- **Readout** puntual `MLP(x^L_v) → score (B, N)`. **Loss = CE de grafo completo**.
- La arista de la query (y su reversa) se quita del grafo en train (`graph_mask`).

El nodo destino `dst` agrega de `src` (arista `src --rel--> dst`), igual que el message
passing de NBFNet para predecir cola.

---

## Repositorio

```
Attention/
  src/
    model.py     # NUEVO. GraphTransformer + RelationalAttentionLayer. Editar aquí.
    data.py      # CONSERVADO de KnowFormer. Loaders KG transductivo + inductivo.
    metric.py    # CONSERVADO. MR / MRR / Hits.
    rspmm/       # kernel CUDA de KnowFormer. NO se usa en este proyecto.
  data/          # datasets: wn18rr, fb15k-237, nell-995, yago3-10, inductive/
  train.py       # NUEVO. Entry point limpio (argparse + PL module + datamodule).
  NBFNet/        # repo ORIGINAL de NBFNet (referencia/baseline). No modificar.
  Exphormer/     # repo ORIGINAL de Exphormer (referencia). No se usa.
  transformer_vs_nbfnet.tex   # análisis teórico que justifica el diseño. LEER.
  manuscrito_candidatura.md   # propuesta de tesis (referencia).
  CLAUDE.md
```

**Legacy (de KnowFormer, NO usar)**: `main.py`, `lightning.py` importan el viejo
`Knowformer` que ya no existe en `model.py` → quedan rotos a propósito. El entry point
nuevo es `train.py`. Los `.md`/`.sh` heredados (SESSION_LOG.md, PLAN_*.md, run_*.sh,
sbatch_*.sh) son del proyecto previo y NO son guía para este.

---

## Entorno

### Tres entornos, uno por codebase (NO mezclarlos)

| env | para qué | punto de entrada |
|---|---|---|
| **`attention`** | **nuestro harness** (`train.py`, `src/model.py`) | `source env.sh` |
| `venv_nbfnet` | **NBFNet-PyG** (reimplementación oficial, PyG) | `run_nbfnet_pyg.py` |
| **`astarnet`** | **repos ORIGINALES**: A\*Net y `NBFNet/` (torchdrug) | `source env_astarnet.sh` |

- **Python del harness**: `/home/jreutter/miniconda3/envs/attention/bin/python` (torch 2.1.0+cu121,
  PyG 2.4.0, pytorch_lightning 1.9.1, torchmetrics 0.11.4). ⚠️ **Corregido 2026-08-19**: esta nota
  decía `/nfs_ssd/mojeda_imfd/miniconda3/envs/knowformer/bin/python`, que es del cluster VIEJO y ya
  no existe.
- **Setup obligatorio**: `source env.sh`. ⚠️ **Corregido 2026-08-19**: esta nota decía que hacía
  `module load gnu12, cuda/12.6`. **El cluster nuevo (desde 2026-07-31) NO tiene module system ni
  CUDA preinstalado**; `env.sh` solo hace `PYTHONNOUSERSITE=1` + `conda activate attention`.
- **NO usar `conda run`** — la ruta directa al binario funciona.
- Nuestro modelo **no compila kernels** (no usa rspmm). Pure PyTorch ⇒ el env `attention` no
  necesita `nvcc`.
- **`nvcc` SÍ existe, pero solo en el env `astarnet`** (2026-08-19). Los repos originales exigen
  compilar el kernel CUDA `rspmm` de torchdrug; se instaló `cuda-nvcc 12.1` + gcc 11.4 en un env
  conda aislado. **Verificado: compila y corre.** Receta: `setup_astarnet_env.sh`,
  `setup_astarnet_pip{,2}.sh`, `patch_torchdrug_aten.sh`. Detalle de los 6 bloqueos y sus causas:
  entrada **2026-08-19 (c)** de `SESSION_NOTES.md`.
  ⚠️ Ojo con `gcc`: **CUDA 12.1 soporta GCC hasta 12.2**; con gcc 12.4 falla con errores de
  plantilla *dentro de `pybind11/cast.h`* que no mencionan la versión del compilador.
- Cluster: **A100-SXM4-40GB**, 8 GPUs en `nodeGPU01`, partición `AI` (`--account=puc
  --qos=external`). ⚠️ **Son 40 GB, no los "H100 NVL 95GB" que decía esta nota** (corregido
  2026-08-07 con `nvidia-smi` en el nodo). Importa: full attention en ind v1 usa **11.8 GB**,
  pero en **WN18RR v1 llega a 33.7 GB de 40** (84 %) ⇒ subir `hidden_dim`, activar `--use_rpb`
  o pasar a splits más grandes puede dar OOM. Con 2 GPUs (DDP) baja a 16.9 GB.
- **QOS `external`: máximo 4 jobs simultáneos** por usuario (y 8 GPUs, 64 CPUs). Un barrido de
  6 corridas entra en dos oleadas, no en una.
- ⚠️ **Pedir GPUs con `--gpus=N`, NO con `--gres=gpu:...`** — el cluster avisa que va a rechazar
  las segundas. Tampoco existe `/usr/bin/time` en `nodeGPU01` (cronometrar con `date +%s`), y
  **`/tmp` NO se comparte con el nodo de cómputo**: los scripts que se pasan a `srun` deben vivir
  en `$HOME`.
- **Tiempos de los BASELINES ORIGINALES en transductivo** (2026-08-19, A100, 1 GPU, sus configs,
  con kernel `rspmm`; 20 épocas = lo que usan los cuatro papers):

  | brazo | batch | h/época | eval | **TOTAL / semilla (20 ép)** |
  |---|---:|---:|---:|---:|
  | A\*Net FB15k-237 | 64 | 0.49 | 0.49 h | **~10.3 h** |
  | NBFNet FB15k-237 | 64 | 0.90 | 0.50 h | **~18.4 h** |
  | A\*Net WN18RR | 64 | 0.07 | 0.05 h | **~1.6 h** |
  | NBFNet WN18RR | 32 | 0.43 | 0.14 h | **~8.6 h** |

  ⇒ **los 4 brazos × 3 semillas = ~117 GPU-h ≈ 1.5 días de wall** con los 4 slots de QOS.
  Sin el kernel (NBFNet-PyG con `NBFNET_PYG_NO_RSPMM=1`) FB15k-237 cuesta **3× más (2.67 h/época)
  y ni siquiera entra al batch 64** (OOM; tope 16), porque la ruta no fusionada usa O(B·E·d) en
  vez de O(B·V·d); el eval también es 4.4× más lento (11 min vs 2.5 min por validación).
  ⇒ **para transductivo, usar el env `astarnet` con kernel.**
- Tiempos medidos (2026-08-07, full attention, 1 GPU): FB15k-237 v1 **88 s/época**
  (50 ép ≈ 1 h 14); WN18RR v1 **313 s/época** (50 ép ≈ 4 h 21). DDP a 2 GPUs escala 2.18× en
  FB237 y 1.92× en WN18RR. Con GPUs libres conviene **una semilla por GPU en paralelo** antes
  que DDP: el wall total pasa a ser el de una sola corrida.
- **No se habilita `torch.use_deterministic_algorithms(True)`** en `train.py`: el modelo usa
  `index_add_` (no determinista en CUDA). **Esto NO es inocuo** (medido 2026-08-05): la misma
  semilla corrida dos veces da test_mrr **0.202 y 0.317** en el **sparse**. NBFNet (σ=0.004) y
  el full attention (σ=0.002) en cambio son estables. ⇒ **toda corrida del sparse requiere
  n≥3 semillas**; en el full n=3 basta y sobra, pero un número suelto nunca es una medición.

## Comandos típicos

```bash
source env.sh && PY=$(which python)

# Run principal: RFAT en FB15k-237 inductivo v1 (recipe alineado a NBFNet config).
$PY train.py --data_path ./data/inductive/fb15k-237_v1 \
  --num_layer 6 --hidden_dim 32 --num_heads 8 \
  --batch_size 16 --test_batch_size 16 --max_epochs 20 \
  --learning_rate 5e-3 --weight_decay 1e-4 --drop 0.1 --seed 42 \
  --checkpoint_save_path ./experiments/gt_fb15k237_v1

# Smoke test: --num_layer 2 --batch_size 8 --max_epochs 1
```

**Protocolo de medición (obligatorio desde 2026-08-05)**: `--max_epochs 50` para modelos de
atención (a 20 quedan subentrenados: el sparse gana +0.093 solo con 50) y **n≥3 semillas**
(n≥6 para contrastes finos), reportando media ± sd. Comparar brazos con distinto número de
épocas invierte conclusiones — ya pasó una vez. Sbatch de referencia: `sbatch_edrop_final.sh`.

`--edge_drop P` (DropEdge estructural, 2026-08-05): elimina al azar una fracción P de aristas
en cada forward de **train** (eval usa el grafo completo). Regulariza contra el sobreajuste a
la topología del train graph. **P=0.2 es parte de la best config del sparse** (+0.057 ± 0.013)
pero **es NULO en el full attention** (−0.007 ± 0.017, t=−0.40, 2026-08-07) ⇒ no aplicarlo
por defecto: sirve solo donde las aristas definen el soporte de la atención.

`is_inductive(data_path)` detecta inductivo por el sufijo `_vN` del nombre del dataset.
El split de test usa el grafo **disjunto** `<dataset>_ind`; val usa el train graph.

---

## Baselines para el paper WWW (transductivo) — **repos ORIGINALES, ya operativos**

Desde el 2026-08-19 los repos originales corren (env `astarnet`). Configs listos en
`astarnet_cfg/` (4 líneas de diff contra los suyos: solo `output_dir` y `path`).

```bash
source env_astarnet.sh
python run_astarnet.py -c astarnet_cfg/fb15k237_astarnet_trans.yaml --gpus "[0]" --seed 1024
```

- **NBFNet** — correrlo con **`astarnet_cfg/*_nbfnet_trans.yaml`** (que sale de
  `AStarNet/config/transductive/`), NO con NBFNet-PyG: así comparte codebase, pipeline de datos y
  kernel con A\*Net ⇒ el contraste entre ambos aísla la poda A\*.
- **A\*Net** (Zhu et al., NeurIPS 2023) — baseline **obligatorio** para WWW: es el método
  path-based escalable. Transductivo: MRR **0.411** en FB15k-237 y **0.549** en WN18RR.
  Reproducimos su eficiencia: 1.83× sobre NBFNet en FB237 (ellos 2.1×) y 5.8× en WN18RR (6.8×).
- ⚠️ **Las recetas de FB15k-237 y WN18RR NO son iguales** (`dependent`, `adversarial_temperature`
  y `remove_one_hop` difieren). Copiar la config de un dataset al otro invalida la comparación.
- **A\*Net es además el camino a >1M nodos**: en ogbl-wikikg2 (2.5 M entidades) NBFNet da OOM
  incluso con batch 1 y A\*Net propaga el 0.2 % de nodos/aristas. Nuestro GT disperso tiene la
  misma estructura de costo que NBFNet (`O(B·E·d)` por query) ⇒ **hereda el mismo muro**.

## Baseline de comparación: NBFNet (implementado en este harness)

`src/model.py::NBFNet` — reimplementación fiel (message passing DistMult dependiente de
query + PNA + short-cut + layer-norm), MISMA data y MISMO eval full-filtered que el RFAT.
Correr con `--model nbfnet --aggregate pna`. Alineado a `NBFNet/config/inductive/fb15k237.yaml`.

### Resultado head-to-head VIGENTE (FB15k-237 ind v1, full-filtered, 50 ép, n≥3, media ± sd)

| test (50 ép) | **NBFNet** (n=6) | **Full attention** (n=3) | **Sparse + edrop 0.2** (n=6) |
|--------------|-----------------:|-------------------------:|-----------------------------:|
| **MRR**      | **0.458 ± 0.004** | 0.3764 ± 0.0021         | 0.397 ± 0.019 |
| Hits@1       | **0.371**        | 0.315                    | — |
| Hits@3       | **0.520**        | 0.410                    | — |
| Hits@10      | **0.605**        | 0.489                    | — |
| MR           | **117**          | 193                      | 176 |

Contrastes (Welch, todo a 50 ép): **NBFNet vs full +0.0816 ± 0.0020 (t=39.9)** ·
NBFNet vs sparse +0.061 ± 0.008 (t=7.49) · **sparse vs full +0.0206 ± 0.0079 (t=2.62)**.

**Hallazgo (2026-06-21, confirmado con barras de error el 2026-08-07):** el Graph Transformer
full attention —el techo de expresividad de cualquier atención sparse— pierde **decisivamente**
contra NBFNet. Confirma empíricamente `transformer_vs_nbfnet.tex`: la atención reimplementa peor
el message passing y no lo supera. Nuestro NBFNet cae en el rango de literatura (~0.42–0.46),
implementación validada.

**Orden vigente: NBFNet (0.458) > sparse+edrop (0.397) > full (0.376)** — INVERTIDO respecto al
histórico (que ponía el full sobre el sparse). El full no se movió al re-medirlo; el sparse subió
0.059. Nótese que **la brecha del full (0.082) es MAYOR que la del sparse (0.061)**: el techo de
expresividad rinde peor que su propia restricción.

<details><summary>Tabla histórica (n=1, 20 ép, seed 42, dim 32) — punto de partida, superada</summary>

| test | RFAT (full) | NBFNet (pna) |
|------|------------:|-------------:|
| MRR | 0.318 | **0.459** |
| Hits@1 | 0.271 | **0.371** |
| Hits@3 | 0.341 | **0.520** |
| Hits@10 | 0.405 | **0.605** |
| MR | 273 | **117** |

</details>

### Sweep RFAT (20 ep, confirma que el gap es estructural)

Probadas 3 configs para descartar subentrenamiento (dim 64, sin dropout):

| test    | base(d32,dp.1,lr5e-3) | A(d64,L6,lr5e-3) | C(d64,L3,lr5e-3) | **B(d64,L6,lr1e-3)** | NBFNet |
|---------|----------------------:|-----------------:|-----------------:|---------------------:|-------:|
| MRR     | 0.318                 | 0.353            | 0.363            | **0.375**            | **0.459** |
| Hits@10 | 0.405                 | 0.429            | 0.441            | **0.471**            | **0.605** |

El tuning subió el RFAT 0.318 → **0.375** (mejor: dim 64, drop 0.0, lr **1e-3**), pero sigue
bajo NBFNet. ~~El gap es estructural, no un artefacto de entrenamiento.~~

**CORREGIDO (2026-08-05):** ese sweep varió solo dim/L/lr — nunca épocas ni regularización
estructural, que es justo lo que ataca el modo de fallo diagnosticado. Medido en el sparse:
**+0.093 por entrenar 50 épocas en vez de 20** y **+0.057 por `--edge_drop 0.2`**. La brecha
contra NBFNet **existe y es significativa** (sparse: +0.061 ± 0.008, t=7.49, n=6, todo a 50
ep) pero mide **la mitad** de lo que decía la bitácora (0.121).

**RE-MEDIDO (2026-08-07, n=3, 50 ép): el full atención NO estaba subentrenado.** 0.375 (n=1,
20 ép) → **0.3764 ± 0.0021**: Δ **+0.001**. Ninguna de las dos correcciones del 2026-08-05
aplica al full — **son del sparse**:
- **épocas**: sparse +0.093 de 20→50 ép; **full +0.001** ⇒ el 0.375 histórico era correcto.
- **`--edge_drop 0.2`**: sparse +0.057 ± 0.013; **full −0.007 ± 0.017 (t=−0.40) ⇒ NULO.** No es
  un regularizador genérico de atención: sirve donde las aristas **definen el soporte** de la
  atención (sparse), no donde la atención es all-pairs y las aristas solo entran por el bias
  relacional y la corrección de valor (full). Coherente con quién se sobreajusta: gap val→test
  del sparse 0.108 vs 0.056 del full.

Config RFAT vigente: `--num_layer 6 --hidden_dim 64 --drop 0.0 --learning_rate 1e-3
--max_epochs 50`, **sin** edge dropout. Script: `sbatch_full_v1.sh`.

### Sparse Graph Transformer (atención por adyacencia) — `--model sparse`

`src/model.py::SparseGraphTransformer`: cada nodo atiende SOLO a sus vecinos del grafo
(segment-softmax sobre aristas entrantes + self-loop), no a los N. Es "NBFNet con
agregación aprendida por atención". Mismo sweep A/B/C (dim 64, drop 0.0).

| test (best config)  | NBFNet | RFAT denso (full) | **Sparse adyacencia** |
|---------------------|-------:|------------------:|----------------------:|
| valid_mrr           | 0.492  | 0.429             | 0.446                 |
| **test_mrr**        | **0.459** | 0.375          | **0.338**             |
| Hits@10             | 0.605  | 0.471             | 0.451                 |

**Hallazgo (2026-06-21):** el sparse es el **PEOR** de los tres en test, aunque es
estructuralmente el más cercano a NBFNet. Clave: el sparse tiene **valid_mrr MÁS alto que
el denso (0.446 > 0.429) pero test_mrr MÁS bajo (0.338 < 0.375)** ⇒ ajusta mejor el train
graph y **transfiere peor al grafo inductivo disjunto**. Eco del fracaso de los expander
en inductivo (proyecto Exphormer-Max): la agregación blanda aprendida por atención
sobre-ajusta la estructura del train y generaliza peor que la agregación FIJA de NBFNet.

**Orden en test inductivo: NBFNet (0.459) > full attention (0.375) > sparse adyacencia
(0.338).** Ni la atención full ni la sparse superan a NBFNet; la sparse encima generaliza
peor. La dirección "graph transformer (full o sparse) para KGC" queda descartada como
reemplazo de NBFNet; el único margen es inyectar evidencia fuera del horizonte de
propagación, no recombinar/reponderar lo que el message passing ya entrega.

**ACTUALIZACIÓN 2026-08-05 (n=6, 50 ep, `--edge_drop 0.2`):** el sparse re-medido con el
protocolo nuevo da **0.397 ± 0.019** (vs 0.338 n=1). NBFNet **0.458 ± 0.004**. La brecha
**+0.061 ± 0.008 (t=7.49)** sigue siendo decisiva ⇒ la conclusión no cambia, pero:
- el sparse ya **no** es peor que el full ~~(pendiente de confirmar)~~ ⇒ **CONFIRMADO
  2026-08-07**: con el full re-medido (0.3764 ± 0.0021), **sparse+edrop > full**,
  +0.0206 ± 0.0079 (t=2.62). El orden histórico era artefacto — pero **del sparse**
  subentrenado y sin regularizar, no del full, que no se movió.
- ~~el ruido es de la familia de atención~~ **CORREGIDO 2026-08-07: el ruido es del SPARSE,
  no de la familia de atención.** El full attention también es agregación aprendida y mide
  **σ = 0.0021** sin regularizar — comparable a NBFNet (0.004) y **13× más estable que el
  sparse sin regularizar (0.027)**, a igual protocolo y datos. Formulación correcta: *la
  atención por ADYACENCIA transfiere de forma inestable; la atención densa no.* Sigue siendo
  una firma de la causa identificada por eliminación, pero apunta al sparse en particular.
  (Nota: `--edge_drop 0.2` le infla la varianza al full 14×, 0.0021 → 0.0301, sin mover la
  media — eso sí es solo la estocasticidad del train.)

---

## Convenciones

- Reportar siempre **valid_mrr** y **test_mrr** (+ hits@1/3/10, mr). Model selection por
  `valid_mrr` (val = train graph); test sobre el grafo inductivo disjunto.
- **n≥3 semillas + media ± sd, y ≥50 épocas** en modelos de atención. Reportar contrastes con
  error estándar, no deltas puntuales. Todas las entradas de SESSION_NOTES anteriores al
  2026-08-05 son n=1 a 20 épocas. **σ depende del modelo, no de la familia** (medido): sparse
  0.019–0.054 ⇒ ahí una diferencia de 0.01–0.05 con n=1 es ruido; full 0.002 y NBFNet 0.004 ⇒
  ahí n=3 resuelve efectos chicos. Aplicar la advertencia **fila por fila**.
- **Una sola arquitectura** debería servir transductivo e inductivo; solo cambian
  hiperparámetros (requisito metodológico del manuscrito).
- Si train↑ y test↓ con magnitudes grandes → overfit estructural (el mal del proyecto
  previo). Parar y analizar.

## Lista negra (refutado / NO repetir)

1. **NO construir sobre KnowFormer** (V-RMPNN/QK-RMPNN/RSPMM). Atención decorativa, meses perdidos.
2. **NO expander graphs** en inductivo (metían ruido — proyecto Exphormer-Max).
3. **NO embeddings de entidad ni PE de nodo** → rompen inductividad (caso (a) del análisis).
4. **NO atención ciega a la estructura** (sin aristas/relaciones) → caso (a), inviable.
5. **NO BCE-k negative sampling** como loss principal → usar CE de grafo completo.
6. **NO leer `h` acumulada en K/V de un stream separado** (causa del overfit previo).
7. **NO positional/structural encoding por nodo (RWSE ni LapPE) en la atención** (FB15k-237 v1+v2,
   2026-06-28 / 2026-07-08): ninguno da mejora robusta ni supera a NBFNet.
   - **RWSE (PE LOCAL, diag(P^k))**: full neutral-negativo (v1 −0.009, v2 −0.014); sparse INVIERTE
     signo entre splits (v1 −0.046, v2 +0.022 → no robusto); sparse_nbfv plano (ruido). Estructura
     DENTRO del horizonte que el MP ya captura → redundante.
   - **LapPE (PE GLOBAL, autovectores del Laplaciano)**: en v1 parecía ayudar al full (+0.035,
     0.375→0.410) pero **NO se replica en v2 (+0.005, dentro de ruido) → artefacto de split, no
     mejora estructural**. Sparse −0.003/−0.005; sparse_nbfv +0.013/−0.003 (inconsistente). Ninguno
     supera a NBFNet (v1 0.410 vs 0.459; v2 0.496 vs 0.526). Ver SESSION_NOTES 2026-07-08.
   => Ni PE local ni global mueven la aguja de forma confiable. La señal útil debe venir de FUERA del
   horizonte pero **como evidencia composicional/relacional** (caminos, no coordenadas de nodo), no
   de re-codificar la estructura del grafo como feature por nodo.
   ⚠️ **Salvedad de medición (2026-08-05)**: los Δ de RWSE/LapPE son de 0.003–0.046 con n=1 a 20
   épocas, **dentro del ruido** (σ≈0.02–0.05). La conclusión "ningún PE supera a NBFNet" se
   sostiene por el margen grande contra NBFNet, pero los signos y las comparaciones entre PEs
   **no están medidos**. No citar esos deltas como evidencia mecánica sin re-medir.

8. **NO expander graphs tampoco en TRANSDUCTIVO** (FB15k-237, 2026-08-05): `sparse_exp` deg3
   termina en test 0.3973 vs 0.3965 del sparse simple a config y batch idénticos (38399
   steps/época en ambos) ⇒ Δ +0.0008, las 6 métricas coinciden una por una. Cuesta ~1.67×
   por época (aristas aleatorias = accesos no coalescidos). Una arista expander es un
   **no-hecho**: en KGC toda arista es evidencia, y las aleatorias fabrican caminos sin
   soporte relacional además de homogeneizar la distancia al head, que es lo que hace
   funcionar al labeling trick. Combinado con la lista negra #2 (inductivo): línea cerrada.
