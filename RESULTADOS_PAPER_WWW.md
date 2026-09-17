# RESULTADOS_PAPER_WWW.md — resultados transductivos para el paper (ACM WWW)

Protocolo: loss **BCE**, MRR/Hits **full-filtered**, **A100-40GB**. Baselines con los repos
**originales** (env `astarnet`), n=3 semillas (1024/1025/1026). **Modelos propios en n=1** ⚠️.
Plan, justificación y análisis de fallo: `OBJETIVOS_Y_PLAN_WWW.md`. Bitácora: `SESSION_NOTES.md`.

> 🔄 **ACTUALIZADO 2026-09-15.** El modelo del paper dejó de ser el GT podado a secas y pasa a ser
> **GT podado + BLOQUE GLOBAL** (`--global_tokens 32 --global_from 2 --global_pool topk --hop_pe`,
> `src/model.py::GlobalInducedBlock`): M=32 tokens inductores por query que hacen pool sobre los K
> nodos expandidos y se difunden a los S activos, con salida inicializada en CERO (al arrancar es
> bit a bit el modelo anterior; `diag_global_equiv.py`). Todas las tablas de abajo traen las dos
> filas, con y sin el bloque.
>
> 🎯 **La comparación de referencia son los números PUBLICADOS**, no nuestras reproducciones
> (decisión del usuario, 2026-09-10). Nuestras 6 reproducciones caen 0.0003-0.0099 **por debajo**
> de lo publicado, así que comparar contra ellas nos favorece indebidamente.
>
> 📌 **Estado en una línea**: YAGO3-10 **supera a los tres publicados**; FB15k-237 supera a A\*Net
> publicado y queda −0.003 bajo NBFNet; WN18RR no se mueve; ogbl-wikikg2 en curso. **Todo n=1.**

---

## 0. Tabla resumen contra los números PUBLICADOS

| dataset | **nuestro (GT+global)** | *NBFNet pub.* | *A\*Net pub.* | *ULTRA pub.* | Δ vs el mejor publicado |
|---|---:|---:|---:|---:|---:|
| FB15k-237 | **0.4122** | *0.415* | *0.411* | — | **−0.0028** |
| WN18RR | 0.5321 | *0.551* | *0.549* | — | −0.0189 |
| **YAGO3-10** | **0.5780** | *0.563* | *0.556* | *0.557* | **+0.0150** ✅ |
| ogbl-wikikg2 | en curso (job 93414) | *OOM* | *0.6767* | — | — |

⚠️ Los cuatro son **n=1**. El bloqueador para reportar es cerrar n=3, no mejorar el número.

---

## 1. FB15k-237 transductivo

### 1.1 Tabla del paper (modelo actual)

| modelo | **MRR** | **Hits@1** | **Hits@3** | **Hits@10** | MR | épocas | semillas |
|---|---:|---:|---:|---:|---:|---:|---:|
| *NBFNet (publicado)* | *0.415* | *0.321* | — | *0.599* | *114* | 20 | — |
| *A\*Net (publicado)* | *0.411* | — | — | *0.586* | — | 20 | — |
| **GT + poda + BLOQUE GLOBAL, 30 ép** | **0.4122** | **0.3214** | **0.4518** | 0.5898 | **181.2** | 30 | 1 ⚠️ |
| GT + poda + bloque global, 20 ép | 0.4111 | 0.3199 | 0.4511 | **0.5900** | 182.4 | 20 | 1 ⚠️ |
| GT + poda (sin bloque global) | 0.4018 ± 0.0013 | 0.3105 | 0.4400 | 0.5815 | 260.1 | 20 | **3** |
| NBFNet — reproducido por nosotros | 0.4147 ± 0.0010 | 0.3210 | 0.4547 | 0.5982 | 115.9 | 20 | 3 |
| A\*Net — reproducido por nosotros | 0.4084 ± 0.0014 | 0.3192 | 0.4483 | 0.5834 | 497.8 | 20 | 3 |

- **El bloque global vale +0.0104 sobre nuestro propio baseline** (pareado, t=12.0, 15 732 queries
  ganadas contra 14 586) y **mejora el MR de 260 a 181**.
- Contra **publicados**: **+0.0012 sobre A\*Net**, **−0.0028 bajo NBFNet**. Hits@10 queda −0.009
  bajo NBFNet, que es su métrica más fuerte por propagar sobre el grafo completo.
- **Las épocas ya no son palanca acá**: 20 → 30 ép compra +0.0011 y el valid hace plató en la 20.

### 1.2 Brazos históricos (trazabilidad; modelo anterior)

| modelo | **MRR** | **Hits@1** | **Hits@10** | MR | min/época | mem GPU |
|---|---:|---:|---:|---:|---:|---:|
| GT sin poda | 0.4183 | 0.3219 | 0.6079 | 109.1 | 206.1 | 24.4 GB |
| GT + poda, sin normalizar | 0.4022 | 0.3111 | 0.5821 | 308.1 | 67.6 | **3.7 GB** |
| GT + poda, softmax | 0.3927 | 0.3024 | 0.5692 | 351.1 | 73.9 | 3.7 GB |

- **La poda cuesta −0.016** de MRR y casi 3× el MR, pero da **3.05× de velocidad** y **6.6× menos
  memoria**. ⚠️ El brazo sin poda no es reportable: el modelo tiene que ser uno solo en los cuatro
  datasets, y a 2.5 M entidades sin poda no corre.

## 2. WN18RR transductivo

| modelo | **MRR** | **Hits@1** | **Hits@3** | **Hits@10** | MR | épocas | semillas |
|---|---:|---:|---:|---:|---:|---:|---:|
| *NBFNet (publicado)* | *0.551* | *0.497* | — | *0.666* | *636* | 20 | — |
| *A\*Net (publicado)* | *0.549* | — | — | — | — | 20 | — |
| NBFNet — reproducido | 0.5466 ± 0.0015 | 0.4906 | 0.5706 | 0.6619 | **652.9** | 20 | 3 |
| A\*Net — reproducido | 0.5431 ± 0.0013 | 0.4890 | 0.5669 | 0.6523 | 5 853.7 | 20 | 3 |
| **GT + poda + bloque global** | 0.5321 | 0.4818 | 0.5534 | 0.6313 | 6 516.6 | 20 | 1 ⚠️ |
| GT + poda (sin bloque global) | 0.5266 ± 0.0049 | 0.4770 | 0.5490 | 0.6297 | 6 574.9 | 20 | 3 |

🔴 **WN18RR es el dataset donde NO funcionamos.** El bloque global da +0.0054 sobre la media del
baseline propio, pero **la semilla 42 del baseline vale exactamente 0.5321**, o sea el efecto está
**dentro del ruido entre semillas** (σ = 0.0049). Contra publicados quedamos **−0.019**.

Tres hechos medidos que explican por qué, y que van al paper (detalle en
`OBJETIVOS_Y_PLAN_WWW.md` §3.4.5):
1. **El modo de fallo acá es DISTANCIA, no cardinalidad.** El 87 % de los fallos comunes está a 4+
   saltos del head y el **43 % fuera del horizonte de 6 capas o desconectado**. El bloque global no
   crea caminos, así que no puede tocar ese 43 %.
2. **La poda deja respuestas sin score.** El 12 % de los fallos comunes recibe el escalar de
   fallback (rank = N) en el GT y en A\*Net, y **0 % en NBFNet**, que propaga por todo el grafo.
   Ése es exactamente nuestro MR de 6 516 contra los 653 de NBFNet.
3. **Hay un techo duro de 0.966**: 212 queries (3.4 % del test) tienen la respuesta **sin ninguna
   arista** en el grafo de train, y los tres modelos dan MRR 0.000 ahí.

⚠️ Canceladas las semillas 43/44 del brazo con bloque global (decisión del usuario, 2026-09-12)
para liberar GPU; queda `last.ckpt` de la s43 en la época 12.

## 3. YAGO3-10 transductivo (123 182 entidades, 8.5× FB15k-237)

### 3.1 🎯 Tabla del paper: el GT + bloque global SUPERA A LOS TRES PUBLICADOS

| modelo | **MRR** | **Hits@1** | **Hits@3** | **Hits@10** | MR | pasadas | × A\*Net | semillas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *NBFNet (publicado)* | *0.563* | *0.480* | *0.612* | *0.708* | — | *0.40* | 1.0× | — |
| *ULTRA (publicado)* | *0.557* | — | — | *0.71* | — | — | — | — |
| *A\*Net (publicado)* | *0.556* | *0.470* | *0.611* | *0.707* | — | *0.40* | 1.0× | — |
| **GT+global, ép. 17** | **0.5780** | **0.5024** | **0.6209** | **0.7159** | **800.5** | 0.68 | 1.7× | 1 ⚠️ |
| **GT+global, ép. 11** | **0.5693** | 0.4902 | 0.6141 | 0.7138 | 864.1 | 0.44 | **1.1×** | 1 ⚠️ |
| **GT+global, ép. 10** | 0.5632 | 0.4849 | 0.6083 | 0.7042 | 991.2 | **0.40** | **1.0×** | 1 ⚠️ |
| GT + poda (sin bloque global) | 0.5427 ± 0.0084 | 0.4709 | 0.5959 | 0.6946 | 1 274 | 0.40 | 1.0× | 3 |
| A\*Net — reproducido | 0.5461 ± 0.0015 | 0.4660 | 0.5953 | 0.6924 | 2 489 | 0.40 | 1.0× | 3 |

**Las dos lecturas, las dos defendibles:**
- **A presupuesto de entrenamiento EXACTAMENTE igualado** (0.40 pasadas, que es lo que usan ellos):
  **0.5632, empata a NBFNet publicado** (+0.0002) y supera a A\*Net (+0.0072) y ULTRA (+0.0062).
- **Al dejarlo converger** (0.68 pasadas, 1.7× su presupuesto, valid en plató desde la época 14):
  **0.5780, +0.0150 sobre NBFNet**, **+0.0220 sobre A\*Net**, y **Hits@10 0.7159 por encima de los
  tres**. Hay que declarar el 1.7×.

Contra el baseline propio: **+0.0265 pareado (t=13.5)**, ganando en **los seis estratos de
cardinalidad** (máximo +0.0515 en |T|=1-3 y +0.0483 en 101+). El **MR baja de 2 489 a 800**, o sea
3.1× mejor que A\*Net, que es la métrica donde la poda suele castigar.

Coste: **7.3 GB** con `--grad_ckpt` (el baseline sin checkpointing usaba 29 GB) y 2.7 it/s.

### 3.2 Brazos históricos (trazabilidad)

| modelo | **MRR** | **Hits@1** | **Hits@3** | **Hits@10** | MR | min/época | h/semilla | h total | épocas | dim | capas | GPUs | semillas | mem GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GT + sin padding** | **0.5551 ± 0.0023** | **0.4751** | **0.6035** | **0.6977** | **1 006** | 88.6 | 14.8 | 29.6 | 0.40 pas. | 32 | 6 | 1 | 2 | 31.6 GB |
| **GT + estado disperso, cap 1024** | **0.5478** | **0.4626** | **0.5990** | **0.6990** | **1 184** | 92.6 | 15.4 | 15.4 | 0.40 pas. | 32 | 6 | 1 | 1 | 35.5 GB |
| GT + estado disperso, cap 512 | 0.5317 | 0.4461 | 0.5849 | 0.6858 | 1 399 | 90.1 | 15.0 | 15.0 | 0.40 pas. | 32 | 6 | 1 | 1 | 33.3 GB |
| GT + estado disperso, cap 256 | 0.5147 ± 0.0054 | 0.4356 | 0.5650 | 0.6514 | 1 780 | 88.2 | 14.7 | 29.4 | 0.40 pas. | 32 | 6 | 1 | 2 | 32.2 GB |
| GT + estado disperso, cap 128 | 0.4833 ± 0.0068 | 0.4170 | 0.5197 | 0.6007 | 2 166 | 87.9 | 14.6 | 29.2 | 0.40 pas. | 32 | 6 | 1 | 2 | 31.6 GB |
| GT + estado disperso, cap 64 | 0.4498 ± 0.0025 | 0.3840 | 0.4889 | 0.5635 | 3 015 | 67.5 | 11.3 | 22.6 | 0.40 pas. | 32 | 6 | 1 | 2 | 23.1 GB |
| GT, cupo de estado al 29 % ✗ | 0.3788 | 0.3338 | 0.4073 | 0.4565 | 31 238 | 32.5 | 5.4 | 5.4 | 0.40 pas. | 32 | 6 | 1 | 1 | 13.2 GB |

Las «épocas» de YAGO se cuentan en **pasadas sobre el train set**: 0.40 pasadas = 10 épocas-PL de
5 395 pasos × batch 8 sobre 1 079 040 triples. Es el protocolo de A\*Net (Tabla 9: `#epoch 0.4`).

✗ Brazo **SUPERADO**, se deja por trazabilidad. `node_slots` topaba el estado en el **29 % de N**
y, al desbordar, desalojaba quedándose con los **ids globales más chicos** — criterio arbitrario,
sin relación con la relevancia. Cobertura de la respuesta **65.4 %**: el 34.6 % restante caía en
un bloque de empate del fallback y nunca recibía score. Ver §5 **H9**.

🟢 **−0.015 contra NBFNet y −0.008 contra A\*Net**, con **H@10 casi empatado** (0.699 vs 0.707
y 0.707). Era −0.184 el 2026-08-26. Los **+0.169** son **arreglos de BÚSQUEDA, ninguno de
modelo**: quitar el desalojo por id global (+0.071) y aflojar `edge_cap` de 64 a 1024 (+0.098).

**`edge_cap` es la palanca dominante y NO ha saturado** (§5 H10). El último tramo medido es el
**más** eficiente por arista, no el menos. El tope es un **artefacto de nuestra implementación**:
A\*Net no tiene tope por nodo. Implementada la ruta sin padding (`--edge_cap 0`), pendiente de
su test de equivalencia.

**La cobertura está agotada** (98.3 %) y **las épocas no son palanca** (plató desde la época 2 a
20 épocas, §5 H11). ⚠️ **cap 1024 es n=1**; σ del brazo = 0.0068 (medida en cap 128).


## 3b. ogbl-wikikg2 transductivo (2 500 604 entidades, 16.1 M triples) — EN CURSO

⚠️ **MÉTRICA DISTINTA A LAS DEMÁS SECCIONES.** OGB rankea contra **500 negativos fijos** por
dirección (1 000 por tripleta), no contra las 2.5 M entidades — rankear completo costaría días.
Los negativos vienen **pre-filtrados por OGB** (verificado: 0 de 40 M son tripletas del KG), y
el mismo protocolo aplica a valid y a test. **No comparar estos números con los de YAGO.**

⚠️ El split es **TEMPORAL** (Wikidata mayo/agosto/noviembre 2015 → train/valid/test), no
aleatorio: una caída valid→test es cambio de distribución, no sobreajuste.

| modelo | MRR | H@1 | H@3 | H@10 | MR |
|---|---:|---:|---:|---:|---:|
| *A\*Net (publicado)* | *0.6767 (test) / 0.6851 (valid)* | — | — | — | — |
| *NBFNet (publicado)* | ***OOM*** *(ni con batch 1)* | — | — | — | — |
| **GT+global — EN CURSO** (job 93414, 2 GPUs) | valid 0.537 / 0.319 / 0.508 / 0.524 / 0.537 | — | — | — | — |
| *GT, 1 época, checkpoint no válido (histórico)* ⚠️ | *0.4964* | *0.4444* | *0.5208* | *0.5891* | *96.5* |

### 3b.1 Protocolo de evaluación de OGB — AUDITADO el 2026-09-14

| qué | estado |
|---|---|
| fórmula de rank `0.5·(optimista+pesimista)+1` | ✅ **idéntica** a `ogb...Evaluator._eval_mrr` (rama torch), comparada línea por línea |
| 500 negativos por dirección, positivo en la columna 0 | ✅ verificado en los tensores (valid 429 456 × 500, test 598 543 × 500) |
| 2 ítems por tripleta, cola y cabeza, **intercalados** | ✅ (el bug del prefijo 100 % cola se arregló el 2026-09-01) |
| negativos **ya filtrados por OGB** ⇒ no volver a filtrar | ✅ **verificado**: de 40 M negativos, **0** son aristas de train, y el gold nunca está entre sus propios negativos |
| grafo de evaluación = solo aristas de train | ✅ |
| **A\*Net publica con `fast_test: 5000`** | 🔴 **hallazgo nuevo**: su config submuestrea valid y test a **5 000 ítems aleatorios** (seed 1024) de 1 197 086 ⇒ su 0.6767 tiene **se ≈ 0.006** |
| nuestro submuestreo | era un **PREFIJO** y está sesgado (el prefijo de 2 000 tiene grado mediano de cola 1 939 contra 1 230 del total). **Corregido**: `--eval_subsample N --eval_seed`, permutación aleatoria como la de ellos. Con 64 000 ítems, **se ≈ 0.0018** |

⚠️ **Consecuencia para la comparación**: su número publicado no es exacto. Una ventaja nuestra de
menos de ~0.012 sobre 0.6767 no sería distinguible de su propio ruido de muestreo.

⚠️ **Los empates dominan bajo poda.** Los candidatos que la búsqueda no visita reciben todos el
mismo escalar de fallback; con el desempate de OGB, un gold no visitado cae al rango ~250 entre los
negativos empatados. En la práctica **MRR ≈ P(gold visitado) × MRR|visitado**, y por eso ellos
suben `test_node_ratio` de 0.002 a 0.01 al evaluar (ya portado: `--test_top_nodes 25006`).

### 3b.2 Costo medido (barrido de batch, job 93400)

| batch (train) | memoria | s/paso | tripletas/s |
|---:|---:|---:|---:|
| 8 | 21.2 GB | 2.41 | 3.32 |
| 16 | 36.1 GB ⚠️ borde | 4.32 | 3.70 |
| 24 y 32 | **OOM** | — | — |

| test_batch | ítems/s |
|---:|---:|
| 2 | 2.67 |
| 4 | 1.83 |
| 8 | 2.63 |
| 16 | **OOM** |

⇒ **Subir el batch NO compra tiempo** (+11 % de 8 a 16, y 16 está al borde de OOM). Mismo resultado
que ya se había medido en NBFNet-PyG: el throughput por tripleta es casi invariante al batch.

⇒ **Igualar el presupuesto de datos de A\*Net (2 560 000 tripletas) cuesta 196 h en 1 GPU.** Por eso
la corrida va a **2 GPUs × batch 8** (decisión del usuario), 40 épocas de 4 000 pasos, ~109 h de
train, en **dos jobs encadenados** (el tope de la partición son 84 h).

⚠️ **Resultado PRELIMINAR, no reportable**: (a) 1 sola época de las 40 del presupuesto;
(b) el checkpoint se eligió con una validación que medía **solo predicción de cola** (bug del
ordenamiento, corregido); (c) falta `test_node_ratio` (ver §6.5). se = 0.0018 sobre 64 000 de
1 197 086 items.

**Asimetría cola/cabeza — es del DATASET, no nuestra**: MRR **0.7573** en cola contra **0.2354**
en cabeza. Las relaciones son casi funcionales hacia adelante (media **0.41** destinos por
consulta) y masivamente muchos-a-uno hacia atrás (media **8 605**, máx 1.5 M); el nodo desde el
que propaga la búsqueda tiene grado mediano **8** en cola y **1 268** en cabeza. A\*Net enfrenta
lo mismo. Nuestra debilidad está **localizada en cabeza**, que es el mismo modo de fallo de
H4/H5 (escasez de evidencia en fuentes hub) en su forma extrema.

**Costo**: 1 GPU, batch 16, 28.5 GB, 2.47 s/it, ~2.8 h por época de 4 000 pasos.
`--grad_ckpt` **no es opcional**: sin él hay OOM incluso a batch 8.

## 3c. Ablación del bloque global (a épocas y config IGUALADAS)

Cada fila cambia **una sola variable** contra su baseline, que es el mismo modelo con el flag
apagado. Todos los flags nuevos tienen **init cero** ⇒ el forward inicial es bit a bit el del
baseline (`diag_global_equiv.py`), así que la ablación es limpia.

| mecanismo | dataset | Δ MRR | pareado | veredicto |
|---|---|---:|---|---|
| **bloque global** (`--global_tokens 32 --hop_pe`) | FB15k-237 | **+0.0104** | t=12.0 · 15 732 ganadas / 14 586 | ✅ **adoptado** |
| **bloque global** | YAGO3-10 | **+0.0265** | t=13.5 · 3 729 / 2 410 | ✅ **adoptado** |
| **bloque global** | WN18RR | +0.0054 | t=3.2 · 1 575 / 1 686 | ⚠️ **dentro del ruido de semilla** |
| frontera consciente de la relación (`--rel_frontier 1.0`) | FB15k-237 | +0.0013 | t=1.7 · 14 104 / **16 001** | ❌ **nulo** (y el MR empeora, 271 vs 260) |
| fallback informado por PPR (`--fallback ppr`) | WN18RR | −0.0006 | t=−0.4 · 1 254 / **2 033** | ❌ **nulo** (y los sin-score suben de 4.4 % a 14.7 %) |
| PPR como condición de borde (`--indicator ppr`) | WN18RR | **+0.038** | 3 ép cortas, misma semilla | 🟡 prometedor, sin n≥3 |
| PPR como condición de borde | FB15k-237 | **−0.155** | 3 ép cortas (0.142 vs 0.297) | ❌ **colapsa** (efecto de dataset, no de `--dependent`: sin él es peor aún, 0.097) |

**De los cuatro mecanismos probados, tres son nulos y uno funciona.** Los tres nulos tocan la
búsqueda o el score; el que funciona es el único que agrega un canal de cómputo nuevo.

---

## 3d. Métricas de CONJUNTO para queries multi-respuesta

**Por qué.** El protocolo estándar evalúa cada tripleta por separado, pero queries como
(EE.UU., `nationality⁻¹`, ?) tienen **301 respuestas en test** y el KG está incompleto: medir el
conjunto recuperado es más fiel. Medido: |T| por query tiene **mediana 1** en los tres datasets
(78.6 % / 90.9 % / 94.8 % con |T|=1), pero en FB15k-237 el **23 % de los ítems** vive en queries
con |T| ≥ 10.

⚠️ Con |T|=1, **Recall@K ≡ Hits@K** y **Precision@K ≤ 1/K por construcción** ⇒ Precision@K promediada
sobre todo el test mide el tope, no el modelo. Por eso la métrica titular es **R-Precision**
(K = |T| de cada query, donde precisión = recall) y Precision@K se reporta solo sobre |T| ≥ K.

Se calculan **sin reevaluar nada**, desde los volcados de `--dump_ranks`: el conjunto de
distractores filtrados es idéntico para todas las respuestas de una misma (h,r), así que el rank
filtrado es monótono en el score y la posición real de la m-ésima respuesta es su rank + m.
Script: `analyze_set_metrics.py`.

**FB15k-237, todas las queries (22 850)**

| modelo | R-Prec | MAP | R@1 | R@3 | R@10 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| GT + bloque global (30 ép) | 0.4411 | 0.5273 | 0.3964 | 0.5512 | 0.6978 | 0.8699 |
| GT + poda (sin bloque) | 0.4328 | 0.5189 | 0.3886 | 0.5438 | 0.6895 | 0.8674 |
| NBFNet | 0.4318 | 0.5224 | 0.3889 | 0.5491 | 0.6981 | **0.8817** |
| A\*Net | **0.4454** | **0.5297** | **0.4012** | **0.5542** | 0.6954 | 0.8629 |

🔴 **INVERSIÓN que la métrica estándar esconde**: ganamos en MRR (0.4122 contra 0.410 de A\*Net)
pero **perdemos en R-Precision** (0.4411 contra 0.4454). MRR solo mira la primera respuesta
correcta; R-Precision mira el conjunto. **A\*Net recupera mejor el conjunto, nosotros rankeamos
mejor la primera.** Va al paper: es un eje donde los métodos difieren y nadie lo reporta.

**FB15k-237, subconjunto |T| ≥ 10** (339 queries = 1.5 % de las queries pero **23.2 % de los ítems**)

| modelo | R-Prec | MAP | P@10 | R@100 |
|---|---:|---:|---:|---:|
| **GT + bloque global (30 ép)** | **0.3866** | 0.3890 | **0.4637** | 0.6811 |
| GT + poda (sin bloque) | 0.3701 | 0.3716 | 0.4537 | 0.6723 |
| NBFNet | 0.3799 | **0.3901** | 0.4587 | **0.6927** |
| A\*Net | 0.3708 | 0.3709 | 0.4525 | 0.6713 |

⇒ **Donde el bloque global paga es exactamente en las queries multi-respuesta**: ahí somos los
mejores en R-Precision y en Precision@10, por encima de NBFNet y A\*Net.

**YAGO3-10 y WN18RR, todas las queries**

| YAGO (8 560 queries) | R-Prec | MAP | | WN18RR (5 716) | R-Prec | MAP |
|---|---:|---:|---|---|---:|---:|
| **GT + global (20 ép)** | **0.5428** | **0.6147** | | GT + global | 0.5021 | 0.5480 |
| GT + global (11 ép) | 0.5313 | 0.6058 | | GT + poda | 0.5021 | 0.5481 |
| GT + poda | 0.5113 | 0.5874 | | NBFNet | **0.5097** | **0.5627** |
| A\*Net | 0.5082 | 0.5841 | | A\*Net | 0.5101 | 0.5610 |

**NBFNet gana Recall@100 en los tres datasets** (0.8817 / — / 0.7919 contra 0.8699 / — / 0.7587).
Es el costo de la poda expresado en una métrica que lo captura bien: propaga sobre el grafo
completo, así que su cola es mejor. Mismo fenómeno que el MR.

**Métrica ponderada por |T|** (la propuesta del usuario, `nRec@K` = número medio de respuestas
recuperadas en el top-K; máximo alcanzable en FB15k-237 a K=100 es 1.734):

| modelo | nRec@1 | nRec@3 | nRec@10 | nRec@100 |
|---|---:|---:|---:|---:|
| GT + bloque global | 0.4618 | 0.7057 | **0.9827** | 1.4154 |
| NBFNet | 0.4515 | 0.6976 | 0.9760 | **1.4344** |
| A\*Net | **0.4655** | **0.7063** | 0.9748 | 1.3952 |

Conviene reportar al lado su versión normalizada (`microR@K`, en [0,1], pondera por |T|): a K=10 da
0.6450 (nuestro), 0.6406 (NBFNet), 0.6399 (A\*Net).

---

## 3e. Análisis de fallo: los tres métodos fallan en LAS MISMAS queries

Medido alineando por nombre de entidad los ranks de NBFNet, A\*Net y los nuestros (100 % de las
queries alineadas en los tres datasets). Detalle en `OBJETIVOS_Y_PLAN_WWW.md` §3.3. Scripts:
`analyze_overlap_v2.py`, `analyze_seeds_fb237.py`, `analyze_wn18rr_distance.py`.

| Jaccard medio de los conjuntos de fallo (rank > 10) | FB15k-237 | YAGO3-10 | WN18RR |
|---|---:|---:|---:|
| GT (sin bloque) vs baselines externos | 0.792 | 0.816 | 0.829 |
| **GT + bloque global vs baselines externos** | **0.789** | **0.813** | **0.827** |
| NBFNet vs A\*Net (comparten codebase y kernel) | 0.799 | — | 0.846 |
| **entre SEMILLAS del mismo modelo** | **0.854** | **0.863** | **0.883** |
| fallan los tres a la vez | 33.8 % | 24.3 % | 30.6 % |

**Dos lecturas, las dos para el paper:**
1. **Cambiar de arquitectura mueve el conjunto de fallos apenas más que cambiar de semilla.** Los
   tres métodos computan esencialmente la misma función, que es lo que predice la cota de `rawl2`
   para los C-MPNN (Huang et al., NeurIPS 2023).
2. 🔴 **El bloque global NO cambia eso** (0.789 contra 0.792): mejora el ranking **sin** volver
   resolubles queries que antes no lo eran. ⇒ **No se puede reclamar haber salido de la clase
   C-MPNN apoyándose en MRR.** El reclamo defendible es empírico: un canal de atención global
   mejora el ranking de forma consistente, con ganancia máxima en YAGO y en las queries
   multi-respuesta.

**Qué caracteriza a los fallos compartidos** (FB15k-237, re-medido con el modelo nuevo):

| | fallan los tres | aciertan los tres |
|---|---:|---:|
| n | 13 816 (33.8 %) | 21 147 (51.7 %) |
| **respuestas de (h,r) ya en train, mediana** | **37** | **4** |
| grado mediano de la respuesta / de la fuente | 70 / 182 | 252 / 86 |
| queries con cardinalidad ≥ 31 | **51.9 %** | 19.5 % |

MRR por cardinalidad, **para los tres modelos por igual**: 0.68 con 0 respuestas conocidas → 0.18
con más de 100. El 21.5 % del test son queries de más de 100 respuestas, donde Hits@10 es 0.33.

⇒ **Ese 34 % es incompletitud y ambigüedad del KG, no falta de expresividad**, y explica por qué
FB15k-237 lleva años en 0.41-0.45 para todo método estructural. Oráculo de los tres modelos: MRR
0.4913 y H@10 0.6625, o sea aun combinándolos queda un tercio del test irresoluble.

En **WN18RR** el modo de fallo es otro: **distancia**. El 87 % de los fallos comunes está a 4+
saltos y el 43 % fuera del horizonte o desconectado; 212 queries (3.4 %) tienen la respuesta
aislada, con MRR 0.000 para los tres ⇒ **techo duro de 0.966** para toda la familia.

---

## 4. Nota sobre los tiempos

**Todos los brazos usan dim 32, 6 capas y 1 GPU** ⇒ las comparaciones son a capacidad igualada.
`h total` = `h/semilla` × `semillas`. La memoria es el pico de `torch.cuda.max_memory_allocated`.

⚠️ Los tiempos de NBFNet y A\*Net son **con** el kernel CUDA fusionado `rspmm`. **Nuestra atención
no puede usarlo**: se desactiva cuando los pesos de arista requieren gradiente
(`AStarNet/reasoning/layer.py:132`), que es la definición de atención. A\*Net lo esquiva pesando
**nodos**, no aristas. Vale ~3× y es **estructural de la familia**, no de nuestra implementación
— corroborado por *On Efficient Scaling of GNNs via IO-Aware Layers Implementations* (Yandex,
**ICML 2026 Spotlight**), cuyos kernels dedicados a atención en grafos siguen dando **speedup <1×
en el backward** por las atómicas.

⚠️ Los baselines corren a **batch 64** (32 en NBFNet WN18RR) y los nuestros a **batch 8**; ver §6.

⚠️ **De dónde sale nuestra memoria en YAGO: 31.6 GB contra 13 GB de A\*Net y 26.1 de NBFNet.**
A\*Net calcula sobre **nodos**: `rspmm` fusiona mensaje y agregación, así que los mensajes por
arista **nunca se materializan** y su estado es `O(B·K·d)` ≈ 15 MB/capa. Nuestra atención
necesita `q_e, k_e, v_e, g_e, logit, alpha, msg` **por arista** (`src/model.py:1153-1186`), todos
retenidos para el backward: un `(B,L,d)` son 0.44 GB ⇒ ~2.6 GB/capa ⇒ **~15.7 GB en 6 capas**,
del orden de lo medido. El factor es **L/K = 37×**: cada arista pesa lo mismo que un nodo y hay
37 veces más aristas propagadas que nodos activos. **Es el precio estructural de la atención en
esta familia** — el mismo motivo por el que no podemos usar `rspmm` — no una ineficiencia
optimizable sin cambiar el mecanismo. NBFNet gasta 26.1 GB por lo contrario: propaga por las
2.16 M aristas y mantiene estado sobre **todos** los nodos; A\*Net baja a 13 GB quedándose con
el 10 %. A esto se suma nuestro β al 211 % (§6.0), que duplica L y **sí** es corregible.

---

## 5. Hallazgos

> ⚠️ **H1-H14 son de la etapa ANTERIOR (agosto y principios de septiembre), con el GT podado SIN
> bloque global.** Siguen siendo el registro válido de lo medido y varios se citan en el paper,
> pero los que hablan del resultado principal quedaron superados por §1, §3 y §3c-§3e. En
> particular: **H12 está retractado**, **H13 (la interacción de escalado con β) no replica fuera de
> YAGO** y dejó de ser la contribución, y el encuadre de "salir de la clase C-MPNN" está refutado
> por §3e. Lo que sí se sostiene y se cita: **H1** (reproducción), **H2** (poda sin pérdida en
> inferencia), **H3** (la poda destruye la cola del ranking), **H4-H5** (modos de fallo, ahora
> ampliados en §3e), **H8** (el muro de escala es el estado, no las aristas), **H9-H11** (cobertura,
> `edge_cap`, épocas) y **H14** (expander nulo a presupuesto igualado).

**H1 — Los cuatro brazos de baseline reproducen dentro del 1.1 %.** NBFNet FB15k-237 **−0.0003**,
A\*Net FB15k-237 −0.0026, NBFNet WN18RR −0.0044, A\*Net WN18RR −0.0059. Las cuatro desviaciones
son negativas y de magnitud parecida ⇒ firma de pipeline correcto, no de errores independientes.

**H2 — La poda de A\*Net es SIN PÉRDIDA en inferencia.** Mismo checkpoint evaluado con
`test_node_ratio` 0.1 / 0.5 / 1.0: MRR **0.4100 / 0.4092 / 0.4102** (Δ **+0.0002**). Comparación
pareada sobre 40 932 queries: 26 % mejora / 48 % empata / 26 % empeora — simétrico, o sea ruido.
Es más fuerte que lo que ellos reportan. ⚠️ Mide poda **en inferencia**, no en entrenamiento.

**H3 — La poda destruye la cola del ranking, en los dos datasets.** MR de A\*Net vs NBFNet:
**4.3×** peor en FB15k-237 (497.8 vs 115.9) y **9.0×** en WN18RR (5853.7 vs 652.9), con MRR/H@10
casi idénticos. Los nodos que la búsqueda nunca visita quedan sin score. **Es la métrica donde un
modelo que propaga por todo el grafo debería ganar** — y nuestro GT sin poda da **MR 109.1**.

**H4 — El modo de fallo dominante NO es el alcance ni el sesgo de popularidad: es escasez de
evidencia.**
- *Distancia*: de las 17 056 queries que fallan fuera de top-10, solo el **0.3 %** está fuera del
  horizonte de 6 saltos; el 99.2 % de los pares está a 2-3 saltos.
- *Sesgo*: el distractor top-1 supera en grado a la respuesta el **50.0 %** de las veces, ratio
  mediano **1.01** ⇒ **no hay sesgo de popularidad**.
- *Grado de la respuesta*: MRR **0.2316** en el decil más bajo (grado 0-14) contra **0.9172** en
  el más alto (≥655). **4× de diferencia**, sin sesgo que lo explique.

**H5 — Los dos ejes del fallo son separables y se componen.** MRR por (grado fuente × grado
respuesta), peor celda *fuente hub × respuesta periférica* = **0.154**:

| fuente \ resp | 0-27 | 28-99 | 100-249 | 250+ |
|---|---:|---:|---:|---:|
| 0-27 | 0.266 | 0.373 | 0.595 | **0.849** |
| 250+ | **0.154** | 0.216 | 0.613 | 0.644 |

El eje de la **respuesta** es más fuerte (3.2× de rango) pero es el limitado por información; el
de la **fuente** es más débil (1.7×) y es el atacable — concentra el **27 % del déficit total**.
Techo de neutralizar la inundación por completo: **+0.027 ⇒ ~0.435**, por encima de NBFNet.

**H6 — La ventaja de la atención está LOCALIZADA en el régimen de inundación.** Estratificado por
grado de la fuente, GT sin poda vs A\*Net: empata en cinco estratos (±0.003) y gana **+0.0306** en
el de fuentes de grado ≥250. **El 91 % de su ventaja agregada viene de ese único estrato.**
⚠️ Es la única predicción sobre atención escrita **antes** de medir que se cumplió en este
proyecto; las cinco anteriores fallaron. n=1 ⇒ no establecido.

**H7 — Bajo poda, la agregación SIN NORMALIZAR le gana al softmax: +0.0095 en test.** El
segment-softmax normaliza sobre las aristas **seleccionadas**, así que un nodo con 100 entrantes
de las que se eligen 5 queda "como si tuviera 5 vecinos" — la normalización **esconde** que se
descartó evidencia. NBFNet y A\*Net **suman**, así que lo notan.
✅ **Corroboración independiente**, del apéndice E de A\*Net: *"PNA does not generalize well when
degrees are dynamically determined by the priority function. Therefore, we precompute the degree
on the full graph."* Mismo fenómeno, otro grupo, otra arquitectura.

**H8 — Para escalar, el muro es el ESTADO, no las aristas.** De FB15k-237 a YAGO hay 4.1× de
aristas pero **8.5× de nodos**, y el modelo se puso **10.9× más lento** ⇒ escala con N. Con estado
denso `(B,N,d)`, wikikg2 pediría **15.6 GB solo de estado** retenido por 6 capas. Con estado
disperso `(B,S,d)`, S=30 000 ⇒ **30 MB**. Medido en YAGO: **3.5× menos memoria, batch 32 deja de
dar OOM**. A\*Net resuelve lo mismo con `edge_mask(compact=True)` + `VirtualTensor`.

**H9 — En YAGO el gap era COBERTURA, no calidad; y un condicional sesgado no se extrapola.**
Con el cupo de estado al 29 % de N, la distribución de ranks era **bimodal** (p60 = 674,
p70 = 87 753): el 65.4 % de las respuestas recibía score y el 34.6 % caía en un bloque de
empate. `0.654 × 0.5788 = 0.3785` reproducía el MRR global exacto. La causa: al desbordar el
cupo, `merged[:, :S]` se quedaba con los **ids globales más chicos**. A\*Net no desaloja nunca
— su `graph.score` es un VirtualTensor sobre los N nodos que sólo **acumula** claves
(`AStarNet/reasoning/model.py:322`); su `node_ratio` limita **quién se expande**, no **quién
recibe score**. Nosotros conflacionábamos los dos ejes en un solo parámetro.
⚠️ **Advertencia metodológica que costó una predicción fallida**: de `MRR|alcanzada = 0.5788`
proyectamos 0.55–0.58 con cobertura total. Salió **0.4881**. Ese condicional estaba sesgado
hacia lo fácil (grado mediano de la respuesta **87**); las queries que faltaban eran las
periféricas (grado **7-9**) y al incorporarlas el condicional **bajó** a 0.4989. Un condicional
sobre un subconjunto seleccionado por dificultad es un **techo optimista**, no una proyección.

**H10 — `edge_cap` es la palanca dominante, y el tope es un ARTEFACTO de implementación, no
un método.** Truncar a `cap` las salientes por nodo afecta a poquísimos nodos, pero son hubs y
concentran mucha arista (grado saliente de YAGO: media 17.5, mediana 10, **máx 61 044**).

| cap | nodos truncados | aristas retenidas | MRR | MRR por punto de arista |
|---:|---:|---:|---:|---:|
| 64 | 2.76 % | 74.2 % | 0.4498 | — ⚠️ |
| 128 | 1.48 % | 81.5 % | 0.4833 | 0.0045 ⚠️ |
| 256 | 0.63 % | 88.6 % | 0.5147 | 0.0044 |
| 512 | 0.19 % | 93.6 % | 0.5317 | 0.0034 |
| 1024 | 0.02 % | **96.2 %** | **0.5478** | **0.0061** |

⚠️ **El escalón 64→128 está CONFUNDIDO**: esa corrida cambió a la vez el `cap` (64→128) y el
`edge_budget` (228 122 → 456 244, o sea β de ~106 % a ~211 %), porque el script decía «L en
proporción» — pero `cap` y `L` son ejes independientes. Su +0.034 **no es atribuible al `cap`
solo**. Los tramos 128→256→512→1024 sí son limpios (β fijo en 456 244) y sostienen la
conclusión por sí mismos: +0.031, +0.017, +0.016.

**+0.098 en total y sin saturar**: el último tramo es el **más** eficiente por arista — las
aristas de los hubs extremos valen más que las medianas. **No actúa por cobertura**: entre cap
512 y 1024 la cobertura no se mueve (98.3 %) y `MRR|alcanzada` sube 0.5410 → 0.5571 ⇒ el efecto
es sobre la CALIDAD de la evidencia agregada.

**A\*Net no tiene este tope**: `neighbors()` (`AStarNet/reasoning/data.py:276`) devuelve todas
las salientes y el único recorte es global vía `variadic_topks`. Nuestro tope existía sólo para
armar el pool como matriz rectangular `(B, K·cap)`, y tomaba las **primeras `cap` aristas de la
fila CSR** — el orden del archivo del dataset, no de relevancia. Portada su ruta sin padding
(Alg. 2, dos `sort` estables) como `--edge_cap 0`; el pool pasa de 12.6 M casillas/query a
~1.06 M aristas reales (**11.8× menos**).

**H11 — En YAGO las épocas no son palanca, y una trayectoria suelta no es una tendencia.** A 20
épocas (0.80 pasadas) el brazo hace plató desde la época 2: **0.471 → 0.477 en ocho épocas**. La
lectura previa de «no convergió, el valid sube monótono 0.419 → 0.487» venía de **una sola
semilla**; la semilla 43 del mismo brazo eligió su mejor checkpoint en la **época 1**. σ entre
semillas = **0.0068**, no 0.0025 como sugería el brazo cap 64.

**H12 — ⚠️ RETRACTADO. La ventaja estratificada era el PRESUPUESTO, no la arquitectura.**
Medido sobre el brazo de β=211 %, el GT ganaba +0.0277 (t = 3.92) en el estrato de fuentes con
grado ≥1000, con el 44.4 % de la ventaja en el 12.3 % de las queries — y encajaba con la
predicción mecánica (A\*Net pesa **nodos**, `layer_input = sigmoid(score) * hidden`,
`AStarNet/reasoning/model.py:319`; nosotros pesamos **aristas**).

**Rehecho sobre β=100 %** (presupuesto igualado, que es el que usa el paper) **el efecto
desaparece y se invierte**:

| grado fuente | n | Δ MRR | **t** | gana/pierde |
|---|---:|---:|---:|---|
| 0-27 | 2 984 | −0.0029 | −0.70 | 806/907 |
| 28-99 | 3 262 | −0.0132 | −3.78 | 514/605 |
| 100-249 | 824 | −0.0560 | −5.44 | 230/336 |
| 250-999 | 1 671 | −0.0391 | −4.75 | 511/674 |
| ≥1000 | 1 223 | +0.0047 | **0.59** | **396/610** |
| **GLOBAL** | 9 964 | **−0.0158** | **−6.24** | |

El estrato de hubs deja de ser significativo (t = 0.59) y los conteos pareados dan **396/610**:
perdemos más veces de las que ganamos. **A presupuesto igualado el GT pierde contra A\*Net por
−0.0158 (t = −6.24).** El resultado principal en transductivo está ABIERTO.

**En curso — la pregunta reformulada**: no *"¿podemos usar β=211 %?"* sino **cómo escala cada
método con el presupuesto de evidencia**. Nuestra curva no satura (0.5301 → 0.5551 al duplicar
L); falta la de ellos (A\*Net con `degree_ratio: 2`, n=3). Hipótesis: su agregación suma/PNA es
**fija** y satura; la atención **selecciona** dentro del conjunto ampliado y no. Si se confirma,
la contribución es *"la atención convierte evidencia adicional en precisión"*, que es una
propiedad de la arquitectura y no una ventaja de cómputo.

**H13 — Los dos métodos escalan en direcciones OPUESTAS con el presupuesto de evidencia. Ésta es
la contribución.** Barrido de β (`degree_ratio`, las aristas propagadas por capa) en YAGO3-10:

| | β=100 % (L≈216 k) | β=200 % (L≈432-456 k) | **pendiente** |
|---|---:|---:|---:|
| **A\*Net** (medido) | **0.5461 ± 0.0015** (n=3) | 0.5390 ± **0.0109** (n=3) | −0.0071 (t = −1.11, **n.s.**) |
| **GT (nuestro)** | 0.5301 (n=1) | **0.5551 ± 0.0023** (n=2) | **+0.0250** |

⚠️ **A\*Net NO degrada de forma significativa** (t = −1.11 con n=3; con n=2 daba t = −2.77 y se
reportó como degradación — corregido). **Lo que sí muestra es INESTABILIDAD: su σ es 7.1× mayor
a β=200 %** (0.0109 vs 0.0015), rango 0.0217 entre semillas. Coherente con la dilución: con
agregación fija, *qué* aristas entren al conjunto ampliado pasa a importar, y eso depende de la
semilla. **Hay cruce**: a β=100 % ganan ellos (−0.0160, t = −6.24 pareado); a β=200 % ganamos
nosotros (**+0.0161, t = +2.47**). **Interacción = +0.0321**, mayor que cualquier ventaja individual.

**Mecanismo, predicho antes de medir**: su agregación suma/PNA es **fija**, así que cada arista
adicional aporta evidencia **y ruido** en la misma proporción y termina diluyendo; la atención
**selecciona** dentro del conjunto ampliado. ⇒ El reclamo no es *"les ganamos"* —a β=100 % es
falso— sino **"la atención convierte evidencia adicional en precisión; la agregación fija no"**,
que es una propiedad de la arquitectura y no una ventaja de cómputo.

Explica por qué ellos fijan β=100 % en los **cuatro** datasets mientras varían α 50× (10 % →
0.2 %): para su arquitectura, subir β **perjudica**.

⚠️ **Nuestra pendiente sigue anclada en n=1** (β=100 %). En curso: GT β=100 % semillas 43/44 y
GT a β=**200 % exacto** (hoy usamos 211 %, un 5.7 % más de presupuesto a nuestro favor ≈ +0.0007).
**No reportar hasta tener las cuatro celdas en n=3.**


**H14 — Los expander graphs son NULOS a presupuesto igualado (tercera refutación, la más
controlada).** YAGO3-10, mismo modelo, misma semilla, mismo `edge_budget` = 326 700, único
cambio el expander (grado 4, relación reservada `R_exp`):

| | MRR | H@1 | H@10 | MR |
|---|---:|---:|---:|---:|
| con expander | 0.5388 | 0.4533 | 0.6964 | 1 238 |
| **sin expander** | **0.5400** | 0.4572 | 0.6893 | 2 077 |
| | **Δ −0.0012** | | | |

El +0.0087 que aparecía contra el brazo de L=215 804 era **íntegramente el presupuesto extra**:
agregar 985 456 aristas expander sube el E del grafo un 43 %, y β=100 % ⇒ L = K·E/N ⇒ el brazo
con expander propagaba **1.51× más aristas**. Predicción escrita antes de correr (~0.542): 0.5400.

Coherente con H4: los expander resuelven **alcanzabilidad**, y el cuello medido no es ése
(**0.3 %** de los fallos cae fuera del horizonte de 6 saltos; cobertura **98.3 %** en YAGO) sino
**escasez de evidencia** — y una arista aleatoria no es evidencia. Tres refutaciones
independientes: inductivo, FB15k-237 transductivo (Δ +0.0008) y YAGO (Δ −0.0012).


---

## 6. Desviaciones de protocolo que hay que DECLARAR

> 🔄 **Lista vigente al 2026-09-15** (la numerada de abajo es de la etapa anterior y se conserva):
> 1. **Épocas**: en YAGO el mejor checkpoint es el de 0.68 pasadas contra sus 0.40, o sea **1.7×**
>    su presupuesto. Se reportan **las dos filas** (§3.1) y la de presupuesto igualado ya empata a
>    NBFNet. En FB15k-237 corrimos 30 épocas contra sus 20, pero el valid hace plató en la 20 y
>    compra +0.0011 ⇒ irrelevante.
> 2. **Batch 8** contra 64 (FB15k-237), 40 (YAGO) y 128 (wikikg2). **No es elección**: medido, a
>    batch 16 en wikikg2 estamos al borde de OOM y a 24 revienta, y el throughput por tripleta es
>    casi invariante al batch (§3b.2).
> 3. **wikikg2**: batch efectivo 16 contra su 128; 8 192 negativos contra su 1 048 576; **sin**
>    `--indicator ppr` ni `--edge_dropout_p` (los dos medidos como dañinos o sospechosos); test
>    sobre **64 000 ítems aleatorios** contra su `fast_test: 5000`.
> 4. **Selección de checkpoint por `valid_mrr`** sobre trayectorias de 20-30 épocas. Es lo que
>    hacen ellos, pero con n=3 hay que reportar la media de los mejores por semilla.
> 5. **Nosotros tuneamos y ellos no.** Batch, lr, épocas, β y los flags nuevos se eligen por
>    `valid_mrr` de nuestro lado; los baselines corren con su config publicada. Va en la tabla, no
>    en una nota al pie.
> 6. **`--grad_ckpt`** en YAGO y wikikg2. No cambia el resultado (verificado, gradientes a 2.66e-6)
>    pero sí la memoria reportada: en YAGO 7.3 GB con checkpointing contra 29 GB sin él.

0. ~~**β ≈ 211 %** contra su 100 %~~ — **NO es una desviación de protocolo**: en su Tabla 9 α y β
   son **hiperparámetros que ellos ajustan por dataset** (α va de 10 % en FB15k-237/WN18RR/YAGO
   a **0.2 % en wikikg2**). Elegimos β=211 % por `valid_mrr` (0.559 vs 0.537), que es el criterio
   correcto. **Se declara el valor y se reporta la ablación**: β=100 % da test **0.5301** contra
   **0.5551**, o sea −0.025, y además entrena peor (mejor época 1 de 10, valid oscilando). Sí hay
   que declarar que β=211 % duplica L y con ello la memoria (31.6 vs 22.8 GB).
1. **Batch 8** en nuestros modelos, contra **64** (FB15k-237) y **40** (YAGO) de ellos. No es
   elección: nuestro costo es `O(B·E·d)` sin kernel fusionado y a batch 64 la memoria se dispara.
2. ~~**`edge_cap`** (tope de salientes por nodo) contra su **β = 100 %**~~ — **EN VÍAS DE
   ELIMINARSE**. No era una desviación menor: valía **+0.098 de MRR** en YAGO (H10), más que el
   gap contra NBFNet. Portada la ruta sin padding de A\*Net (`--edge_cap 0`); una vez validado
   su test de equivalencia, esta desviación **desaparece** en vez de mitigarse.
3. **Semilla 1025 de NBFNet FB15k-237 recuperada de un checkpoint de la época 8** de 20 (su job
   murió por un disco lleno del cluster). Defendible porque NBFNet satura en la época 4, pero no
   es idéntico al protocolo de las otras dos.
4. **Nuestros modelos son n=1-2**; los baselines n=3.
5. **CUATRO opciones que A\*Net usa SOLO en wikikg2** y que no teníamos (implementadas el
   2026-09-02, pendientes de correr): `indicator_func: ppr` (condición de borde con distancia
   en vez de ceros), `num_negative: 1048576` (contra nuestros 32 — el **42 %** de las entidades),
   `edge_dropout: 0.2` y `break_tie: yes`. Más `test_node_ratio: 0.01` (5× el presupuesto de
   búsqueda al evaluar), implementado el 2026-09-01 y **medido: +0.0259 de MRR**.
   ⚠️ En sus configs de FB15k-237/WN18RR/YAGO **ninguna de las cinco aparece**, así que esto
   NO afecta a los resultados de esos datasets.
6. **`lr` 1e-3 en YAGO contra su 5e-3.** Heredado de la config del inductivo y **nunca
   justificado**. Es la desviación más criticable del resultado de YAGO: hay que correr un
   brazo con su lr para mostrar que no elegimos el que nos favorece.
7. **Métricas no comparables entre datasets**: FB15k-237/WN18RR/YAGO usan full-filtered sobre
   todas las entidades; wikikg2 usa el ranking muestreado de OGB (500 negativos por dirección).

## 7. Qué falta (2026-09-15)

### 🚩 Bloqueador único para reportar: **n = 1 en los tres brazos buenos**

Ningún número de las secciones 1, 2 y 3 es reportable hasta cerrar semillas. No es un problema de
mejorar el modelo, es de barras de error.

| brazo | estado | costo estimado |
|---|---|---:|
| **FB15k-237 + bloque global, s43 y s44** | canceladas por el usuario; `last.ckpt` en las épocas 5 y 1 ⇒ **reanudables** | ~40 h |
| **YAGO3-10 + bloque global, s43 y s44** | nunca corridas | ~60 h |
| WN18RR + bloque global, s43 y s44 | canceladas; `last.ckpt` de la s43 en la época 12 | ~22 h |

⚠️ Al cerrar n=3 hay que decidir y declarar **cómo se elige el checkpoint**: hoy se reporta el
mejor por `valid_mrr` de una sola corrida (que es lo que hacen NBFNet y A\*Net), pero con n=3 lo
correcto es la **media de los mejores por semilla**, no el mejor de los tres.

### En curso

- **ogbl-wikikg2** (jobs 93414 + 93415 encadenados, 2 GPUs × batch 8, 40 épocas de 4 000 pasos).
  Protocolo de OGB auditado y correcto (§3b.1). ⚠️ **Las validaciones oscilan fuerte**
  (0.537 / 0.319 / 0.508 / 0.524 / 0.537) y la `train_loss` no baja: la corrida difiere de la
  estable (92755, que llegó a 0.576) en **tres cosas a la vez** — 8 192 negativos contra 32, el
  bloque global, y `break_tie`. Sospechoso principal: los negativos. **Decisión pendiente del
  usuario**: dejarla correr o relanzar con 32 negativos para aislar una variable.

### Huecos metodológicos, sin GPU

- **No hay explicación mecánica establecida de por qué funciona el bloque global.** La predicción
  escrita antes de medir falló en sus dos mitades (§3e y `OBJETIVOS_Y_PLAN_WWW.md` §3.6).
  Lo barato y en CPU: analizar **a qué atienden los tokens inductores**.
- **Falta la ablación del bloque global a épocas igualadas en WN18RR** con n≥3.
- **`--indicator ppr` queda abierto**: ayuda en WN18RR (+0.038 en corto) y colapsa en FB15k-237
  (−0.155). Es el único mecanismo con razón medida para WN18RR (43 % de los fallos fuera del
  horizonte), pero necesita n≥3.

### Cerrado (no volver a intentar)

Frontera consciente de la relación (nulo, §3c) · fallback informado por PPR (nulo y empeora la
cobertura) · expander graphs (tres refutaciones) · PE por nodo dentro de la atención · subir el
batch en wikikg2 para ganar tiempo (medido: +11 % de 8 a 16, y 24 da OOM).
