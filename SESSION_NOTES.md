# SESSION_NOTES.md — bitácora de resultados y análisis

Registro cronológico de sesiones: qué se corrió, qué salió, qué se concluyó. Más reciente
arriba. Objetivos estables en `GOALS.md`; briefing operacional en `CLAUDE.md`.
**Tablas del paper transductivo (solo lo que va a la publicación): `RESULTADOS_PAPER_WWW.md`.**

Para cada entrada: **Resultado** (números), **Análisis** (causa mecánica), **Decisión**.

---

> 🚩 **PRIORIDAD VIGENTE (2026-08-18, defensa de candidatura APROBADA): manda el régimen
> TRANSDUCTIVO, con paper a ACM WWW, deadline la primera semana de octubre de 2026. El
> inductivo se retoma DESPUÉS del envío.** Bloqueador #1: no hay baseline NBFNet transductivo
> propio en ningún dataset. Ver la entrada **2026-08-18** para el inventario y la cola.
> Casi toda esta bitácora es inductiva: sigue siendo el registro válido de lo medido, pero
> **ya no fija la prioridad de cómputo**.

> 🔒 **REGLA DE LOSS (2026-08-21, decisión del usuario): TODOS los experimentos van con
> `--loss bce`, salvo que se pida explícitamente lo contrario.** Incluye **re-correr los
> experimentos de Graph Transformer**, que hoy están todos medidos con CE de grafo completo.
> ⚠️ El **default del flag sigue siendo `ce`** (para no cambiarles el objetivo en silencio a los
> ~40 scripts viejos) ⇒ **la regla se cumple pasando `--loss bce` en cada script; verificarlo
> antes de lanzar.** Consecuencia de lectura: **todo número de esta bitácora anterior al
> 2026-08-19 es con CE** y no es comparable con los baselines externos, que usan BCE-k.

> 🎓 **LEER PRIMERO — EL MARCO DE LA TESIS (registrado 2026-08-08, ver entrada (f)).**
> `GOALS.md` y `CLAUDE.md` describen el **sanity check de la etapa actual** ("¿el full attention
> supera a NBFNet?"), **NO la tesis**. La tesis es: **Graph Transformer DISPERSO (Exphormer,
> expander) + codificación relacional TRANSFERIBLE (ULTRA) para inferencia de enlaces
> ZERO-SHOT en grafos de conocimiento.** El objeto central es el modelo **sparse**; el full
> attention es un **control** que descarta la alternativa densa, no el entregable.
> Consecuencia práctica: **los resultados del sparse son los que importan para la tesis**, y
> las hipótesis H1/H2 del manuscrito están **en tensión con lo medido** sobre el expander.
> Antes de proponer o descartar cualquier línea, leer la entrada **2026-08-08 (f)**.

> ⚙️ **REGLA OPERACIONAL (2026-08-21, entrada (b)).** La capacidad relacional se agrega
> **por QUERY (`--dependent`), no por ARISTA (`--rel_param lowrank`)**. Es el mecanismo de
> NBFNet/A\*Net y **cuesta 1.01–1.08× en tiempo** (medido, job 92055) contra el **2.33×** de
> `lowrank`. Nuestro modo estático ya equivalía a su `dependent: no`. ⚠️ Dos límites medidos:
> en **WN18RR transductivo** el pico sube a **30.0 GB de 40**, y con **`hidden_dim 64` los
> params llegan a 9.2 M (3.9× NBFNet)** ⇒ el argumento *parameter-matched* solo vale en
> transductivo (d=32: 2.07 M contra 3.10 M de NBFNet).

> ⚙️ **REGLA OPERACIONAL (2026-08-09).** Los lookups relacionales por arista van con
> **`table.index_select(axis, rel)`**, NUNCA con `table[:, rel]`: la indexación avanzada tiene
> un backward por atómicos que cuesta **hasta 2×** (sparse_exp 3.33 → 6.83 it/s), y hace que el
> expander parezca caro cuando no lo es (sobrecosto 1.76× → **1.08×**). Ver entrada 2026-08-09
> y `SparseRelationalAttentionLayer._rel_lookup`. **Los tiempos previos al 2026-08-09 no son
> comparables; los resultados sí (cambio bit a bit idéntico).**
>
> ⚙️ **REGLA OPERACIONAL VIGENTE (2026-08-08, entrada (k)).** `--rel_param lowrank` usa la
> versión **optimizada**: la matriz de valor `M[r] = diag(g[r]) + U[r]W[r]ᵀ` se precomputa **por
> relación** (`_rel_matrix`), no por arista. **Todos los experimentos deben usar esta versión**
> — no revertir a la implementación por arista. Es **bit a bit idéntica** (diff 0.000e+00), 1.7×
> más rápida y 1.8× más liviana, y **su costo ya no depende de k** ⇒ usar **k=8** salvo que el
> objetivo sea un ablation de capacidad. ⚠️ Los **tiempos** medidos antes del 2026-08-08 (k) no
> son comparables con los de después; los **resultados** sí.

---

## TABLA RESUMEN — 4 modelos × datasets inductivos (best config, seed 42)

> ⚠️ **ADVERTENCIA DE VALIDEZ (2026-08-05).** Todos los números de esta tabla y de las
> entradas anteriores al 2026-08-05 son de **UNA corrida, semilla 42, 20 épocas**. El
> estudio de semillas del 2026-08-05 midió **σ ≈ 0.054** en el test_mrr del sparse sin
> regularizar, y **la misma semilla corrida dos veces dio 0.202 y 0.317** (no-determinismo
> de `index_add_` en CUDA). Además **20 épocas subentrena a los modelos de atención**: el
> sparse pasa de 0.247 a 0.347 solo con 50 épocas. En consecuencia:
> - Las diferencias de **0.01–0.05 con n=1** que sostienen varios ablations (sigmoid +0.018,
>   degree +0.028, rel +0.020, RWSE −0.009/+0.022, LapPE −0.003/+0.035/−0.005) **NO son
>   distinguibles de ruido**. Las entradas correspondientes interpretan mecánicamente
>   deltas que la medición no resuelve.
> - **La columna NBFNet sí es sólida**: a 50 épocas NBFNet mide 0.458 ± 0.004 (n=6) en
>   ind v1, o sea el 0.459 registrado es correcto. El ruido alto es específico de los
>   modelos de ATENCIÓN (σ 5-7× mayor), lo cual es en sí un hallazgo.
> - Las **filas de atención están subestimadas** por entrenar solo 20 épocas: el sparse gana
>   +0.093 con 50 épocas y +0.057 más con `--edge_drop 0.2`. La comparación cualitativa
>   (ninguna atención supera a NBFNet) **se sostiene** y ahora tiene significancia
>   (t=7.49, n=6); las magnitudes de los gaps **no**.
> Ver la entrada del 2026-08-05 para el detalle y qué habría que re-correr.
>
> ⚠️ **ACOTACIÓN (2026-08-07 b).** El punto anterior vale **fila por fila, no por familia**:
> re-medido el full attention (50 ép, n=3) da **0.3764 ± 0.0021** contra el 0.375 registrado
> ⇒ Δ +0.001, **el full NO estaba subentrenado**. El subentrenamiento a 20 épocas y la alta
> varianza son del **sparse**, no de los modelos de atención en general (full σ=0.002 vs
> sparse σ=0.027, mismo protocolo). Las filas `Full attention` de estas tablas son fiables;
> las de `Sparse attention` son pisos.
>
> ⚠️ **SEGUNDA ACOTACIÓN (2026-08-08).** La acotación anterior también vale fila por fila: el
> sparse de **ind v2** re-medido a 50 ép (n=3) da **0.4556 ± 0.0225** contra el 0.450 de la
> tabla ⇒ Δ **+0.006**, o sea **en v2 el sparse tampoco estaba subentrenado**. El +0.093 de
> 20→50 ép es **específico del sparse en v1**. Tercera vez que una advertencia enunciada "por
> familia" resulta ser de una fila concreta ⇒ **regla operativa: ninguna propiedad medida en un
> brazo se hereda a otro brazo ni a otro split sin re-medirla.**

Cada modelo con su mejor config: NBFNet (dim 32, drop 0.1, lr 5e-3, pna);
full/sparse/sparse_nbfv (dim 64, drop 0.0, lr 1e-3). Todos L6, 20 ep. Métricas full-filtered.
Negrita = mejor por columna.

### FB15k-237 ind v1

| modelo              |    MRR | Hits@1 | Hits@3 | Hits@10 |
|---------------------|-------:|-------:|-------:|--------:|
| **NBFNet**          | **0.459** | **0.371** | **0.520** | **0.605** |
| Full attention      |  0.375 |  0.320 |  0.402 |   0.471 |
| Sparse attention    |  0.338 |  0.276 |  0.359 |   0.451 |
| Sparse_nbfv (V=NBF) |  0.421 |  0.339 |  0.476 |   0.554 |
| Full + RWSE         |  0.366 |  0.310 |  0.407 |   0.459 |
| Sparse + RWSE       |  0.292 |  0.227 |  0.337 |   0.376 |
| Sparse_nbfv + RWSE  |  0.424 |  0.332 |  0.480 |   0.593 |
| Full + LapPE        |  0.410 |  0.351 |  0.441 |   0.524 |
| Sparse + LapPE      |  0.335 |  0.285 |  0.363 |   0.410 |
| Sparse_nbfv + LapPE |  0.434 |  0.346 |  0.500 |   0.571 |
| Full + source_rw    |  0.313 |  0.266 |  0.332 |   0.395 |

Variantes de agregación del sparse (flag `--attn`, misma config; detalle en las entradas 2026-07-22 y
2026-07-24): sigmoid 0.356 · degree 0.366 · anchor 0.278 · **rel 0.358**. Ninguna sale de la banda
0.28–0.37 ni se acerca a NBFNet (0.459).

**Valores con protocolo vigente (50 ép, n≥3, media ± sd) para las filas ya re-medidas de ind v1** —
estos reemplazan a los de la tabla, que son n=1 a 20 ép:

| modelo | 20 ép, n=1 (tabla) | **50 ép, n≥3** | Δ |
|--------|-------------------:|---------------:|--:|
| NBFNet | 0.459 | **0.458 ± 0.004** (n=6) | −0.001 |
| Full attention | 0.375 | **0.3764 ± 0.0021** (n=3) | +0.001 |
| Sparse attention | 0.338 | **0.397 ± 0.019** (n=6, con `--edge_drop 0.2`) | +0.059 |
| Sparse attention | — | 0.340 ± 0.027 (n=6, sin edge dropout) | — |

**Orden vigente: NBFNet (0.458) > sparse+edrop (0.397) > full (0.376)** — invertido respecto al
histórico. `sparse_nbfv` y todas las filas de PE siguen SIN re-medir.

### WN18RR ind v1

| modelo              |    MRR | Hits@1 | Hits@3 | Hits@10 |
|---------------------|-------:|-------:|-------:|--------:|
| **NBFNet**          | **0.740** |  0.689 | **0.774** |   0.822 |
| Full attention      |  0.673 |  0.638 |  0.691 |   0.739 |
| Sparse attention    |  0.738 |  0.686 | **0.774** |   0.819 |
| Sparse_nbfv (V=NBF) | **0.740** | **0.691** |  0.766 | **0.832** |
| Full + RWSE         |  0.670 |  0.638 |  0.684 |   0.734 |
| Sparse + RWSE       |  0.677 |  0.628 |  0.697 |   0.769 |
| Sparse_nbfv + RWSE  |  0.728 |  0.684 |  0.755 |   0.798 |

### FB15k-237 ind v2

| modelo              |    MRR | Hits@1 | Hits@3 | Hits@10 |
|---------------------|-------:|-------:|-------:|--------:|
| NBFNet              |  0.526 |  0.416 | **0.595** | **0.727** |
| Full attention      |  0.491 |  0.389 |  0.544 |   0.686 |
| Sparse attention    |  0.450 |  0.369 |  0.494 |   0.586 |
| **Sparse_nbfv (V=NBF)** | **0.527** | **0.431** |  0.586 |   0.690 |
| Full + RWSE         |  0.477 |  0.382 |  0.535 |   0.652 |
| Sparse + RWSE       |  0.472 |  0.382 |  0.517 |   0.644 |
| Sparse_nbfv + RWSE  |  0.529 |  0.427 |  0.589 |   0.703 |
| Full + LapPE        |  0.496 |  0.397 |  0.556 |   0.681 |
| Sparse + LapPE      |  0.445 |  0.358 |  0.486 |   0.592 |
| Sparse_nbfv + LapPE |  0.524 |  0.424 |  0.585 |   0.687 |
| Full + source_rw    |  0.363 |  0.292 |  0.402 |   0.479 |

**Valores con protocolo vigente (50 ép, n=3, media ± sd) para las filas ya re-medidas de ind v2**
(2026-08-08) — reemplazan a los de la tabla, que son n=1 a 20 ép:

| modelo | 20 ép, n=1 (tabla) | **50 ép, n=3** | Δ |
|--------|-------------------:|---------------:|--:|
| Sparse attention (`--edge_drop 0.0`) | 0.450 | **0.4556 ± 0.0225** | +0.006 |
| Sparse attention (`--edge_drop 0.2`) | — | **0.5157 ± 0.0088** | — |
| sparse_exp deg3 (`--edge_drop 0.0`) | — | 0.4736 ± 0.0120 | — |
| **sparse_exp deg3 (`--edge_drop 0.2`)** | — | **0.5275 ± 0.0032** ⚠️ piso | — |
| NBFNet (nuestro, CE de grafo completo) | 0.526 | **0.5245 ± 0.0042** (n=3) | −0.001 |
| *NBFNet-PyG (autores, BCE-k) — referencia externa* | — | *0.3589 ± 0.0459* (n=3, 20 ép) | — |

✅ **RESUELTO 2026-08-08 (j)**: el GT disperso **IGUALA a NBFNet en v2** — sparse_exp+edrop
0.5275 ± 0.0032 vs NBFNet 0.5245 ± 0.0042, **+0.0030 ± 0.0031 (t=0.97)**, indistinguibles.
El 0.526 histórico de NBFNet era correcto (Δ −0.0015) y tampoco estaba subentrenado.
⚠️ **El 0.359 de los autores NO es el brazo de comparación** — su receta usa BCE con 32
negativos y reporta H@10_50 (ahí da 0.9472 ± 0.0108 contra ~0.941 del paper, o sea reproduce).
Su MRR full-filtered es bajo **por la loss**, no por la implementación. Ver entrada 2026-08-08 (c).
⚠️ El 0.5275 es un **piso**: sus best-valid caen en las épocas 39/45/48 de 50 (subentrenado).
`full`, `sparse_nbfv` y todas las filas de PE siguen SIN re-medir en v2.

**Lectura cruzada**: ninguna variante de atención supera claramente a NBFNet en MRR.
FB15k-237 v1 (composicional): todas pierden, sparse el peor, sparse_nbfv el mejor de atención.
FB15k-237 v2: mismo orden (sparse el peor, full por debajo de NBFNet), pero **sparse_nbfv
empata/roza a NBFNet** (0.527 vs 0.526, mejor Hits@1). WN18RR v1 (local): full el peor,
sparse/sparse_nbfv empatan a NBFNet. El message passing sigue siendo el techo en todos los
regímenes. Detalle por experimento en las entradas de abajo.

### TRANSDUCTIVO (resumen, agregado 2026-08-07)

Los números transductivos estaban dispersos en 4 entradas y sin tabla. Todos n=1, 20 ep,
seed 42, mismas salvedades de validez que arriba. **Ojo con la columna batch**: no todas son
apples-to-apples entre sí.

> ⚠️ **Estas filas son TRANSDUCTIVAS y por eso corren L4 / dim 32** — la config chica se eligió
> el 2026-07-20 por costo (en transductivo dim64/L6 cuesta 2.4× por época y 2.7× en memoria
> para comprar +0.006 de MRR). **En INDUCTIVO todo corre L6 / dim 64** (`sbatch_edrop_final.sh`,
> `sbatch_exp_typing.sh`, `sbatch_full_v1.sh`). No confundir los dos regímenes al comparar.

| dataset | modelo / config | batch global | valid_mrr | **test_mrr** | H@10 | MR |
|---------|-----------------|-------------:|----------:|-------------:|-----:|---:|
| FB15k-237 | sparse softmax L6/d64 | 16 | 0.407 | **0.4028** | 0.595 | 116.4 |
| FB15k-237 | sparse softmax L4/d32 | 8 | 0.402 | 0.3965 | 0.591 | 128.4 |
| FB15k-237 | sparse_exp deg3 L4/d32 | 8 | 0.4012 | 0.3973 | 0.589 | 126.8 |
| FB15k-237 | sparse **degree** L4/d32 | **32** | 0.4023 | 0.3957 | 0.592 | 126.1 |
| FB15k-237 | *sparse + `--remove_one_hop`* | 8 | 0.4335 | ***0.42893*** | *0.606* | *130.0* |
| WN18RR | sparse softmax L4/d32 | 32 | 0.5481 | **0.5536** | 0.644 | 1616 |
| WN18RR | sparse **degree** L4/d32 | 32 | 0.5468 | 0.5533 | 0.642 | 1610 |
| WN18RR | *sparse_exp deg3* | 32 | 0.5531 | ***0.5566*** | *0.647* | *1497* |

**BASELINES EXTERNOS (repos ORIGINALES, env `astarnet`, BCE-k, 20 ép, n=3 semillas
1024/1025/1026)** — agregados 2026-08-21, entrada de esa fecha:

| dataset | brazo | **test MRR** | H@10 | MR | *paper* |
|---------|-------|-------------:|-----:|---:|--------:|
| FB15k-237 | **A\*Net** | **0.4084 ± 0.0014** | 0.5834 | 497.8 | *0.411* |
| FB15k-237 | **NBFNet** | **PENDIENTE** (~18.4 h/semilla) | — | — | *0.415* |
| WN18RR | **NBFNet** | **0.5466 ± 0.0015** | 0.6619 | **652.9** | *0.551* |
| WN18RR | **A\*Net** | **0.5431 ± 0.0013** | 0.6523 | 5853.7 | *0.549* |

⚠️ **Los brazos propios de esta tabla NO son comparables con los baselines**: los nuestros
son n=1, 20 ép, L4/d32 y con **CE de grafo completo**; los baselines usan **BCE-k**, que es
el confounder que motivó el cambio de loss del 2026-08-19. La comparación se habilita recién
al re-correr el harness propio con `--loss bce` y n≥3.
⚠️ **Falta el 4º brazo (NBFNet FB15k-237)** ⇒ el bloqueador #1 sigue abierto en ESE dataset,
que es donde vive el mejor número transductivo del proyecto (0.42893).

**RWSE (filas `+ RWSE`, 2026-06-28)**: no es mejora confiable. Full neutral-negativo en v1/v2
(−0.009, −0.014); sparse INVIERTE signo entre splits (−0.046 v1, +0.022 v2 → no robusto);
sparse_nbfv plano (ruido). Ninguno supera a NBFNet. **WN18RR v1 + RWSE (registrado 2026-07-08):
daña los 3** — full −0.003 (0.673→0.670), sparse −0.061 (0.738→0.677, su firma de overfit otra vez),
sparse_nbfv −0.012 (0.740→0.728). Consistente con FB15k-237: RWSE nunca supera a NBFNet y al sparse
lo perjudica fuerte. RWSE completo en los 3 datasets ⇒ conclusión (lista negra #7) confirmada.

**LapPE (filas `+ LapPE`, 2026-07-08, v1 + v2)**: PE GLOBAL (autovectores del Laplaciano de TODO el
grafo, no local como RWSE). En v1 dio un bump prometedor al full (+0.035, 0.375→0.410), PERO **NO se
replica en v2 (+0.005, 0.491→0.496, dentro de ruido) ⇒ el efecto NO es robusto**, cae al split como
el sparse+RWSE. Δ test por LapPE: Full +0.035(v1)/+0.005(v2); Sparse −0.003/−0.005; Sparse_nbfv
+0.013(v1)/−0.003(v2). Ningún signo consistente y positivo en ambos splits; **ninguno supera a
NBFNet** (v1 full+LapPE 0.410 vs 0.459; v2 full+LapPE 0.496 vs 0.526). Conclusión: LapPE se une a RWSE
como PE no confiable para atención. El bump de v1 fue un artefacto de split, no una mejora estructural.

**source_rw (filas `Full + source_rw`, 2026-07-08)**: labeling trick CONDICIONADO A LA FUENTE
(no node-only como RWSE/LapPE): feature del nodo v = proj de las probabilidades de landing de un
random walk de k=1..8 pasos que ARRANCA en el head de la query (fila head de P^k, P=D^-1 A). En
principio es la señal query-relativa que GOALS deja abierta. Resultado: **el más dañino de los tres
encodings**. A diferencia de RWSE/LapPE (que degradan poco y sólo en test), source_rw **derrumba
valid Y test en ambos splits**: full v1 test 0.375→0.313 (−0.062), v2 test 0.491→0.363 (−0.128);
valid v1 0.429→0.234, v2 0.461→0.279. No es firma de overfit (valid↑/test↓) sino degradación neta
in-distribution incluida. Se une a RWSE/LapPE como fuente de señal NO no-redundante para la atención.

---

## 2026-09-07 — 🔴 **Corrección teórica: la atención entre vecinos NO rompe la cota de rawl2.** El PPR sí es la palanca. Y contra números PUBLICADOS no ganamos en ninguno

### La comparación que importa: contra números PUBLICADOS

El supervisor y el área piden comparar contra lo publicado, no contra nuestras reproducciones.

| dataset | **nuestro** | *A\*Net pub* | Δ | *NBFNet pub* | Δ |
|---|---:|---:|---:|---:|---:|
| FB15k-237 | 0.3907 | *0.411* | **−0.0203** | *0.415* | −0.0243 |
| WN18RR | 0.5321 | *0.549* | **−0.0169** | *0.551* | −0.0189 |
| **YAGO3-10** | **0.5515** | *0.556* | **−0.0045** | *0.563* | −0.0115 |
| wikikg2 | 0.5216 | *0.6851* | −0.1635 | *OOM* | — |

**No ganamos en ninguno.** El "+0.013" de YAGO era contra **nuestra medición** de A\*Net (0.5390 a
β=200 %), no contra su publicado. YAGO sigue siendo el más cerca: **−0.0045**, dentro de nuestro
ruido (σ=0.0148).

⚠️ **Sesgo sistemático a tener en cuenta**: nuestras reproducciones de A\*Net **con su código y
su config** caen por debajo de lo publicado en los **6 brazos**: −0.0026 (FB237), −0.0059
(WN18RR), −0.0099 (YAGO). Crece con el tamaño del dataset. Causa probable: ellos usan **4 GPUs**
(batch efectivo 256) y posiblemente selección entre corridas. ⇒ Al comparar contra publicados
arrancamos con ~0.005-0.010 de desventaja que **no viene de nuestro método**. Argumento para
reportar **ambas columnas** (publicado y reproducido), que es práctica aceptada.

### 🔴 Corrección teórica que invalida mi propio argumento

Propuse la **atención entre vecinos** (los mensajes entrantes se atienden entre sí antes de
agregar) diciendo que **rompe la cota de rawl2** de los C-MPNN. **ES FALSO.**

Si `m_i = f(h_src_i, r_i)`, cualquier función de los mensajes entrantes — **incluida la atención
completa por pares `Σ_{i,j} g(m_i, m_j)`** — sigue siendo una **función del multiconjunto**
`{m_i}`, y ésa es exactamente la clase que C-MPNN cubre. Para salir haría falta información
**estructural entre los vecinos** (si `u₁` y `u₂` están conectados, triángulos, caminos de
orden 2), no sus features.

Lo que sí queda es un argumento **más modesto pero real**: a ancho finito, sumar `d` mensajes en
un vector de dim 32 pierde información, y la interacción por pares puede computar "evidencia
redundante" con menos parámetros. Es ganancia práctica, **no de clase**.

**La ironía**: lo que SÍ excede WL local es el **PPR** — el PageRank personalizado captura
conectividad global que el message passing no computa en T rondas. Ya estaba implementado y
verificado, y **A\*Net lo usa sólo en wikikg2**.

### 📏 Medición: la distribución de grados de entrada mata la versión ingenua

Instrumentado el forward real (job 93070), grados de entrada **entre las L aristas
seleccionadas**:

| capa (YAGO) | d medio | p50 | p90 | **MAX** | Σd² | vs uniforme `L²/K` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1001.4 | 1 | 2 | **215 801** | 4.6·10¹⁰ | **12 291×** |
| 2 | 4.4 | 1 | 5 | 209 267 | 1.3·10¹⁰ | 3 333× |
| 5 | 4.7 | 2 | 6 | 109 361 | 4.1·10⁹ | 1 086× |

**Mi estimación de `L²/K` = 3.8 M pares se quedó corta por TRES órdenes de magnitud.** El sesgo
es extremo: en la capa 0 un solo nodo recibe **215 801 de las 215 804** aristas.
⚠️ **Sexta estimación fallida del proyecto, y la peor: error de 1000×.**

**Pero el diseño HÍBRIDO sí es viable** (todas las aristas contribuyen por la suma; sólo las
top-M por `alpha` interactúan):

| M | nodos truncados | pares/query (YAGO) |
|---:|---:|---:|
| 16 | 2.40 % | 745 620 |
| **32** | **1.04 %** | **1 284 655** |
| 64 | 0.37 % | 2 135 672 |

Con M=32 el costo baja **3 000×** y sólo se trunca el 1 % de los nodos.

⚠️ **Dos hallazgos que moderan la expectativa**: (a) el **p50 es 1-2** y el p90 es 5-6 ⇒ la gran
mayoría de los nodos recibe una o dos aristas, donde el mecanismo **no hace nada**; el beneficio
vendría del ~10 % con d≥6 (los hubs de H5), así que el efecto agregado será chico aunque funcione
donde aplica. (b) Las capas 0 y 1 son **degeneradas** ⇒ aplicarlo sólo de la capa 2 en adelante.

### Lanzado: PPR en los tres datasets (una variable cambiada)

| job | dataset | referencia sin PPR | falta para el publicado |
|---|---|---:|---:|
| **93071** | WN18RR β=100 % | 0.5321 | +0.017 |
| **93072** | FB15k-237 β=100 % | (93049 en curso) | ~+0.02 |
| **93073** | YAGO β=200 % | **0.5515** | **+0.0045** ← el más alcanzable |

Todos con la config coherente (batch 8 + lr 1e-3) y la receta de cada dataset
(`adversarial_temperature` 1 en WN18RR / 0.5 en los otros; `dependent` y `remove_one_hop` sólo
donde su config lo pide).

### WN18RR con config coherente: la pendiente de β es NEGATIVA

| | β=100 % | β=200 % |
|---|---:|---:|
| batch 32 / lr 5e-3 | 0.5134 | 0.5163 |
| **batch 8 / lr 1e-3** | **0.5321** | **0.5282** |

Con hiperparámetros buenos, **subir β nos BAJA 0.004 en WN18RR**. ⇒ La interacción de H13 **no
replica** ahí, y no era un artefacto de hiperparámetros. H13 queda como fenómeno de YAGO.

---

## 2026-09-06 (b) — 🔴 **Dos diagnósticos míos equivocados seguidos. La causa real se reparte 50/50 entre PODA y BATCH/LR — y hay un confusor ENTRE DATASETS**

### El 2×2 que cierra la pregunta "¿por qué FB15k-237 da 0.39 y antes daba 0.42?"

Comparación **pareada a la misma época (2)**, que es lo que permite atribuir:

| | modelo | batch | lr | valid ép. 2 | costo aislado |
|---|---|---:|---:|---:|---:|
| 92223 | `sparse` (544 230 aristas) | 8 | 1e-3 | — (final **0.424**) | referencia |
| **93038** | `sparse_state` SIN poda | 32 | 5e-3 | **0.379** | **batch/lr: −0.017** |
| **93049** | `sparse_state` con poda | 8 | 1e-3 | **0.378** | **poda: −0.018** |
| 93028 | `sparse_state` con poda | 32 | 5e-3 | 0.361 | ambos: −0.035 |

**Los dos factores pesan casi igual y son aditivos**: 0.018 + 0.017 = 0.035 ≈ los 0.031 de
brecha (0.424 → 0.393). **Ni la poda sola ni batch/lr solo explican la caída.**

Resultado final de 93028 (poda, β=100 %, su config): **test 0.3907** contra **0.4084** de A\*Net
⇒ −0.018. ⚠️ **93029 (β=200 %) murió por TIMEOUT** en la época 14 con valid 0.394; relanzar.

### 🔴 DOS diagnósticos míos equivocados, seguidos

**(1) "Es la poda"** — lo afirmé porque el modelo viejo era `--model sparse` (sin presupuesto).
Lo desarmó el usuario con un dato que yo tenía y no había mirado: **duplicar β daba sólo +0.006**
(0.372 → 0.378), así que el presupuesto **no es la restricción activa** y no podía explicar una
caída de 0.031. Correcto: la poda explica ~la mitad.

**(2) "Es `--dependent`"** — afirmé que el modelo viejo NO lo usaba, con 181 585 params contra
3 102 066. **FALSO**: el log de 92223 muestra `--dependent` en el comando real. Lo "verifiqué"
con un `grep -aoE "--model|--edge_budget|..."` **cuyo patrón no incluía `--dependent`** ⇒
inferí ausencia de una búsqueda que no podía encontrarlo.
⚠️ **Es el mismo error que el descarte del padding-free** (2026-08-30): una verificación
construida de forma que no podía fallar. Segunda vez.
Dato útil que salió igual: **`dependent` aporta +0.07** (0.321 sin él contra 0.393 con él).

### 🔴 Y el hallazgo más incómodo: un CONFUSOR entre datasets

| dataset | batch | lr | vs A\*Net |
|---|---:|---:|---:|
| **YAGO** | **8** | **1e-3** | −0.018 a β=100 %, **+0.0125 a β=200 %** ← lo único que ganamos |
| WN18RR | 32 | 5e-3 | −0.030 |
| FB15k-237 | 32 | 5e-3 | −0.018 |

**El único dataset donde le ganamos a A\*Net es el único donde usamos batch 8 + lr 1e-3**, y
acabamos de medir que batch 32 + lr 5e-3 cuesta ~0.017 en FB15k-237.

Peor: en WN18RR sí probamos batch 8, pero **con lr 5e-3** — la combinación **incoherente**
(batch chico con lr grande). Dio 0.4931 contra 0.5134 de batch 32 + lr 5e-3, y de ahí concluí
"batch 8 es peor en WN18RR". **Nunca probamos batch 8 + lr 1e-3**, que es la combinación
coherente y la que usa YAGO. Jobs **93050/93051**.

Si WN18RR sube de 0.5134 a ~0.53, la brecha cae de 0.030 a 0.013 y **buena parte de lo que
atribuimos a la arquitectura era hiperparámetros**.

### ⚠️ Advertencia metodológica que hay que declarar en el paper

Estamos **ajustando hiperparámetros mirando resultados**. Lo defendible es (a) elegirlos por
`valid_mrr`, (b) reportar el barrido completo y no sólo la mejor combinación, y (c) **decir
explícitamente que nosotros tuneamos y A\*Net corre con su config publicada**. Si terminamos
usando batch 8 + lr 1e-3 en los tres datasets, esa asimetría hay que ponerla en la tabla de
desviaciones, no en una nota al pie.

### Estado del resultado principal (todo con poda, protocolo ~igualado)

| dataset | β=100 % nuestro | A\*Net | Δ | β=200 % nuestro | A\*Net | Δ |
|---|---:|---:|---:|---:|---:|---:|
| YAGO | 0.5280 ± 0.0148 | 0.5461 ± 0.0015 | −0.018 | **0.5515** | 0.5390 ± 0.0109 | **+0.013** |
| WN18RR | 0.5134 | 0.5431 ± 0.0013 | −0.030 | 0.5163 | 0.5451 ± 0.0002 | −0.029 |
| FB15k-237 | 0.3907 | 0.4084 ± 0.0014 | −0.018 | (timeout) | — | — |

**Sólo ganamos en YAGO y sólo a β alto** — y ese resultado está ahora bajo sospecha de
confusor por hiperparámetros. Es el estado honesto.

### Interacciones de escalado (H13) tras corregir el batch de WN18RR

| dataset | nuestra pendiente | la suya | interacción |
|---|---:|---:|---:|
| YAGO | +0.0235 | −0.0071 | **+0.0306** |
| WN18RR | +0.0029 | +0.0020 | **+0.0009 (nula)** |
| FB15k-237 | ? (timeout) | ? | ? |

---

## 2026-09-05/06 — 🔴 **Tres cambios simultáneos otra vez**: la caída del GT en FB15k-237 y WN18RR NO es la poda (probablemente)

### WN18RR: el batch era el problema (parcialmente)

Su batch en WN18RR es **256** (`batch_size: 64` × 4 GPUs; Tabla 9 lo confirma), no 64.
Corrimos a **8** ⇒ 217 087 pasos contra sus 6 783, o sea **32× más pasos, 32× más chicos, con
SU lr 5e-3**, que está calibrado para batch 256. En YAGO la desviación era sólo 5× (8 vs 40).

| batch | β=100 % | β=200 % |
|---|---:|---:|
| 8 | 0.4931 | 0.5000 |
| **32** | **0.5134** | **0.5163** |
| | **+0.0203** | **+0.0163** |

Corregir el batch vale **+0.02**. Sigue **−0.03 por debajo de A\*Net** (0.5431/0.5451), pero es
la mitad de la brecha que teníamos.
⚠️ Batch 64 NO entra: medido, β=100 % usa **38.7 GB de 40** y β=200 % da **OOM**. La memoria
escala **casi lineal** con el batch (5.53 GB a batch 8, 19.4 a batch 32).

**Interacción en WN18RR con el batch corregido**: nuestra pendiente +0.0029, la suya +0.0020
⇒ **+0.0009: nula**. En YAGO era +0.0306.

### 🔴 FB15k-237: comparé dos puntos que difieren en TRES variables

El modelo que dio test **0.4183** (job 92223) y las corridas actuales no son el mismo experimento:

| | 92223 (0.4183) | ahora (valid ~0.37) |
|---|---|---|
| modelo | `--model sparse` (TODAS las 544 230 aristas) | `--model sparse_state`, L=54 419 |
| **`dependent`** | **no** — **181 585** params | **sí** — **3 102 066** params (**17×**) |
| **lr** | **1e-3** | **5e-3** |

`--dependent` agrega un `Linear(32 → 475·2·32)` por capa = **2.9 M parámetros extra sobre
272 115 triples de entrenamiento**. Lo puse porque su config de FB15k-237 dice `dependent: yes`,
pero el modelo del 0.4183 no lo tenía.

**El dato que desarma la explicación por poda** (lo notó el usuario): duplicar β da sólo
**+0.006** (0.372 → 0.378). Si el presupuesto casi no mueve la aguja, **no puede explicar** la
caída de 0.418 a 0.37 — que requeriría una curva absurdamente no lineal. ⇒ el sospechoso
principal pasa a ser **`dependent`**, no la poda.

**Ablación lanzada** (una variable cada una):
| job | qué aísla |
|---|---|
| **93038** `fb_nopoda` | K=N, L=558 771 (todas) ⇒ aísla **la poda** |
| **93039** `fb_nodep` | sin `--dependent` ⇒ aísla **dependent** |

### ⚠️ Punto metodológico del usuario, y es correcto

Reportar **sin poda** en los datasets chicos y **con poda** en los grandes sería equivalente a
que A\*Net publicara YAGO/wikikg2 con poda y FB15k-237/WN18RR con NBFNet. **El modelo tiene que
ser uno solo.** Si el GT podado funciona, tiene que funcionar decentemente en los cuatro.
⇒ El brazo sin poda (93038) es **diagnóstico**, NO un resultado reportable.

### Estado del barrido de β (tres datasets)

| dataset | grado máx | nuestra pendiente | la suya | **interacción** |
|---|---:|---:|---:|---:|
| **YAGO** | 61 044 | +0.0235 | −0.0071 | **+0.0306** |
| **FB15k-237** | — | +0.006 (valid) | ? | **?** |
| **WN18RR** | ~4 medio | +0.0029 | +0.0020 | **+0.0009** |

La hipótesis de que **el efecto escala con la asimetría de grado** sigue viva, pero con dos
puntos (uno fuerte, uno nulo) no se sostiene. FB15k-237 es el que decide — y ahora está
contaminado por el confusor de `dependent`, así que hay que resolver la ablación antes.

### Brechas actuales contra A\*Net medido

| dataset | β=100 % | β=200 % |
|---|---:|---:|
| YAGO | −0.0181 (t=−2.10) | **+0.0125** (t=+1.99) |
| WN18RR | −0.0297 | −0.0288 |
| FB15k-237 | ~−0.04 (valid, contaminado) | ~−0.03 (ídem) |

**Sólo ganamos en YAGO y sólo a β alto.** Es el estado honesto del resultado principal.

### Errores propios de esta tanda (patrón repetido)

1. **Tres variables a la vez** en FB15k-237 (poda + dependent + lr). Cuarta vez en el proyecto.
2. **Batch 8 en WN18RR** sin comparar contra su Tabla 9 (32× de desviación).
3. **Batch 64** después, sin calcular: 38.7 GB al borde y OOM en β=200 %.
4. **Memoria de FB15k-237** estimada en ~13/26 GB, real **3.4/5.5** — extrapolé linealmente en
   `L` ignorando que el estado `(B,S,d)` baja con N (14 541 contra 40 943 nodos). Quinta
   estimación de memoria fallida, esta vez por el lado seguro.

---

## 2026-09-04 (b) — ⚠️ **CORRECCIÓN a H13: "A\*Net degrada" era prematuro (n=2). Con n=3 no es significativo — pero se vuelve 7× más INESTABLE.** Y el expander queda refutado por tercera vez

### 🔴 Corrección: la tercera semilla de A\*Net a β=200 % mueve mucho el número

| semilla | 1024 | 1025 | **1026** |
|---|---:|---:|---:|
| MRR | 0.5287 | 0.5378 | **0.5505** |

| | media | σ | pendiente 100→200 |
|---|---:|---:|---:|
| A\*Net β=100 % (n=3) | 0.5461 | **0.0015** | — |
| A\*Net β=200 % (n=3) | **0.5390** | **0.0109** | **−0.0071, se 0.0064, t = −1.11** |

Con **n=2** daba −0.0128 (t = −2.77) y lo reporté como *"A\*Net no satura: DEGRADA"*.
**Con n=3 es t = −1.11: NO significativo.** ⇒ **Retirar la afirmación de que degrada.**
Es exactamente el riesgo que había anotado al escribir H13 (*"no reportar hasta tener las cuatro
celdas en n=3"*); el aviso sirvió, pero el número ya estaba escrito como hallazgo.

**Lo que apareció en su lugar puede ser mejor: su σ es 7.1× MAYOR a β=200 %** (0.0109 contra
0.0015), con un rango de **0.0217** entre semillas. No es que su método empeore en promedio —
**se vuelve INESTABLE**. Es una firma distinta y más específica que un descenso, y sigue siendo
coherente con el mecanismo de dilución: con agregación fija, qué aristas entren en el conjunto
ampliado pasa a importar mucho, y eso depende de la semilla.

### Lo que sigue en pie

| | valor | antes (n=2) |
|---|---:|---:|
| nosotros vs ellos a β=200 % | **+0.0161, t = +2.47** | +0.0218, t = 4.52 |
| nuestra pendiente 100→200 | +0.0250 | (n=1 en β=100 %) |
| interacción | +0.0321 | +0.0378 |
| nosotros vs ellos a β=100 % | −0.0160 (pareado t = −6.24) | — |

**El cruce se mantiene**: a presupuesto bajo ganan ellos, a presupuesto alto ganamos nosotros.
Pero el contraste clave pasó de t=4.52 a **t=2.47**.

⚠️ **Nuestra pendiente sigue anclada en n=1** (β=100 % = 0.5301). Jobs 92948/92949 (semillas
43/44) en curso; **92963** corre β=**200 % EXACTO** (L=431 608) para quitar el 5.7 % de
presupuesto extra que hoy tenemos a favor.

### ✅ EXPANDER: refutado por TERCERA vez, y ésta es la medición limpia

| | MRR | H@1 | H@10 | MR |
|---|---:|---:|---:|---:|
| con expander (deg 4), L=326 700 | 0.5388 | 0.4533 | 0.6964 | 1 238 |
| **sin expander, MISMO L=326 700** | **0.5400** | 0.4572 | 0.6893 | 2 077 |
| | **Δ −0.0012** | | | |

**Nulo, levemente negativo.** El +0.0087 aparente contra el brazo de L=215 804 era **íntegramente
el presupuesto extra**: al agregar 985 456 aristas expander, β=100 % implica L = K·E/N con el E
ya aumentado ⇒ el brazo con expander propagaba **1.51× más aristas**.

**La predicción escrita ANTES de correr (~0.542, rango [0.528, 0.546]) se cumplió: 0.5400.**

⇒ Tres refutaciones independientes: inductivo (ruido), FB15k-237 transductivo (Δ +0.0008, las 6
métricas coincidiendo una a una) y ahora **YAGO a presupuesto igualado (Δ −0.0012)**. Ésta es la
más controlada de las tres: mismo modelo, mismo L, misma semilla, único cambio el expander.
**Línea cerrada con evidencia, no por descarte** — que es lo que se le puede mostrar al supervisor.

Coherente con el marco: los expander resuelven **alcanzabilidad**, y medimos que ése no es el
cuello (0.3 % de los fallos fuera del horizonte, cobertura 98.3 %). El cuello es **escasez de
evidencia**, y una arista aleatoria no es evidencia.

⚠️ Una inconsistencia menor que se arrastra: en el brazo del expander calculé E **incluyendo**
los self-loops y en el de 215 804 **sin** incluirlos (3.9 % de diferencia). No afecta la
comparación expander/control (los dos usan 326 700) pero sí la que va contra 0.5301.

---

## 2026-09-04 — 🎯 **CONFIRMADO: A\*Net no satura con más aristas, EMPEORA. Los dos métodos escalan en direcciones opuestas**

El experimento que decidía el paper dio positivo, y en la dirección fuerte.

### La tabla 2×2 de escalado

| | β=100 % (L≈216 k) | β=200 % (L≈432-456 k) | **pendiente** |
|---|---:|---:|---:|
| **A\*Net** | **0.5461 ± 0.0015** (n=3) | 0.5333 ± 0.0064 (n=2) | **−0.0128** |
| **nuestro GT** | 0.5301 (n=1) | **0.5551 ± 0.0023** (n=2) | **+0.0250** |

Semillas de A\*Net a β=200 %: **0.528735** (1024) y **0.537827** (1025).

- **A\*Net 100 → 200 %: −0.0128, se 0.0046, t = −2.77.** No satura: **degrada**.
- **A β=200 %, nosotros vs ellos: +0.0218, se 0.0048, t = 4.52.**
- A β=100 %, nosotros vs ellos: −0.0160 (pareado: t = −6.24).
- **INTERACCIÓN (diferencia de pendientes): +0.0378** — mucho mayor que cualquier ventaja
  individual. **Ése es el efecto real.**

**Hay CRUCE**: a presupuesto bajo ganan ellos, a presupuesto alto ganamos nosotros. La perilla
que a ellos les hace daño a nosotros nos sirve.

### El mecanismo, predicho ANTES de medir

Su agregación es **suma/PNA: fija, no aprendida** ⇒ cada arista adicional aporta evidencia **y
ruido en la misma proporción** ⇒ a partir de cierto punto **diluye**. Nuestra atención puede
**seleccionar** dentro del conjunto ampliado ⇒ convierte evidencia adicional en precisión.

⇒ El reclamo del paper deja de ser *"les ganamos"* (que a β=100 % es **falso**) y pasa a ser
**"la atención convierte evidencia adicional en precisión; la agregación fija no"**. Es una
propiedad de la ARQUITECTURA, no una ventaja de cómputo, y se sostiene sobre una interacción
medida con el baseline propio.

**Explica además por qué ellos fijaron β=100 % en los CUATRO datasets mientras variaban α 50×**
(10 % → 0.2 %): para su arquitectura subir β no solo no ayuda, **perjudica**. No fue falta de
exploración: es que su óptimo está ahí.

### Lo que falta para que sea reportable (lanzado)

| job | qué | cierra |
|---|---|---|
| **92947** | A\*Net β=200 %, semilla 1026 | su n=2 → n=3 (σ actual 0.0064 = **4× la de β=100 %**) |
| **92948** | GT β=100 %, semilla 43 | nuestro punto de β=100 % es **n=1** y es el que ANCLA la pendiente |
| **92949** | GT β=100 %, semilla 44 | cierra n=3 |
| *pendiente* | GT β=**200 % EXACTO** (L=431 608) | hoy comparamos nuestro **211 %** contra su **200 %**: 5.7 % más de presupuesto a nuestro favor. Vale ~+0.0007 por la pendiente medida (despreciable frente a +0.0218) pero **es mejor medirlo que argumentarlo** |

⚠️ La cuarta no entró: la QOS topa en **4 jobs simultáneos** (no faltan GPUs, hay 6 libres).
`sbatch_gt_b200_s42.sh` queda listo.

Con las cuatro, la tabla 2×2 queda con **n=3 en las cuatro celdas y β exactamente igualado en
ambos lados** ⇒ defendible sin asteriscos.

### Advertencia metodológica

Este resultado descansa en una **interacción**, no en una diferencia de medias. Eso lo hace más
robusto (no depende de ganar en un punto) pero también más frágil a los grados de libertad: con
n=2 en dos de las cuatro celdas, y σ de A\*Net a β=200 % cuatro veces mayor que a β=100 %, la
pendiente de ellos es el número menos establecido de los cuatro. **No escribir el paper sobre
esto hasta tener las cuatro celdas en n=3.**

---

## 2026-09-03 (b) — 🔴 **H12 NO sobrevive a presupuesto igualado: la ventaja era el presupuesto, no la arquitectura**

Rehecho el análisis estratificado sobre el brazo de **β=100 %** (`yago_beta100_ranks.pt`, test
0.5301), que es el que el paper va a usar. El de la entrada anterior era sobre β=211 %.

| grado fuente | n | Δ MRR | se | **t** | gana/pierde |
|---|---:|---:|---:|---:|---|
| 0-27 | 2 984 | −0.0029 | 0.0041 | −0.70 | 806/907 |
| 28-99 | 3 262 | −0.0132 | 0.0035 | **−3.78** | 514/605 |
| 100-249 | 824 | −0.0560 | 0.0103 | **−5.44** | 230/336 |
| 250-999 | 1 671 | −0.0391 | 0.0082 | **−4.75** | 511/674 |
| ≥1000 | 1 223 | +0.0047 | 0.0079 | **0.59** | **396/610** |
| **GLOBAL** | 9 964 | **−0.0158** | 0.0025 | **−6.24** | |

**Perdemos en todos los estratos.** El de hubs —donde estaba toda la tesis de H12— da **t = 0.59
(no significativo)** y con conteos pareados **396/610: perdemos más veces de las que ganamos**.
La media positiva la arrastran pocas ganancias grandes. MR: **2 092 contra 1 840**, también peor.

⇒ **La ventaja de +0.0277 en el estrato de hubs que reporté era el efecto del DOBLE de
presupuesto de aristas, no del mecanismo de atención.** H12 queda retractado tal como estaba
escrito. Es el mismo error de atribución de siempre: comparé dos brazos que diferían en dos
cosas (arquitectura Y presupuesto) y le atribuí la diferencia a una sola.

**Estado real del resultado principal: a presupuesto igualado (β=100 %) perdemos por
−0.0158 (t = −6.24) contra A\*Net medido.**

### Reformulación: la pregunta ya no es "¿podemos usar β=211 %?" sino cómo ESCALA cada método

Nuestra curva presupuesto → precisión **no satura**:

| L (aristas/capa) | nuestro GT | A\*Net |
|---:|---:|---:|
| 215 804 (β=100 %) | 0.5301 | **0.5461 ± 0.0015** (n=3) |
| ~431-456 k (β≈200 %) | **0.5551** (n=2) | **??? ← el experimento** |

**Hipótesis mecánica** (escrita antes de correr): su agregación es **suma/PNA, fija y no
aprendida** ⇒ más aristas aportan evidencia y ruido en la misma proporción ⇒ debería **saturar**.
Nuestra atención puede **seleccionar** dentro del conjunto ampliado ⇒ sigue subiendo. Si se
confirma, el resultado deja de ser *"les ganamos gastando el doble"* y pasa a ser **"la atención
convierte evidencia adicional en precisión; la agregación fija no"** — que es una propiedad de
la arquitectura y sí es publicable.

**Por qué es plausible que ellos nunca lo probaran**: en su Tabla 9 **α varía 50×** entre
datasets (10 % → 0.2 %) pero **β es 100 % en los cuatro**. Un hiperparámetro fijado en el valor
"natural" en todos lados sugiere que no se exploró. Y tienen motivo: su contribución es la
**eficiencia**, y subir β va justo contra esa narrativa.

**Los dos resultados son informativos**: si A\*Net sube a ~0.55 no hay diferenciación por esta
vía y conviene saberlo antes de escribir; si se queda en ~0.546, la tabla 2×2 es inatacable.

Lanzado: `degree_ratio: 2` en su config (`astarnet_cfg/yago310_astarnet_beta200.yaml`, 1 línea
de diff) ⇒ L = 431 608. Jobs **92936** (semilla 1024) y **92941** (1025); falta la 1026.
⚠️ Su L=431 608 contra nuestro 456 244 son 5.4 % menos, o sea les damos algo MENOS de
presupuesto. Por la pendiente medida eso vale ~0.001 de MRR, así que no debería cambiar la
lectura — pero hay que declararlo.

### 🛑 wikikg2 CANCELADO (job 92813)

Con la receta completa iba **peor** que la versión simple, a épocas comparables:

| | valid por época |
|---|---|
| 92813 (PPR + break_tie + edge_dropout + 8192 neg) | 0.465 / 0.392 / 0.415 / **0.483** / 0.464 |
| 92755 (sin las 4 opciones) | 0.434 / 0.526 / **0.576** / 0.573 |

La `train_loss` **sí baja** (0.548 → 0.445 → 0.463) mientras el valid oscila sin tendencia ⇒
desajuste train/eval, no falta de entrenamiento.

**Sospechoso principal, y es un error MÍO de porte**: `nn.Dropout` **reescala los sobrevivientes
por `1/(1-p)`**, lo que con agregación por suma preserva la esperanza. Mi implementación solo
ELIMINA el 20 % de las candidatas, sin reescalar. Con `prune_attn sigmoid` (que **no normaliza**)
el agregado en train queda ~20 % más chico que en eval — exactamente el patrón observado.
Segundo sospechoso: el PPR podría estar lavando el contraste del labeling trick, porque ahora
todos los nodos arrancan con estado no-nulo.

**Qué se pierde y qué no**: la contribución de escala **sigue en pie** — el modelo corre a
2.5 M entidades en **11.87 GB** donde **NBFNet da OOM**, medido y documentado. Lo que se cancela
es buscar un número competitivo ahí. Si se retoma: **ablación de las 4 opciones empezando por
sacar `edge_dropout`**, NO más épocas.

### Prioridad de experimentos (revisada)

1. **A\*Net β=200 %, n=3** — decide si hay contribución. En curso.
2. **β=100 % con SU lr (5e-3)** — usamos 1e-3, heredado del inductivo y **nunca justificado**.
   Parte del −0.0158 podría ser esto y no la arquitectura.
3. **Selección de frontera consciente de la relación** — la única ventaja estructural
   identificada que no estamos usando (su prioridad no ve la relación; su capa 0 elige a ciegas).
4. **n≥3 en el brazo de β=100 %** (hoy n=1 contra su n=3).
5. Control del expander (92916, en curso) y PPR en YAGO.

---

## 2026-09-03 — 🎯 **H6 se REPLICA en YAGO y queda como el argumento mecánico del paper**: la ventaja de la atención es significativa SOLO en el régimen de inundación

Análisis estratificado del GT contra **A\*Net medido**, alineando por `(h, r, t)` (no por
posición: los dos harness ordenan el test distinto y nuestra convención de relaciones es
intercalada `2r`/`2r+1` contra su offset `r+R`). **9 964 de 10 000 alineadas (99.6 %)**.

### Test PAREADO por estrato de grado de la fuente

| grado fuente | n | Δ MRR | se | **t** | gana/pierde/empata |
|---|---:|---:|---:|---:|---|
| 0-27 | 2 984 | +0.0044 | 0.0042 | 1.05 | 897/839/1248 |
| 28-99 | 3 262 | +0.0053 | 0.0035 | 1.52 | 555/534/2173 |
| 100-249 | 824 | −0.0051 | 0.0082 | −0.62 | 250/246/328 |
| 250-999 | 1 671 | +0.0097 | 0.0068 | 1.42 | 520/502/649 |
| **≥1000** | **1 223** | **+0.0277** | 0.0071 | **3.92** | **660/313/250** |
| GLOBAL | 9 964 | +0.0076 | 0.0023 | 3.30 | |

**Un solo estrato es significativo, y es el de los hubs.** Aporta el **44.4 % de la ventaja con
el 12.3 % de las queries** y su Δ es **3.6× el global**. Sumando ≥250: **65.6 %**. En los demás
estratos los conteos pareados están casi empatados (897/839, 555/534, 520/502) ⇒ ruido alrededor
de cero. **El hueco del estrato 100-249 que parecía una anomalía era ruido** (t = −0.62,
conteos 250/246): no hay nada que explicar ahí.

Y es el régimen **más difícil**: MRR absoluto 0.29-0.32 en ≥1000 contra 0.72 en 28-99. Ganamos
donde el problema realmente falla. MR: **762.5 contra 1 840.5** (2.4×).

### La explicación estructural (y es la tesis del paper)

A\*Net pesa **NODOS**: `layer_input = F.sigmoid(subgraph.score).unsqueeze(-1) * subgraph.hidden`
(`AStarNet/reasoning/model.py:319`). Todas las aristas salientes de un nodo reciben el **mismo**
peso ⇒ **no pueden decir "este hub es relevante por la relación r₁ pero no por r₂"**. Nosotros
pesamos **ARISTAS**: `softmax/sigmoid(q·k + b_rel)`. Esa libertad extra sólo rinde cuando hay
muchas aristas que discriminar — o sea **en hubs**, exactamente donde se concentra la ventaja.

⇒ El reclamo del paper deja de ser "la atención es mejor" y pasa a ser **"la atención gana en el
régimen de inundación, por una razón estructural que A\*Net no puede replicar"**, con la
predicción mecánica hecha ANTES de medir y confirmada en dos datasets (FB15k-237 y YAGO).

⚠️ **n=1 por brazo**: estos t comparan ESTAS DOS CORRIDAS, no los dos métodos. La localización
del efecto está establecida; la magnitud necesita más semillas (tenemos la 43 nuestra y las
1025/1026 de A\*Net sin volcar).

### 🔴 Ventaja estructural que tenemos y NO estamos usando

Nuestra selección de aristas rankea por **estado del destino**:
```python
d_prio = torch.where(d_found, prio.gather(1, d_slot), base.expand(...))
```
Es la heurística de A\*Net portada tal cual — y **hereda su punto ciego**: su `score` es función
de `(hidden, query)` **sin ningún término de relación** (`model.py:322`). En la **capa 0** todos
los destinos no visitados tienen el mismo estado ⇒ la misma prioridad ⇒ **la primera frontera se
elige de forma esencialmente arbitraria**. Por eso necesitan `break_tie: yes`: es un parche a que
la decisión no tiene información.

Nosotros **sí** tenemos información en la capa 0: `b_rel[head, r]` dice cuán compatible es la
relación con la query **sin depender de ningún estado de nodo**. Ellos no pueden calcularlo.

⇒ **Propuesta: selección de frontera CONSCIENTE DE LA RELACIÓN**, rankeando por
`prioridad_destino + λ·compatibilidad(r, r_q)`. Reutiliza tablas existentes, es barato, y ataca
el punto ciego de su algoritmo. Evidencia indirecta de que la dirección es correcta: nuestro MR
ya es 2.4× mejor en YAGO y el suyo es 4.3× peor que NBFNet en FB15k-237 — el MR es justo la
métrica que castiga las decisiones tempranas malas de la búsqueda.

### Sobre los expander graphs (respuesta al supervisor)

Los expander resuelven **alcanzabilidad**, y medimos que ése **no es nuestro cuello**: sólo el
**0.3 %** de los fallos está fuera del horizonte de 6 saltos (H4) y en YAGO la cobertura es
**98.3 %**. Optimizan un problema que ya está resuelto. El cuello medido es **escasez de
evidencia**, y una arista aleatoria no es evidencia — además **homogeneiza la distancia al head**,
que es la señal de la que depende el labeling trick.

Predicción cuantitativa: a presupuesto FIJO el expander debe **perjudicar** (desplaza aristas
reales por ruido); a β fijo queda **neutro** (el presupuesto extra compensa). Es lo que mide el
control 92916.

**La versión defendible de la intuición del supervisor es el PPR**: inyecta posición global en la
condición de borde **sin fabricar hechos**. Ya está implementado y verificado, pero **nunca se
corrió en YAGO** (sólo en wikikg2). Si el PPR tampoco mueve la aguja, la conclusión es más fuerte
que "el expander falló": sería que **la información de posición global no es lo que le falta al
modelo en YAGO**.

### En curso

| job | qué | estado |
|---|---|---|
| 92916 | control del expander (sin expander, L=326 700) | valid 0.527 tras 1 época |
| 92813 | wikikg2, receta completa, 2 GPUs | valid 0.465/0.392/0.415/0.483/0.464 (época 5 de 27) ⚠️ oscilando |

---

## 2026-09-02 — Revisión como evaluador: faltaban CUATRO opciones que A\*Net usa SOLO en wikikg2

Auditoría completa de metodología, implementación y evaluación contra el código de `AStarNet/`
y contra la página oficial de OGB. **Veredicto: YAGO sólido, wikikg2 NO defendible hasta hoy.**

### 🔴 Su config de wikikg2 difiere de la de YAGO en SEIS puntos; replicábamos UNO

`diff yago310_astarnet.yaml wikikg2_astarnet.yaml`:

| | A\*Net wikikg2 | A\*Net YAGO | nosotros (antes) | |
|---|---|---|---|---|
| `indicator_func` | **`ppr`** | onehot | onehot | ❌ |
| `num_negative` | **1 048 576** | 32 | 32 | ❌ **32 768×** |
| `edge_dropout` | **0.2** | 0 | 0 | ❌ |
| `break_tie` | **yes** | no | no | ❌ |
| `test_node_ratio` | **0.01** | (no existe) | — | ✅ el 2026-09-01 |
| `aggregate_func` | sum | pna | atención | (es el punto del paper) |

**El PPR es el que más pesa conceptualmente.** Su paper: *"instead of using a boundary condition
of mostly zeros, we find it is better to incorporate distance information"*, con
`1_q(u=v) = 1(u=v)·q + 1(u≠v)·p_{u,v}` y `p_{u,v}` un embedding sobre el PPR **discretizado**
(`AStarNet/reasoning/model.py:349-366`). Nosotros poníamos **cero**. Importa justo a esta
escala: con α = 0.2 % la búsqueda alcanza **≤ 7.7 % de N**, así que el 92 % restante llegaba al
readout **sin ninguna información**.

**`break_tie` es NUESTRO PROPIO BUG, por tercera vez.** Su `multikey_argsort`
(`functional.py:4`) usa `randperm` en vez de `arange` cuando está activo. Sin él los empates del
top-k se rompen **por posición en el CSR = orden del archivo del dataset** — el MISMO criterio
arbitrario que el desalojo por id global (2026-08-27) y el `edge_cap` (2026-08-30). Y en las
primeras capas casi todos los candidatos empatan en la prioridad `base`, o sea el sesgo es
masivo. **Tercera aparición del mismo modo de fallo; ellos lo tienen resuelto con un flag.**

**`num_negative` 1 048 576 vs 32** es la diferencia más grande y no la habíamos visto: usan el
**42 % de las 2.5 M entidades** como negativos por positivo. Es otro régimen de entrenamiento.

### ✅ Lo que SÍ está correcto (verificado contra su código)

- **Filtrado y ranking de YAGO idénticos a torchdrug** (`tasks/reasoning.py:168-201`): filtran
  contra el grafo COMPLETO (train+valid+test), excluyen el gold del mask, ranking **pesimista**
  `sum(pos_pred <= pred) + 1`. Nuestro `sum(score >= answer) + 1` es la misma fórmula.
- **Ambas direcciones en YAGO**: nuestro test son 10 000 items = 5 000 tripletas × 2 (5 000
  relaciones pares + 5 000 impares). Su `target` también apila las dos direcciones.
- **Quitar la arista de la query**: su `remove_easy_edges` corre **antes** de
  `undirected(add_inverse=True)` (`model.py:96-100`), así que borrar la directa borra también la
  inversa; y solo en train (`if all_loss is not None`). Equivalente al nuestro.
- **Protocolo OGB**: fórmula de rank exacta (`max|dif| = 0`), negativos **pre-filtrados por OGB**
  (verificado: 0 de 40 M son tripletas del KG ⇒ **no hay que filtrar de nuevo**), mismo
  protocolo en valid y test, grafo de evaluación = solo aristas de train.
- **Convención de inversas**: nuestra `2r`/`2r+1` ≡ su offset `r+R` (`torchdrug/data/graph.py:1683`).
- **Datos**: md5 de YAGO idénticos a los que descarga torchdrug.

### ⚠️ YAGO — desviaciones que hay que DECLARAR

1. **`lr` 1e-3 contra su 5e-3.** Es lo más criticable del resultado de YAGO: **nunca lo
   justificamos**, viene heredado de la config del inductivo. Un revisor puede pedir su lr y, si
   mejora, acusarnos de haber elegido el que nos favorece. **Correr un brazo con lr 5e-3.**
2. **batch 8 contra 40** (igualamos triples vistos: 431 600 vs 400 000, ×1.08).
3. **β 211 % contra 100 %** — elegido por `valid_mrr` (0.559 vs 0.537), criterio correcto, con
   ablación reportada.
4. **n=2 contra n=3** del baseline.

### Implementado hoy (pendiente de verificación, job 92801)

- **`--indicator ppr`** + `--num_indicator_bin 10`. `personalized_pagerank()` portado de
  `data.py:293` con sus parámetros exactos (α = 0.8, 20 iteraciones, peso `1/grado_saliente`,
  propagación fuente→destino). ⚠️ La condición de borde se aplica **también a los nodos que se
  activan en capas posteriores** (`_boundary` en el merge), que es donde estaba el problema real.
  El `(B,N)` de PPR es el **único** denso sobre N: 120 MB a batch 12, **una vez por batch**, no
  por capa ⇒ no rompe el diseño de estado disperso.
- **`--break_tie`** → pasa a `variadic_topks`, que ya lo soportaba (se portó verbatim).
- **`--edge_dropout_p 0.2`** — ellos lo aplican como `Dropout` sobre `edge_weight`
  (`model.py:97-99`); sin `edge_weight`, el equivalente es sacar esas aristas del pool antes del
  top-L. Solo en train, igual que ellos.
- **`num_negative` masivo** — muestreo por rechazo en chunks para no pedir 100 MB de golpe.

**Defaults sin cambios** (`onehot`, sin break_tie, dropout 0, 32 negativos) ⇒ **nada de lo ya
medido en FB15k-237 / WN18RR / YAGO cambia**.

### Estado de wikikg2

`wikikg2_v2` (92755) corre con los fixes del 2026-09-01 (validación intercalada +
`test_node_ratio`) y va en valid **0.434 → 0.526 → 0.576 → 0.573** (época 4 de 53) — ya sobre el
0.4964 de la corrida anterior, pero **le faltan estas cuatro opciones**, así que también queda
invalidado como número reportable. Sirve como referencia de cuánto aportan los cuatro fixes.

⚠️ **No relanzar hasta que 92801 verifique las cuatro** (PPR suma 1 por query y bins en rango,
`break_tie` cambia la selección, dropout solo en train, 1 048 576 negativos sin reventar y
estrictos). Es la disciplina que evitó los dos desastres del desalojo.

---

## 2026-09-01 — 🔴 La evaluación de wikikg2 estaba MAL: medía solo predicción de cola. Y falta `test_node_ratio`

También cierra **A\*Net YAGO con n=3** ⇒ el resultado de YAGO ya es reportable.

### ✅ YAGO cerrado: ganamos por +0.0090 (t = 4.81)

| | MRR | n |
|---|---|---:|
| **nuestro GT (sin padding)** | **0.5551 ± 0.0023** | 2 |
| A\*Net **medido por nosotros** | 0.5461 ± 0.0015 | 3 |
| *A\*Net publicado* | *0.556* | — |

**Ventaja +0.0090 ± 0.0019 (t = 4.81)**, sin elegir semillas. Semillas de A\*Net: 0.546838 /
0.544333 / 0.547134. Reproduce **−0.0099** bajo su número publicado ⇒ sexto brazo de seis por
debajo (−0.0003, −0.0026, −0.0044, −0.0059, −0.0092, −0.0099). ⚠️ Lo nuestro sigue en n=2.

---

### 🔴 El bug: la validación de wikikg2 medía SOLO predicción de cola

`OGBEvalSet` replicaba el orden de `OGBLKGTest` (`AStarNet/reasoning/dataset.py:274`): primera
mitad items de COLA, segunda mitad de CABEZA. **Ellos evalúan el conjunto completo, así que el
orden no importa. Nosotros SUBSAMPLEAMOS** con `limit_val_batches 40` ⇒ los primeros 1 280
índices son **100 % cola, 0 % cabeza**.

Por eso el job 92695 reportaba `valid_mrr = 0.732`, **por encima del 0.6767 de A\*Net**. No era
un buen resultado: era media métrica.
**Corregido**: `OGBEvalSet.__getitem__` ahora intercala (`i//2, i%2==0`) ⇒ cualquier prefijo
queda 50/50. ⚠️ **La selección de checkpoints del job 92695 es inválida** — eligió por la mitad
fácil. Hay que reentrenar.

### La métrica real del mejor checkpoint (época 0, 64 000 items balanceados)

| dirección | MRR | H@1 | H@3 | H@10 | MR |
|---|---:|---:|---:|---:|---:|
| **cola** (h,r → t) | **0.7573** | 0.7189 | 0.7863 | 0.8188 | 46.6 |
| **cabeza** (t,r⁻¹ → h) | **0.2354** | 0.1699 | 0.2554 | 0.3594 | 146.4 |
| **AMBAS** (la métrica) | **0.4964** | 0.4444 | 0.5208 | 0.5891 | 96.5 |

se = 0.0018. A\*Net publicado: valid 0.6767, **test 0.6851** (NBFNet: **OOM**).

### La asimetría 3.2× es del DATASET, no un bug nuestro (medido)

Ambigüedad en el grafo de train, para las consultas de test:

| | destinos distintos por consulta | grado del nodo desde el que propaga |
|---|---:|---:|
| **cola** `(h,r) → t` | media **0.41**, mediana 0, máx 311 | mediana **8**, media 44.8 |
| **cabeza** `(t,r⁻¹) → h` | media **8 605**, mediana 265, máx **1 510 473** | mediana **1 268**, media **22 032** |

Las relaciones de wikikg2 son casi **funcionales** hacia adelante y masivamente **muchos-a-uno**
hacia atrás (*(Canada, citizen, Hinton)*: el país dado el ciudadano es único; el ciudadano dado
el país son millones). Predecir cabeza arranca desde un hub de 22 000 salientes y elige entre
miles de respuestas válidas. **A\*Net enfrenta lo mismo** — su 0.6851 promedia ambas.
⇒ Nuestra debilidad está **localizada en la dirección de cabeza**, y es el mismo modo de fallo
que H4/H5 en FB15k-237 (escasez de evidencia en fuentes hub), acá en su forma extrema.

### 🔴 Diferencia de protocolo encontrada: `test_node_ratio`

`AStarNet/config/transductive/wikikg2_astarnet.yaml`:
```yaml
node_ratio: 0.002        # entrenamiento: 0.2 %
test_node_ratio: 0.01    # evaluación:    1 %   ← 5x MÁS
```
**Expanden 5× más nodos al evaluar que al entrenar.** Nosotros usábamos 0.002 en ambos.
⚠️ En sus configs de FB15k-237 / WN18RR / YAGO `test_node_ratio` **no existe** (los dos ratios
coinciden) — por eso nunca había aparecido. Solo importa en wikikg2, y es justo lo que más
debería ayudar a la dirección de cabeza.

**Implementado** `--test_top_nodes` / `--test_edge_budget` (default 0 = igual que train, así
nada de lo ya medido cambia):

| | K | L | S = 6·L |
|---|---:|---:|---:|
| train (0.2 %) | 5 001 | 64 434 | 386 604 (15.5 % de N) |
| **test (1 %)** | **25 006** | **322 183** | **1 933 098 (77.3 % de N)** |

`L` escala **con** K para mantener β constante (su `es = β·ks·E/N`): subir solo K cambiaría dos
hiperparámetros a la vez — el mismo error de atribución que cometí con `edge_cap` y β en YAGO.
`S` también crece porque la cota del alcance es 6·L; sin eso volvería el desalojo.
Job **92742** compara el MISMO checkpoint con K = 5 001 vs 25 006 (aísla el efecto sin reentrenar).

### Auditoría del protocolo OGB — verificada contra su código y contra la página oficial

| | estado |
|---|---|
| Ambas direcciones, 50/50 | ✅ tras el fix del ordenamiento |
| Fórmula de rank oficial (`0.5·(opt+pess)+1`) | ✅ `max\|dif\| = 0` contra `Evaluator._eval_mrr` (job 92654) |
| Positivo en índice 0 + 500 negativos de OGB | ✅ |
| Negativos YA filtrados por OGB ⇒ **no filtrar de nuevo** | ✅ **verificado**: 0 de 40 M negativos son tripletas del KG |
| El mismo protocolo aplica a **valid** y a test | ✅ usamos `valid_head_neg`/`valid_tail_neg` |
| Grafo de evaluación = solo aristas de train | ✅ igual que su `fact_graph` |
| Convención de inversas consistente | ✅ nuestra 2r/2r+1 ≡ su offset r+R (`torchdrug/data/graph.py:1683`) |
| `test_node_ratio` 5× | ❌ **faltaba** — implementado hoy |
| Subsample 64 000 de 1 197 086 | ⚠️ declarar (se = 0.0018) |

⚠️ **El split de wikikg2 es TEMPORAL**, no aleatorio (Wikidata mayo / agosto / noviembre 2015 →
train / valid / test). Dos consecuencias: (a) una caída valid→test **no es sobreajuste**, es
cambio de distribución; (b) es un régimen más difícil que el split aleatorio de FB15k-237/YAGO.

⚠️ **wikikg2 y YAGO usan métricas DISTINTAS en nuestro propio harness**: wikikg2 rankea contra
500 negativos fijos (protocolo de OGB, porque rankear contra 2.5 M costaría días) y YAGO contra
las 123 182 entidades (full-filtered, lo que hacen los papers). **Declararlo en el paper**: no
se pueden comparar 0.5551 y 0.4964 como si fueran la misma métrica.

### Por qué murió el job 92695

**OOM en la época 2** (no terminó solo): la memoria subió de 28.5 a 31.6 GB y falló con **7.18 GB
reservados sin asignar** ⇒ fragmentación. Al relanzar: bajar el batch a 12, o
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Pendiente

- Resultado de 92742 (K de test 5 001 vs 25 006).
- **Reentrenar wikikg2**: la selección de checkpoints del 92695 es inválida (validación sesgada).
- Tercera semilla del GT en YAGO (hoy n=2 contra n=3 del baseline).

---

## 2026-08-31 — 🚀 **ogbl-wikikg2 CORRE** (11.87 GB, 2.5 M entidades). Y el desalojo no hacía falta

### Lo primero: el baseline propio de A\*Net en YAGO cambia la conclusión

Con **A\*Net medido por nosotros** (job 92611, n=1, su config, 2 líneas de diff, datos con md5
idéntico al que descarga torchdrug) ganamos en **las cinco métricas**:

| | nuestro (n=2) | **A\*Net MEDIDO** | Δ | *A\*Net publicado* | Δ pub |
|---|---:|---:|---:|---:|---:|
| MRR | **0.5551 ± 0.0023** | 0.5468 | **+0.0083** (3.5 σ) | *0.556* | −0.0009 |
| H@1 | **0.4751** | 0.4660 | **+0.0091** | *0.470* | +0.0051 |
| H@3 | **0.6035** | 0.5953 | **+0.0081** | *0.611* | −0.0075 |
| H@10 | **0.6977** | 0.6924 | **+0.0053** | *0.707* | −0.0093 |
| MR | **1 006** | 2 489 | **2.5× mejor** | — | — |

**A\*Net YAGO reproduce −0.0092 bajo su número publicado ⇒ 5 de 5 brazos por debajo**
(−0.0003, −0.0026, −0.0044, −0.0059, −0.0092). Contra la tabla publicada empatábamos y
ganábamos 1 de 4 métricas; contra el baseline medido ganamos 5 de 5, **sin elegir semillas**.
⚠️ A\*Net YAGO es **n=1**; faltan las semillas 1025/1026.

### β = 100 % cuesta −0.025 y entrena peor

| | test MRR | H@1 | MR | mem | mejor época |
|---|---:|---:|---:|---:|---:|
| β ≈ 211 % (n=2) | **0.5551** | 0.4751 | **1 006** | 31.6 GB | 8-9 de 10 |
| β = 100 % (92602) | 0.5301 | 0.4443 | 2 359 | 22.8 GB | **1 de 10** ⚠️ |

Con β=100 % el valid oscila sin subir (0.508/0.537/0.512/0.503/0.481/0.510/0.474) y nos deja
**por debajo** del A\*Net medido.

🔴 **CORRECCIÓN de encuadre**: veníamos llamando a β «desviación de protocolo» (§6.0). **No lo
es.** En su Tabla 9 α y β son **hiperparámetros que ellos ajustan por dataset** — α va de 10 %
en FB15k-237/WN18RR/YAGO a **0.2 % en wikikg2**, 50× de rango. Elegir β=211 % es *elegir un
hiperparámetro*, y lo elegimos por `valid_mrr` (0.559 vs 0.537), que es el criterio correcto.
Hay que **declarar el valor y reportar la ablación**, no pedir disculpas por él.

### 🚀 ogbl-wikikg2: corre

Datos convertidos (`prep_wikikg2.py` → `/home/jreutter/datasets/ogb/wikikg2.pt`):
N = 2 500 604, R = 535 (×2 = 1070 con inversas), E = **32 218 364**, train 16.1 M triples.
Config de A\*Net (Tabla 9 + `wikikg2_astarnet.yaml`): **α = 0.2 %** (`node_ratio: 0.002`,
**NO 20 %**), β = 100 %, batch 128 = 32 × 4 GPUs, 0.2 pasadas ⇒ K = 5 001, L = 64 434.

| config | memoria | tiempo/iter |
|---|---|---:|
| **batch 8 + `--grad_ckpt`** | ✅ **11.87 GB** de 40 | 1.27 s |
| batch 8 SIN checkpointing | ❌ OOM | — |
| batch 32 + checkpointing | ❌ OOM | — |
| batch 128 + checkpointing | ❌ OOM | — |

**Sin checkpointing no corre ni a batch 8.** Es el argumento honesto para usarlo: no gana una
tabla de memoria, es la diferencia entre correr y no correr. (Y está en el mismo framework que
ellos usan — `torchdrug/layers/conv.py:88`, `gradient_checkpoint = False` por defecto; A\*Net
no lo necesita porque `rspmm` ya le evita materializar los mensajes por arista.)

### ✅ El bloqueador del desalojo se DISUELVE

Con α = 0.2 % el conjunto alcanzable está **acotado por `6 capas × L = 386 604` nodos = 15.5 %
de N** ⇒ `node_slots` **nunca se desborda** y no hay que desalojar. El problema que costó dos
intentos fallidos (`evict prio` valid 0.073, `evict keep` valid 0.013 + NaN) **no aparece a esta
escala**: era un artefacto de haber puesto S al 29 % de N en YAGO, no un problema de wikikg2.

### El fix que lo hizo entrar: `_priority` materializaba un `(B, S, 2D)`

El smoke test reventó exactamente en `src/model.py:1590`, `self.prio_g(torch.cat([x, q], -1))`
— y ese cálculo estaba **fuera** de la región checkpointeada. Con S = 386 604 son **792 MB por
capa** retenidos para el backward, ~9-12 GB de los 23 medidos.

Un `Linear` sobre una concatenación se parte **exactamente**: `W[x;q] + b = x·Wₓᵀ + q·W_qᵀ + b`,
y como `q_emb` es `(B, D)` su término es `(B,1,D)` que **se difunde** sin expandir a `(B,S,D)`.
Verificado antes de usarlo: **max|dif| 1.19e-07**. Además se checkpointeó `_priority`.
⇒ **23.12 GB → 11.87 GB**, casi la mitad, sin cambiar nada del modelo.

⚠️ **Séptima estimación de memoria fallida**: predije ~14 GB y el primer intento dio 23. El
patrón se repite — mis modelos analíticos de memoria subestiman de forma sistemática porque
cuento los tensores «principales» y no los intermedios de las operaciones auxiliares. **Medir,
no estimar.**

### Gradient checkpointing (`--grad_ckpt`), verificado

| | pico (YAGO, batch 8) | loss |
|---|---:|---:|
| sin | 22.14 GB | 11.802240 |
| **con** | **8.40 GB** | 11.802239 |

Gradientes: **max|dif| 2.66e-06** sobre 124 parámetros.
⚠️ La salida daba `max|dif| 1.15e-01`, que parecía un bug. **Control**: dos corridas idénticas
**sin** checkpointing dan **1.395e-01** ⇒ es la no-determinancia de CUDA del harness
(`index_add_`/`scatter_add_`, ya documentada en CLAUDE.md), no el checkpointing.

**Cómo reportarlo sin que sea criticable**: la memoria de la tabla comparativa contra A\*Net va
**sin** checkpointing (22.8 GB contra sus 13); el checkpointing se presenta aparte como **lo que
habilita wikikg2**. Declarar la asimetría: ellos pueden apagarlo porque `rspmm` les resuelve el
problema; nosotros no podemos usar `rspmm` porque nuestros pesos de arista requieren gradiente.

### Pendiente inmediato

- **Protocolo de evaluación de OGB**: wikikg2 rankea contra **500 negativos fijos** por triple
  (`head_neg`/`tail_neg`) con el `linkproppred.Evaluator` oficial
  (`AStarNet/reasoning/task.py:110`), **no** full-filtered sobre las 2.5 M entidades como hace
  nuestro `train.py:246`. Sin implementarlo podemos entrenar pero **no reportar un número
  comparable**. Es la pieza que sigue.
- **Batch**: topamos en 8 contra sus 128 ⇒ **16× menos** por paso. A 1.27 s/iter sus 0.2 pasadas
  son ~2 h 15, así que el problema no es tiempo sino el tamaño del paso de optimización.
- Semillas 1025/1026 de A\*Net YAGO (hoy n=1).
- Su config trae además `indicator_func: ppr`, `break_tie`, `edge_dropout: 0.2` y
  `num_negative: 1048576` (contra nuestros 32), que no soportamos.

---

## 2026-08-30 (b) — `edge_budget` estaba al 211 % de β, y eso CONFUNDE el primer escalón del barrido de `cap`

Al responder de dónde salían los 31.6 GB apareció un error de diseño experimental mío.

### `edge_budget` es el `es` de ellos, y lo pusimos al doble

`AStarNet/reasoning/model.py:258` — `es = degree_ratio · ks · E/N`, con `ks = node_ratio · N`.
Son los dos hiperparámetros de búsqueda del paper: **α = `node_ratio`**, **β = `degree_ratio`**.
En YAGO ellos usan **α 10 %, β 100 %** (Tabla 9) ⇒ con E/N = 17.52 y K = 12 318, **L = 215 804**.

Nosotros corrimos con **456 244 ⇒ β ≈ 211 %**, el doble. La diferencia de forma (ratio vs
número absoluto) escondió la de fondo.

### 🔴 El primer escalón del barrido cambió DOS cosas

| corrida | `edge_budget` | `edge_cap` | MRR |
|---|---:|---:|---:|
| S=N | 228 122 (β≈106 %) | 64 | 0.4498 |
| cap128 | **456 244 (β≈211 %)** | **128** | 0.4833 |
| cap256 | 456 244 | 256 | 0.5147 |
| cap512 | 456 244 | 512 | 0.5317 |
| cap1024 | 456 244 | 1024 | 0.5478 |

El script de cap128 decía «sube cap 64→128 y **L en proporción**». Pero `cap` y `L` son **ejes
independientes**: uno es el tope por nodo al RECOLECTAR candidatas, el otro cuántas SOBREVIVEN
al top-L. Duplicar el segundo «en proporción» al primero no tenía justificación.
⇒ **El +0.034 de ese tramo NO es atribuible al `cap` solo.** Corregido en H10 del doc del paper.

**La conclusión de fondo NO cambia**: los tramos 128→256→512→1024 tienen β FIJO y son limpios
(+0.031, +0.017, +0.016), y sostienen por sí solos que el `cap` es palanca y no ha saturado.

### De dónde salen los 31.6 GB (contra 13 de A\*Net y 26.1 de NBFNet)

**A\*Net calcula sobre NODOS, nosotros sobre ARISTAS.** `rspmm` fusiona mensaje y agregación ⇒
los mensajes por arista nunca se materializan, su estado es `O(B·K·d)` ≈ 15 MB/capa. Nuestra
atención retiene `q_e, k_e, v_e, g_e, logit, alpha, msg` por arista para el backward
(`src/model.py:1153-1186`): un `(B,L,d)` son 0.44 GB ⇒ ~2.6 GB/capa ⇒ **~15.7 GB en 6 capas**,
del orden de los 31.6 medidos. El factor es **L/K = 37×**.

Ya estaba documentado que no podemos usar `rspmm` (se desactiva cuando los pesos de arista
requieren gradiente, `AStarNet/reasoning/layer.py:132`) y lo había cuantificado **sólo en
tiempo (~3×)**. **En memoria es mucho peor que 3×** y eso no estaba medido. Es el precio
estructural de la atención en esta familia, no una ineficiencia optimizable.
NBFNet gasta 26.1 GB por lo contrario: todas las aristas y estado sobre todos los nodos.

### Lanzado: β = 100 % (job 92602, en cola)

`edge_budget 215804`, calculado con **su** fórmula `β·K·E/N` = 1.0 · 12 318 · 17.5194. Confirma
que veníamos en **211.4 %**. **Único cambio contra el job 92593** (sin padding): el
`edge_budget`. Todo lo demás idéntico — `--edge_cap 0`, S=N, K=12 318, sigmoid, dependent,
0.40 pasadas, semilla 42 ⇒ el contraste aísla β limpiamente, que es justo lo que no se hizo la
primera vez.

**Dos predicciones escritas antes de correr:**
1. **Memoria ~17-20 GB.** Si los tensores por arista son el término dominante (~15.7 de 31.6 GB),
   partir L por 2.11 tiene que bajarla mucho. **Si NO baja, el modelo de memoria está mal** y el
   gasto viene de otro lado — ésa es la parte informativa.
2. **MRR: no se predice.** No hay con qué: el único tramo del barrido donde β cambió está
   CONFUNDIDO con el `cap`. Por eso se corre.

**Lo que está en juego no es sólo memoria.** Si β=100 % cuesta poco MRR conviene quedarse ahí
aunque sobre GPU: **elimina una desviación de protocolo declarada** (§6.0) en vez de arrastrarla,
y libera memoria para atacar la otra (batch 8 contra sus 40, §6.1). Para el paper, correr con los
hiperparámetros de búsqueda del baseline vale más que unas milésimas de MRR obtenidas propagando
el doble de aristas que ellos.

### Estado de la ruta sin padding (jobs 92593 / 92594, época 3 de 10)

| | mem | vel. | valid ép. 0 | 1 | 2 |
|---|---:|---:|---:|---:|---:|
| **92593** sin tope, s42 | **31.6 GB** | 1.20 it/s | 0.529 | **0.546** | 0.540 |
| **92594** sin tope, s43 | 31.5 GB | 1.19 it/s | 0.520 | 0.542 | — |
| *92537 cap 1024* | *35.5 GB* | *1.31 it/s* | *0.529* | *0.528* | *0.519* |

**Usa 3.9 GB MENOS que con cap 1024 procesando MÁS aristas** — es exactamente lo que debía pasar
al dejar de pagar padding. Cuesta ~9 % de velocidad (los dos `sort` estables del top-k por
segmento). La memoria estable tras miles de pasos despeja el riesgo de que `total` explotara.
⚠️ Las dos semillas van sobre el valid final de cap 1024 (0.554) ya en la época 1, pero **son
trayectorias incompletas**: no leer tendencia de acá (ya falló tres veces esta semana).

---

## 2026-08-30 — Implementado el top-k SIN PADDING de A\*Net. La `edge_cap` era un artefacto mío, y costó +0.098

### El barrido de `edge_cap` termina: 0.5478, a −0.015 de NBFNet

| brazo | MRR | H@1 | H@3 | H@10 | MR | cobertura | MRR\|alcanz |
|---|---:|---:|---:|---:|---:|---:|---:|
| *NBFNet (lit.)* | *0.563* | *0.480* | *0.612* | *0.708* | — | — | — |
| *A\*Net (lit.)* | *0.556* | *0.470* | *0.611* | *0.707* | — | — | — |
| **92537 cap 1024** | **0.5478** | **0.4626** | **0.5990** | **0.6990** | **1 184** | 98.3 % | 0.5571 |
| 92536 cap 512 | 0.5317 | 0.4461 | 0.5849 | 0.6858 | 1 399 | 98.3 % | 0.5410 |
| cap 256 (n=2) | 0.5147 ± 0.0054 | 0.4356 | 0.5650 | 0.6514 | 1 780 | 97.7 % | 0.5307 |

**Gap contra NBFNet: −0.015. Contra A\*Net: −0.008. H@10 casi empatado (0.699 vs 0.707).**
Desde el 0.3788 del 2026-08-26 son **+0.169, todos por arreglos de BÚSQUEDA**, ninguno de modelo.

**La predicción escrita antes de correr se cumplió**: predije 0.539 para cap 512 y 0.550 para
cap 1024; salió **0.5317** y **0.5478** (la segunda a 0.002).
⚠️ Pero **la duda que anoté a la época 4/10 estaba equivocada**: dije que cap 1024 no se
separaba de cap 512 y que la palanca saturaría. Se separó (+0.016). Era ruido de mitad de
entrenamiento — **tercera vez en la semana** que leo tendencia en una trayectoria incompleta.

### La linealidad se rompe, pero AL REVÉS de lo esperado

| tramo | Δ aristas retenidas | Δ MRR | **MRR por punto** |
|---|---:|---:|---:|
| 64→128 | +7.4 pp | +0.0335 | 0.0045 |
| 128→256 | +7.0 pp | +0.0314 | 0.0044 |
| 512→1024 | **+2.7 pp** | +0.0161 | **0.0061** |

El último tramo es **más eficiente**, no menos. Las aristas de los hubs extremos (grado > 512)
valen **más por arista** que las medianas. **No hay saturación visible.** Y entre cap 512 y 1024
la cobertura no se mueve (98.3 %) mientras `MRR|alcanzada` sube 0.5410 → 0.5571 ⇒ tercera
confirmación independiente de que esta palanca actúa por CALIDAD de evidencia, no por alcance.

### La causa raíz: `edge_cap` es un artefacto de MI implementación. A\*Net no lo tiene

`AStarNet/reasoning/data.py:276` — su `neighbors()` devuelve **TODAS** las salientes de los
nodos seleccionados (`num_neighbors = ends - starts`, rango completo). El único recorte es
**global**, vía `variadic_topks` con presupuesto `es = degree_ratio · ks · E/N`.

Nuestro tope existía sólo para poder armar el pool como matriz **rectangular** `(B, K·cap)`. Y
peor: `src/model.py` tomaba las **primeras `cap` aristas de la fila CSR**, o sea en el orden en
que venían en el archivo del dataset (`CSRGraph` hace `argsort(src)` y dentro de cada fila
queda el orden original). **Misma categoría exacta que el desalojo por id global del
2026-08-27**: truncar por un orden de índice sin relación con la relevancia.

### 🔴 El error de razonamiento, que estaba ESCRITO en mi propio resumen

`A_star_Net_resumen.md:343` decía, sobre el Alg. 2 de A\*Net:

> **No lo usamos**: nuestro top-L/top-M es de tamaño FIJO, que da la misma reducción de
> memoria a cambio de algo de cómputo en padding.

**Ese descarte es la causa raíz de toda la semana.** El error fue aplicar un hecho verdadero a
la parte equivocada del pipeline. Hay DOS lugares con padding y sólo uno es de tamaño fijo:

- **la SALIDA del top-L** — sí es fija (cada query propaga exactamente L aristas). La nota era
  correcta *para esto*, **y por eso sobrevivió a cada relectura**: la frase se verificaba;
- **el POOL DE ENTRADA** — para armar `(B, K·cap)` hay que fijar casillas por nodo ⇒ el padding
  **obliga** al tope. Eso no es cómputo desperdiciado: **descarta evidencia**.

Lo archivé como un costo de velocidad y era un costo de **resultado**: **+0.098 de MRR**
(cap 64 → 1024), más que el gap que nos separaba de NBFNet. Nota corregida (tachada, no
borrada) el 2026-08-30.
**Lección**: una nota que descarta algo por una razón *verdadera pero mal aplicada* es más
peligrosa que una falsa — la falsa se cae al releerla; ésta se confirma cada vez.

### Implementado: la ruta sin padding (`--edge_cap 0`)

Portado **del CÓDIGO**, no de la prosa (lección del 2026-08-26), a `src/sparse_state.py`:

- `multikey_argsort` — verbatim de `functional.py:4`. Ordena por la última clave primero;
  como cada `argsort` es **estable**, termina agrupado por muestra y ordenado por valor dentro.
- `variadic_topks` — verbatim de `functional.py:30`. Verificado a mano antes de conectarlo:
  segmentos de largo 3 y 5, top-2 → `[5,3]` y `[9,8]`. ✓
- `global_to_slot_flat` — **pieza nueva** que hacía falta: mapeo global→slot sobre un arreglo
  PLANO. Clave compuesta `q·(N+1) + nodo`, que queda globalmente ordenada ⇒ un solo
  `searchsorted` 1-D. Sigue sin materializar ningún `(B,N)`.

La ruta densa quedó intacta en `_select_dense` (reproduce lo ya medido y sirve al test).

⚠️ **El padding no desaparece: se muda.** La SALIDA sigue siendo `(B,L)` rectangular porque la
capa de atención la espera así. Lo que se elimina es el pool de ENTRADA: 12.6 M casillas/query
con cap 1024 contra ~1.06 M aristas reales (**11.8× de desperdicio**, peor caso medido).

### ✅ Test de equivalencia PASADO (job 92584, FB15k-237)

```
N=14541  grado maximo=7614
max|dif| = 1.490e-07   media|dif| = 3.250e-09   corr = 1.00000000
```

Ruido de punto flotante, del mismo orden que el `1.788e-07` que validó el estado disperso el
2026-08-26 (b). ⇒ **la ruta sin padding computa lo mismo que la densa** cuando se le da un
`cap` que no trunca. `--edge_cap 0` habilitado para entrenar.

⚠️ Lo que el test **no** dice: el MRR de 0.000099 es de pesos aleatorios (con 14 541 nodos el
azar da ~1/14541 = 0.000069) — lo que importa es que las dos ramas dan el MISMO número, no
cuál. Y la equivalencia se midió en **FB15k-237**, no en YAGO; la lógica no depende del
dataset, pero conviene declararlo.

### Detalle del test (planteo original)

Régimen donde **deben** coincidir: con `cap ≥ grado máximo` ningún nodo se trunca ⇒ la ruta
densa junta exactamente las mismas aristas que la variádica. Mismos pesos, mismo batch, se
compara la salida. **Si `max|dif|` no baja a ~1e-6, hay un bug y no se entrena.**
Es justo lo que faltó en `evict prio` y `evict keep`, que se lanzaron a ciegas y costaron dos
corridas. En cola: las 8 GPUs están tomadas por otros tres usuarios.

### Lanzados tras pasar el test

| job | qué | predicción escrita antes |
|---|---|---|
| **92593** | YAGO `--edge_cap 0`, s42 | **0.555 – 0.575** |
| **92594** | YAGO `--edge_cap 0`, s43 | idem (para n≥2) |
| **92595** | cap 1024, s44 | cierra n=3 en el mejor brazo con tope |

**Predicción**: sin tope se retiene el **100 %** de las aristas contra 96.2 % con cap 1024
(+3.8 pp). Al ritmo del último tramo medido (0.0061 MRR/pp) ⇒ **~0.571, por encima de NBFNet
(0.563)**; si la eficiencia por arista decae en vez de subir, ~0.555. Fuera de 0.555–0.575, el
modelo de «MRR lineal en aristas retenidas» está mal.

⚠️ **Riesgo nuevo a vigilar**: `total` ya no es fijo, depende de los grados de los nodos que la
búsqueda elija. En YAGO está acotado en ~1.06 M/query (peor caso), pero **en wikikg2 hay que
medirlo antes de asumir que entra** — A\*Net por eso trocea el batch en `select_edges`
(`model.py:277-282`).

### Pendiente

- Resultado del test de equivalencia (92584) y, si pasa, entrenar YAGO con `--edge_cap 0`.
- **n≥3 en cap 1024** (hoy n=1; σ del brazo medida en cap 128 = 0.0068).
- El batch 8 contra su 40 (§6.1) sigue abierto: la ruta sin padding libera memoria del pool y
  podría permitir subirlo.

---

## 2026-08-28 (b) — `edge_cap` es la palanca real, no las épocas: YAGO llega a 0.5185 (gap −0.045)

Sigue la entrada anterior. Corregidas **dos lecturas mías de esa misma entrada** y encontrado el
mecanismo cuantitativo del `edge_cap`.

### Resultados (test, YAGO3-10, 0.40 pasadas)

| brazo | MRR | H@1 | H@3 | H@10 | MR | cobertura | MRR\|alcanz |
|---|---:|---:|---:|---:|---:|---:|---:|
| *NBFNet (lit.)* | *0.563* | *0.480* | *0.612* | *0.708* | — | — | — |
| *A\*Net (lit.)* | *0.556* | *0.470* | *0.611* | *0.707* | — | — | — |
| **92501 cap 256** | **0.5185** | **0.4424** | **0.5663** | **0.6507** | **1 751** | 97.7 % | **0.5307** |
| 92456 cap 128 s42 | 0.4881 | 0.4212 | 0.5244 | 0.6065 | 1 919 | 97.8 % | 0.4989 |
| 92500 cap 128 s43 | 0.4785 | 0.4127 | 0.5149 | 0.5949 | 2 414 | 97.5 % | 0.4910 |
| cap 64 (S=N, n=2) | 0.4498 | 0.3840 | 0.4889 | 0.5635 | 3 015 | 96.1 % | 0.4680 |

**cap 128 (n=2): 0.4833 ± 0.0068.** Gap contra NBFNet: **−0.045** (era −0.184 el 2026-08-26).

### Prueba limpia de que `edge_cap` NO actúa por cobertura

Entre cap 128 y cap 256 la **cobertura es idéntica** (97.8 % vs 97.7 %) pero **`MRR|alcanzada`
salta 0.4989 → 0.5307**. No alcanza más respuestas: rankea mejor las que ya alcanzaba, porque
agranda el pool del que el top-L elige. Confirma H10 por una vía independiente de la del
2026-08-28 (a), que se apoyaba en el grado de la fuente.

### El mecanismo, cuantificado

Grado saliente en YAGO: media 17.5, mediana 10, **máximo 61 044**. El cap trunca a poquísimos
nodos pero esos pocos son hubs y concentran mucha arista:

| cap | nodos truncados | **aristas retenidas** | MRR medido |
|---:|---:|---:|---:|
| 64 | 2.76 % | 74.2 % | 0.4498 |
| 128 | 1.48 % | 81.5 % | 0.4833 |
| 256 | 0.63 % | **88.6 %** | **0.5185** |
| 512 | 0.19 % | 93.6 % | corriendo |
| 1024 | 0.02 % | 96.2 % | corriendo |

**El MRR es casi lineal en la fracción de aristas RETENIDAS**: +7.4 pp → +0.038, +7.1 pp → +0.030.

### DOS CORRECCIONES a la entrada 2026-08-28 (a)

**(1) «El mejor brazo no convergió, 0.4881 es un piso» — era una lectura de UNA semilla.**
La corrida a 20 épocas (92499) hace **plató desde la época 2**: 0.471 → 0.477 en ocho épocas.
Y la semilla 43 eligió su mejor checkpoint en la **época 1**. El ascenso monótono de la semilla
42 (0.419 → 0.487) fue en buena parte suerte. ⇒ **Las épocas NO son la palanca**; 92499
cancelado. Leer una tendencia de una sola trayectoria ruidosa es el mismo error de siempre.

**(2) La σ entre semillas es 2.7× la que declaré.** Con S=N medí 0.0025 y lo di como la σ del
brazo; con cap 128 es **0.0068** (0.4881 vs 0.4785). ⇒ n=1 no alcanza y **cap 256 hoy es n=1**.

### Predicción ESCRITA ANTES de correr (para que sea falsable)

Del modelo lineal en aristas retenidas: **cap 512 → ~0.539** (rango 0.53-0.55) y
**cap 1024 → ~0.550**. Si se cumplen, la palanca **se agota cerca de 0.55** y los −0.013 que
quedarían contra NBFNet **no salen de aquí** — habría que tocar el modelo.

**Lectura temprana a la época 4/10 — NO favorece del todo la predicción**: cap 512 y cap 1024
corren ~0.02 sobre cap 256 a épocas comparables, pero **cap 1024 no se separa de cap 512**
(valid 0.522 vs 0.523). Si se sostiene, la palanca **satura antes** de lo que dice el modelo
lineal y cap 1024 quedará corto. Las curvas son ruidosas (cap 256 osciló 0.463–0.519 dentro de
la misma corrida) ⇒ no darlo por refutado todavía.

### Lanzados

| job | qué | mem | predicción |
|---|---|---:|---:|
| **92536** | cap 512, s42 | 33.3 GB | ~0.539 |
| **92537** | cap 1024, s42 | 35.5 GB | ~0.550 |
| **92538** | cap 256, s43 | 32.2 GB | n=2 sobre el mejor |

Memoria estimada antes de lanzar: 33.4 y 35.8 GB. Salió 33.3 y 35.5 ⇒ ninguno dio OOM. El pool
de candidatas `(B, K·cap)` pesa poco: doblar el cap costó **+0.6 GB**, no el doble.

---

## 2026-08-28 — Se cierra el gap de YAGO a la mitad: cobertura 65 %→98 %, MRR 0.379→0.488

Confirmado el diagnóstico del 2026-08-27. Quitar el desalojo por id global y aflojar el
`edge_cap` mueven YAGO de **0.3788 a 0.4881** (+0.109). Faltan −0.075 contra NBFNet.

| brazo | MRR | H@1 | H@3 | H@10 | MR | cobertura | MRR\|alcanz | mejor ép. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *NBFNet (lit.)* | *0.563* | *0.480* | *0.612* | *0.708* | — | — | — | — |
| **92456 S=N + cap 128** | **0.4881** | **0.4212** | **0.5244** | **0.6065** | **1 919** | **97.8 %** | 0.4989 | **9/10** ⚠️ |
| 92463 S=N s43 | 0.4516 | 0.3847 | 0.4925 | 0.5648 | 3 180 | 96.0 % | 0.4705 | 7/10 |
| 92455 S=N s42 | 0.4480 | 0.3833 | 0.4853 | 0.5621 | 2 850 | 96.2 % | 0.4655 | 5/10 |
| 92439 S=36 000 | 0.3788 | 0.3338 | 0.4073 | 0.4565 | 31 238 | 65.4 % | 0.5788 | 7/10 |

**S=N (n=2): 0.4498 ± 0.0025.** El MR cae **16×** (31 238 → 1 919): el bloque de empate se
disolvió, que es exactamente lo que predecía el diagnóstico.

### DOS PREDICCIONES MÍAS QUE FALLARON — anotarlas

**(1) Proyecté 0.55–0.58 y salió 0.45–0.49.** El razonamiento fue: `MRR|alcanzada = 0.5788`,
luego con cobertura 100 % el MRR global tiende a 0.5788. **Falacia de selección**: ese 0.5788
estaba condicionado a un subconjunto sesgado hacia lo FÁCIL (grado mediano de la respuesta 87).
Al alcanzar el 32 % restante —periféricas, grado mediano 7-9— el condicional **bajó** a 0.4989.
⇒ **Un condicional sobre un subconjunto seleccionado por dificultad no se extrapola al total.**
La proyección correcta habría necesitado la calidad esperada *en las queries no alcanzadas*,
que por definición no se podía medir. Lo honesto era dar la cota como techo optimista y decirlo.

**(2) Llamé "ruido" al `edge_cap 128` con datos de la época 0.** Dije que su 0.465 estaba dentro
de la fluctuación entre épocas (~0.02). Es **+0.038 sobre S=N, 15× la σ entre semillas
(0.0025)**: la palanca más grande que queda. Comparé contra la varianza equivocada — la de
época a época dentro de una corrida, no la de semilla a semilla entre brazos.

### La cobertura ya está agotada como palanca

97.8 % alcanzado con `MRR|alcanzada` 0.4989 ⇒ llevarla a 100 % daría **+0.011**. Los −0.075 que
faltan contra NBFNet son **calidad de ranking**, no búsqueda. Cambia qué hay que atacar.

### Lo que queda medido sobre el desalojo (para wikikg2)

Con S=N no hay desalojo, pero **en wikikg2 S=N es imposible** (15.6 GB solo de estado). Dos
mecanismos probados y los dos fallan:

- **`--evict prio`** (92457, valid 0.073): compara la prioridad APRENDIDA de los incumbentes
  contra la CONSTANTE `base` de los nuevos. Como la prioridad es además el `gate` de la
  atención, el entrenamiento la empuja a ~0: base 0.574 contra mediana de incumbentes 0.131
  ⇒ en la capa 5 sobrevivía el **6 %** del estado acumulado. Cada capa borraba lo aprendido.
- **`--evict keep`** (92462, valid 0.013 y después NaN): incumbentes nunca desalojados, los
  nuevos llenan lo libre por orden de la heurística A\*. La **dinámica de admisión verificada
  es la correcta** (incumbentes 1→26→718→18 502→32 311→36 000, monótono), pero entrena peor
  que `prio` desde la época 0 ⇒ hay un error de correctitud en la rama, no solo inestabilidad.
  **No relanzar sin encontrarlo**; van dos intentos y el patrón del proyecto dice portar del
  CÓDIGO (`VirtualTensor` de A\*Net), no diseñarlo de cero.

### Bug de métrica encontrado (afecta a todo el proyecto)

`train.py:246` calculaba `ranks = sum((score >= answer) & ~filt) + 1`. Con scores **NaN**,
`NaN >= NaN` es False ⇒ suma 0 ⇒ **rank 1 para toda query ⇒ `valid_mrr` = 1.000 exacto**, y
Lightning lo guarda como "mejor checkpoint" porque selecciona por max. Pasó en 92462.
**Agregada guarda**: `FloatingPointError` si los scores no son finitos. Revisados todos los
logs del proyecto: 92462 es el único caso ⇒ **ningún resultado reportado está contaminado**.

### Lanzados (2026-08-28)

| job | qué | por qué |
|---|---|---|
| **92499** `cap128_e20` | 20 épocas (0.80 pasadas) | el mejor brazo NO convergió: mejor ckpt en la 9/10, valid monótono sin plató |
| **92500** `cap128_s43` | semilla 43, 10 ép | con n=1 el 0.4881 no es una medición |
| **92501** `cap256` | cap 256 sin subir `edge_budget` | 64→128 dio +0.038; ¿la palanca tiene más recorrido? ⚠️ cap128 usó 31.6/40 GB, puede dar OOM |

---

## 2026-08-27 — El gap de YAGO es COBERTURA, no calidad: el desalojo se hace por id global

**El modelo no es peor que los baselines en YAGO. Es mejor — en el 65 % de las queries donde
llega a puntuar la respuesta.** El 0.184 de brecha es enteramente búsqueda que no alcanza.

### La medición que lo cierra

La distribución de ranks de `gt_yago_fast_ranks.pt` (10 000 queries, test) es **bimodal**:

| percentil | p25 | p50 | p60 | **p70** | p90 |
|---|---:|---:|---:|---:|---:|
| rank | 1 | 34.5 | 674 | **87 753** | 93 687 |

Entre el p60 y el p70 el rank salta de 674 a 87 753. No hay cola: hay **dos poblaciones**.

| | fracción | MRR |
|---|---:|---:|
| respuesta **alcanzada** (rank < 30 000) | **65.4 %** | **0.5788** |
| respuesta **no alcanzada** | 34.6 % | 0.00001 |
| global | 100 % | 0.3788 |

`0.654 × 0.5788 = 0.3785` ⇒ reproduce el MRR global exacto. **MRR|alcanzada = 0.5788 está POR
ENCIMA de NBFNet (0.563) y A\*Net (0.556).** El techo si la cobertura fuera total es 0.5788.

| cobertura | 65 % | 80 % | 90 % | 95 % | 100 % |
|---|---:|---:|---:|---:|---:|
| MRR proyectado | 0.379 | 0.463 | 0.521 | **0.550** | **0.579** |

### La causa: `new_active = merged[:, :S]`

En `SparseStateGraphTransformer`, cuando el activo desborda los `node_slots` hay que tirar
nodos. `merged` viene **ordenado por id global**, así que el truncamiento se queda con los
**S ids MÁS CHICOS**. Es un criterio **arbitrario** — depende de cómo estén numeradas las
entidades, no de su relevancia.

Y el cupo estaba **saturado**: medido en el nodo GPU con la config exacta de 92439,

| S | K | desalojo | nodos con score | respuesta alcanzada | mem |
|---:|---:|---|---:|---:|---:|
| 36 000 | 12 318 | id | **29.2 %** (= 36000/N, tope) | 62.5 % | 2.2 GB |
| 36 000 | 12 318 | prio | 29.2 % | 50.0 % ⚠️ | 2.3 GB |
| **123 182 (=N)** | 12 318 | — | **56.3 %** | **100 %** | 3.3 GB |

⚠️ La fila `prio` es con **pesos aleatorios**: ahí la prioridad es ruido y el número no dice
nada. Se mide entrenado en el job 92457.

Con S=N la búsqueda alcanza ~69 000 nodos por sí sola ⇒ **el cupo de 36 000 truncaba casi la
mitad de lo que el modelo iba a visitar.**

### Por qué A\*Net no tiene este problema

`AStarNet/reasoning/model.py:322` — `graph.score` es un **VirtualTensor sobre los N nodos** que
solo **ACUMULA claves**: `graph.score[node_out] = ...` agrega, nunca desaloja. Su `node_ratio`
limita **quién se expande** (`select_edges` toma top-K de `score.keys`), **no quién recibe
score**. Nosotros conflacionamos las dos cosas en un único `node_slots`. Son ejes distintos:

- `top_nodes` = quién SE EXPANDE por capa → 10 % de N, igual que ellos ✓
- `node_slots` = quién GUARDA ESTADO y recibe score → debía ser N ✗

### Firma coherente con H4/H5 de FB15k-237

Grado mediano, alcanzadas vs no alcanzadas: **respuesta 87 vs 11**, **fuente 19 vs 70**. La
fuente inunda (hub) y la respuesta se muere de hambre (periférica) — los **mismos dos ejes**
del análisis de FB15k-237, pero acá en vez de rankear mal producen no-cobertura directa.

### Lo que NO era

- **No era el presupuesto de entrenamiento**: igualarlo (92439, 0.40 pasadas) compró +0.008.
- **No es el learning rate.** Su config usa lr 5e-3 con batch 40 y 10 000 pasos; nosotros
  1e-3 con batch 8 y 53 950 pasos. Desplazamiento total de Adam ≈ pasos × lr: **50 vs 54**.
  Ya estaba igualado. No gastar GPU en barrer lr.

### Tres brazos lanzados (2026-08-27)

| job | qué aísla | config | mem | h estimadas |
|---|---|---|---:|---:|
| **92455** `yago_slotsN` | **la causa**: único cambio vs 92439 es `node_slots` 36 000 → **N** | K 12 318, L 228 122, cap 64 | 23.1 GB | 11.3 |
| **92456** `yago_cap128` | la OTRA desviación (§6.2): `edge_cap` 64 → **128**, L en proporción | S=N | 31.6 GB | 14.6 |
| **92457** `yago_evictprio` | **la pregunta de wikikg2**: S=36 000 pero desalojando por **prioridad** | `--evict prio` | 13.3 GB | 5.9 |

**Predicción de 92455**: cobertura → ~100 %, MRR **0.55–0.58**.
**92457 es el que decide wikikg2**: con 2.5 M nodos S=N es imposible (15.6 GB solo de estado)
⇒ hay que desalojar sí o sí, y la pregunta es si desalojar bien alcanza. A los 10 min su loss
va en 0.607 contra 0.371/0.386 de los otros dos — mala señal temprana, pero es época 0.

**Cancelado 92450** (S=90 000): confundía `node_slots` con `top_nodes` (subía los dos a la vez).

### Cambio de código

`--evict {id,prio}` en `SparseStateGraphTransformer`, default `id` para no alterar lo ya
medido. `prio` se queda con los S de mayor prioridad (los nuevos entran con la prioridad del
estado cero) y re-ordena por id para conservar el invariante que necesita `global_to_slot`.

---

## 2026-08-26 (c) — 🚀 **8.9× de speedup BIT A BIT IDÉNTICO en YAGO: la capa nueva usaba indexación avanzada en el lookup relacional.** De 14× más lentos que A\*Net a 1.23×. Y el GT en YAGO queda **0.175 por debajo** de los baselines

### 1. 🔴 Referencias de YAGO3-10 (aportadas por el usuario) y nuestro número

| YAGO3-10 transductivo | MRR | H@1 | H@3 | H@10 |
|---|---:|---:|---:|---:|
| **NBFNet** | **0.563** | **0.480** | **0.612** | 0.708 |
| ULTRA | 0.557 | — | — | **0.71** |
| A\*Net | 0.556 | 0.470 | 0.611 | 0.707 |
| RotatE | 0.495 | 0.402 | 0.550 | 0.670 |
| **GT nuestro** (valid, ép. 6/10, presupuesto 4 %) | **0.388** | — | — | 0.460 |

⚠️ **Estamos 0.175 abajo en MRR y 0.25 en H@10.** En FB15k-237 éramos competitivos; en YAGO no.
Los tres baselines quedan **dentro de 0.007 entre sí** ⇒ YAGO está tan saturado como FB15k-237.

### 2. 🚀 EL HALLAZGO: `index_select` — 8.9×, bit a bit idéntico

`PrunedSparseAttentionLayer` hacía el lookup relacional con **indexación avanzada**
(`self.rel_bias.t()[rel]`, `self.rel_value.permute(1,0,2)[rel]`), violando la regla operacional
del **2026-08-09** que el propio proyecto midió con profiler.

| YAGO, presupuesto real (K=12 318, L=228 000, batch 8) | it/s | triples/s | época (43 K triples) |
|---|---:|---:|---:|
| indexación avanzada | 0.364 | 2.9 | 4.8 h |
| **`index_select`** | **3.255** | **26.0** | **~28-40 min** |
| *A\*Net (Tabla 10b)* | — | *32.1* | *20.8 min* |

⇒ **de 14× más lentos que A\*Net a 1.23×**, y el entrenamiento con protocolo igualado pasa de
**~48 h a ~6 h**. Memoria 12.9 GB.

**Por qué fue 8.9× y no el "hasta 2×" documentado**: YAGO tiene **solo 74 relaciones** y
propagamos 228 000 aristas × 8 queries = **1.8 M aristas por paso, TODAS cayendo sobre 74
filas**. El backward de la indexación avanzada acumula con atómicos ⇒ contención extrema. Es la
misma patología del expander (87 K aristas compartiendo `R_exp`, entrada 2026-08-09),
amplificada. **La regla vale MÁS cuanto menos relaciones tiene el grafo.**

⚠️ **Mi descomposición del 14× era COMPLETAMENTE ERRÓNEA.** Había escrito "~3× kernel fusionado
+ ~3.5× desperdicio de padding + ~1.3× trabajo de la atención" y era **una línea de indexación
mal escrita**. Además probé bajar `edge_cap` de 64 a 24 sobre esa teoría y salió **más lento**
(0.314 vs 0.364). **Quinta estimación de costo por inspección fallida del proyecto.**
⇒ **PERFILAR, NO ESTIMAR.** La regla ya estaba escrita tres veces; esta es la cuarta.

### 3. Corrección de protocolo: replicaba sus PASOS, no sus TRIPLES

La Tabla 9 del apéndice (agregada hoy a `A_star_Net_resumen.md`) dice para YAGO
**`batch size 40`** y **`#epoch 0.4`** ⇒ 10 790 pasos que ven **431 616 triples**.
Yo había puesto 10 000 pasos a **batch 8** = 80 000 triples ⇒ **5.4× menos entrenamiento**.
Corregido: 5395 pasos × 10 épocas × 8 = **431 600 triples = 0.40 épocas** ✓

**Estado de la fidelidad al protocolo de A\*Net en YAGO:**

| | paper | nuestro |
|---|---|---|
| α (node ratio) | 10 % | **10.0 %** ✓ |
| #epoch | 0.4 | **0.40** ✓ |
| capas / dim / lr / adv.temp / #neg | 6 / 32 / 5e-3 / 0.5 / 32 | ✓ |
| β (degree ratio) | **100 %** (todas las salientes) | ~100 % agregado **pero con `edge_cap 64` por nodo** ⚠️ |
| batch | 40 | **8** ⚠️ (a 40 la memoria daría ~66 GB) |
| precisión | fp32 | **fp32** (`--precision 16` NO se aplicó) |

### 4. Otro hallazgo del apéndice: corrobora nuestro resultado sobre la normalización

> *"PNA does not generalize well when degrees are **dynamically determined by the priority
> function**. Therefore, we **precompute the degree on the full graph**, and use them no matter
> how many nodes and edges are selected."*

⇒ **Encontraron el mismo fenómeno que medimos**: normalizar por lo seleccionado no funciona, hay
que usar la magnitud del grafo COMPLETO. Es exactamente la causa que propusimos para que
`--prune_attn sigmoid` (sin normalizar) le gane a `softmax` bajo poda. **Evidencia independiente,
de otro grupo, del mismo mecanismo — citable.**

Otros detalles del apéndice: en **wikikg2 usan 1 048 576 negativos** (no 32) y un indicador con
**PageRank personalizado** en vez del 1-hot — a esa escala la frontera casi toda en ceros no les
alcanzó. Es otro régimen de loss; anotado para cuando lleguemos.

**Decisión / EN CURSO**

- **92439**: GT estado disperso en YAGO con protocolo igualado (α 10 %, 0.40 épocas) y el fix de
  `index_select`. ~40 min/época ⇒ **~6.7 h**.
- ⚠️ **Los jobs 92364/92365 (FB15k-237) corren con la indexación LENTA.** Sus **resultados son
  válidos** (el cambio es bit a bit idéntico), solo tardaron de más. Van 19-20/20.
- **Levers pendientes sin tocar parámetros**: `--precision 16` (podría dar otro 1.5-2× en una
  carga limitada por ancho de banda; **cambia la numérica, hay que verificar que no degrade**) y
  **perfilar de verdad** con `torch.profiler` ahora que cayó el cuello obvio.
- ⚠️ **La pregunta abierta sigue siendo el 0.175 de gap en YAGO.** El speedup no lo toca: solo
  permite medirlo bien y rápido. Si con protocolo igualado sigue ~0.15 abajo, **el modelo no
  transfiere a escala** y eso decide si wikikg2 tiene sentido.
- Artefactos: `sbatch_gt_yago_fast.sh`, `bench_scale.py` (presupuesto real),
  `A_star_Net_resumen.md` (apéndices B-F), `logs/bench_opt_92438.log`.

---

## 2026-08-26 (b) — 🧱 **IMPLEMENTADO Y VERIFICADO el GT con ESTADO DISPERSO (`--model sparse_state`), la ruta real a wikikg2.** El muro no eran las aristas: era el estado `(B,N,d)`. En YAGO: **3.5× menos memoria, 1.6× más rápido, y batch 32 deja de dar OOM**

**Contexto**: el requisito duro del usuario es escalar a ogbl-wikikg2. El benchmark de escala
mostró que el diagnóstico anterior era incorrecto y obligó a rediseñar.

### 1. ⚠️ CORRECCIÓN: el cuello de escala es el ESTADO, no las aristas

Medido en YAGO3-10 (job 92413) con el GT podado de estado denso: **0.807 it/s, 21.8 GB, OOM a
batch 32**. Yendo de FB15k-237 a YAGO hay **4.1× de aristas pero 8.5× de nodos**, y el modelo se
puso **10.9× más lento** ⇒ **escala con N, no con E**.

⇒ Corrige lo que yo mismo había documentado (el scoring `O(B·E)` como límite). Extrapolado a
wikikg2 (N = 2.5 M, B=8, d=32, L=6): estado 15.6 GB + scoring 6.7 GB = **22 GB antes de la
atención**, ~19 días/semilla. **A\*Net no mantiene estado para los 2.5 M nodos**
(`model.py:318`, `graph.edge_mask(..., compact=True)` + `VirtualTensor`).

### 2. Diseño: top-M de tamaño FIJO sobre NODOS (no la maquinaria variádica)

Mismo truco que ya funcionó con aristas. `active (B,S)` de ids globales **ORDENADOS** +
`hidden (B,S,d)`. Con S=30 000 en wikikg2 son **30 MB** contra 15.6 GB.
Mapeo global→slot con **`searchsorted`**; **nunca** se materializa un `(B,N)` — es justo el
objeto que hace inviable wikikg2. Grafo en **CSR ordenado por fuente** (`src/sparse_state.py`)
para sacar salientes sin recorrer E. La capa de atención se reutiliza **sin cambios**: opera con
listas `(B,L)` y estados `(B,*,d)` vía gather/scatter, no distingue nodos globales de slots.

### 3. ✅ VERIFICACIÓN EN DOS NIVELES, antes de entrenar

**Primitivas contra fuerza bruta** (job 92414, FB15k-237 real): `out_edges` devuelve exactamente
las mismas 10 706 aristas que la máscara exhaustiva sobre las 544 K · `owner` correcto ·
`global_to_slot` ubica los activos y su máscara coincide con la pertenencia real.

**Equivalencia con el modelo denso** (jobs 92421 y 92425): con `S=N` y presupuestos ≥ grafo
completo, **`max|diff| = 1.788e-07` en los nodos activos** — precisión de punto flotante. La
diferencia restante vive **solo en los no alcanzados**, que es la semántica buscada. Re-verificado
después del fix de autograd.

⚠️ **Me costó TRES iteraciones armar bien el TEST, no arreglar el modelo.** Los dos modelos
tienen tres semánticas entrelazadas y hay que alinearlas de a una:
1. qué nodos **existen** (el denso los tiene todos; el disperso solo los activos)
2. cuáles pasan por el **FFN/residual** (el denso se lo aplica a los N desde la capa 0, así que
   un nodo que se activa en la capa 2 ya llega con deriva acumulada)
3. cuáles **aportan aristas** (el denso restringe a `visited`, que arranca en `{head}`)

Se agregaron `_all_active` (disperso) y `_all_visited` (denso) **solo para el test**. Sin
alinear (2) la corr era 0.902; sin alinear (3), 0.048; alineando ambas, **1.8e-07**.
**Regla: un test de equivalencia entre dos implementaciones exige enumerar TODAS las semánticas
que difieren, no solo la que uno cambió.**

### 4. 🐛 TRES BUGS ATRAPADOS POR LOS TESTS, no por corridas de 30 h

| bug | síntoma | causa |
|---|---|---|
| índice fuera de rango | `device-side assert` | `csr.degree[SENT]` con SENT=N; se indexaba **antes** de enmascarar |
| **`scatter_` con índices duplicados** | estado corrupto, silencioso | los slots descartados iban al **slot 0** y escribían CERO sobre un nodo real ⇒ **slot centinela** |
| operación in-place | `RuntimeError` de autograd | `out.gather()` + `out.scatter_()` sobre el mismo tensor ⇒ versión **fuera de lugar** con centinela |

⚠️ El segundo es la **tercera vez** en la semana que `scatter_` con índices duplicados rompe algo
(las otras: la frontera el 2026-08-25 c). **En este harness, `scatter_` con índices que pueden
repetirse está prohibido: usar `scatter_reduce_` o un slot centinela.**

### 5. Resultado del rediseño (YAGO3-10, batch 8, medido — job 92426)

| | estado denso | **estado disperso** |
|---|---:|---:|
| it/s | 0.807 | **1.306** (1.6×) |
| memoria | 21.8 GB | **6.29 GB** (3.5×) |
| batch 32 | **OOM** | **24.4 GB** ✅ |

⚠️ La memoria bajó 3.5× pero el tiempo solo 1.6× ⇒ **el cuello ya no es el estado sino el
cómputo por paso**. Extrapolado a wikikg2 daría del orden de **1-2 días/semilla**, mucho mejor
que 19 pero lejos de las ~4 h que estimé. **No extrapolar dos veces seguidas: medirlo ahí.**

**Decisión / EN CURSO**

- **LANZADO: GT estado disperso en YAGO3-10** (job 92427), primer salto de escala real del
  proyecto (123 182 entidades, 8.5× FB15k-237). 1.23 it/s, 6.77 GB. **~4-5 h.**
  Protocolo de A\*Net para YAGO (`batch_per_epoch: 1000`, 10 épocas, replicado con
  `--limit_train_batches 1000`); una época completa serían 28.7 h y **no es lo que ellos hacen**.
- En paralelo, los tres brazos de poda en FB15k-237: **92364** (softmax, 15/20, valid 0.394),
  **92365** (sigmoid, 16/20, **valid 0.406**), **92412** (sigmoid + `--rel_readout`, 1/20).
  `sigmoid` le gana a `softmax` en **todas** las validaciones ⇒ la hipótesis de que bajo poda la
  normalización esconde la evidencia descartada se sostiene.
- Referencias vigentes: GT sin poda **0.4165** · NBFNet **0.4147 ± 0.0010** · A\*Net **0.4084**.
- Artefactos: `src/sparse_state.py` (`CSRGraph`, `global_to_slot`),
  `src/model.py::SparseStateGraphTransformer`, flags `--node_slots --top_nodes --edge_budget
  --edge_cap`, `sbatch_gt_yago.sh`, `bench_scale.py`, `DISENO_GT_PODA.md` (adenda),
  `logs/test_csr_92414.log`, `logs/test_sparse_eq_924{21,25}.log`, `logs/bench_scale_924{13,26}.log`.

---

## 2026-08-26 — ❌ **Tres intentos fallidos de poda, y la causa era que implementé la selección de A\*Net MAL en tres puntos.** El código estaba en el repo desde el principio: `select_edges` son 35 líneas de tensor ops que **no usan `rspmm`**

**Contexto**: la poda v1 y v2 se quedaban en valid ~0.33-0.35 contra 0.424 del GT completo. Se
diagnosticó por eliminación (fallback, gate, bug de capa, frontera) sin dar con la causa. El
usuario pasó `A_star_Net_resumen.md` y recordó que el código de A\*Net está en `AStarNet/`.

### Lo que estaba mal, verificado contra `AStarNet/reasoning/model.py:258-292`

```python
score_in = score[node_in]
index = variadic_topks(score_in, num_nodes, ks=ks, ...)   # (1) top-K NODOS por su score
edge_index, node_out = graph.neighbors(node_in)           # (2) aristas SALIENTES de esos nodos
score_edge = score[node_out]                              # (3) ordenadas por score del DESTINO
index = variadic_topks(score_edge, num_edge, ks=e, ...)
```

| | v1 / v2 (mal) | A\*Net (correcto) |
|---|---|---|
| **Criterio de arista** | prioridad del nodo **FUENTE** | prioridad del nodo **DESTINO** |
| **Etapas** | solo top-L aristas | **top-K nodos** → top-L aristas entre sus salientes |
| **Prioridad** | `readout(h)` | `σ(f(h ⊗ g([h,q])))`, **condicionada a la query** |

**El criterio de arista es el error conceptual grande.** Ordenar por el DESTINO *es* la
heurística de A\*: estima la distancia **restante al objetivo**, así que evalúa **adónde se
llega**. Ordenar por la fuente no usa la heurística en absoluto — solo dice "expandí desde
nodos buenos", que es una poda arbitraria con pasos de más.

### 🚩 El dato que descarta el presupuesto como causa

**A\*Net usa 38 610 mensajes en FB15k-237 = 7.1 % de las aristas** (Tabla de eficiencia del
paper), **MENOS que el 10 % que usábamos nosotros**, y obtiene 0.411. Y su presupuesto de
aristas sale de `es = degree_ratio × K × E/N` = 1 × 1454 × 37.4 ≈ **54 400**, casi idéntico a
nuestros 55 877. ⇒ **El problema nunca fue cuánto se poda, sino el criterio.** Eso invalida el
eje "ratio" del barrido que se había lanzado (jobs 92357/92358, cancelados).

### ⚠️ Una diferencia que el resumen NO menciona y que corrige algo que hicimos

En su `indicator`, **TODOS los nodos reciben score inicial** (`self.score(0, query)`), no solo el
head ⇒ `score.keys` cubre el grafo entero y **A\*Net NO impone frontera conectada**: el top-K de
la capa 0 elige el head más K−1 nodos arbitrarios (con `break_tie: no` en FB15k-237).

⇒ La **frontera conectada que agregamos el 2026-08-25 (c) es una DESVIACIÓN NUESTRA**, no una
corrección de fidelidad. Probablemente benigna o mejor (evita gastar presupuesto en nodos de
estado cero, cuyos mensajes son nulos), pero **no citarla como "así lo hace A\*Net"**.

### Señal previa sobre la normalización

Bajo la selección vieja, `--prune_attn sigmoid` (agregación **sin normalizar**) dio valid
**0.368** en la época 0 contra ~0.34 del softmax. Hipótesis: bajo poda el segment-softmax
normaliza sobre las aristas **elegidas** y **esconde** que se descartó evidencia — un nodo con
100 entrantes de las que se eligen 5 queda "como si tuviera 5 vecinos". NBFNet y A\*Net **suman**
(⊕ = PNA en FB15k-237 transductivo) ⇒ descartar aristas reduce la suma y el modelo lo nota. Se
re-testea sobre la selección correcta.

### 🔴 LECCIÓN DE PROCESO — la más cara de la sesión

**El código de A\*Net estaba en el repo desde el primer día y no se leyó para esta parte.** Se
descartó mentalmente porque *"usa `rspmm`, no aplica a nuestro caso"* — pero **`rspmm` vive solo
en la agregación** (`GeneralizedRelationalConv.message_and_aggregate`); `select_edges` y la
función de prioridad son **tensor ops puros, portables tal cual**. Se implementó desde la idea
general de `resumen_papers_KGC.md`, que no incluye el detalle de §3.2.

**Costo: tres ciclos de diagnóstico (fallback → gate → frontera) y ~4 días de GPU.**

⇒ **REGLA: cuando se reimplemente un componente de un paper cuyo código tenemos, portar desde
el CÓDIGO, no desde la descripción. Y si hay una razón para no usar el código (un kernel, una
dependencia), verificar que esa razón aplique al COMPONENTE concreto y no al repo entero.**

**Decisión / EN CURSO**

- **v3 lanzada, dos brazos** con la selección corregida y un cambio cada uno: **92364**
  (`--prune_attn softmax`, control) y **92365** (`sigmoid`, sin normalizar). 4.04 y 3.73 GB.
- **Si la selección corregida sube de ~0.35, el criterio era la causa.** Si no, la poda con
  atención tiene un problema más profundo y **hay que replantear el plan de escala**.
- Flags nuevos: `--node_ratio` (primera etapa) y `--prune_attn {softmax,sigmoid}`.
- Estado sólido a hoy: bloqueador #1 cerrado (4 baselines, n=3, todos dentro del 1.1 % del
  paper) y **el GT sin poda le gana a NBFNet en las cinco métricas** (0.41827 vs
  0.4147 ± 0.0010), en **n=1**.
- Artefactos: `src/model.py::PrunedSparseGraphTransformer._priority` + selección en dos etapas,
  `sbatch_gt_pruned_v3.sh`, `diag_frontier.py`, `A_star_Net_resumen.md` (aportado por el usuario).

---

## 2026-08-25 (c) — 🔍 **La poda v1 FALLÓ (valid 0.35 vs 0.424) y la causa se aisló por eliminación: no era el fallback, ni el gate, ni un bug — era que la selección NO garantizaba conectividad con el head.** Implementada la frontera conectada (v2). Y **cerrado el bloqueador #1**: NBFNet FB15k-237 = 0.4147 ± 0.0010 (n=3)

### 1. 🎯 BLOQUEADOR #1 CERRADO — los cuatro brazos de baseline, con n=3

| brazo | **nuestro (n=3)** | *paper* | Δ |
|---|---:|---:|---:|
| **NBFNet FB15k-237** | **0.4147 ± 0.0010** | *0.415* | **−0.0003** |
| A\*Net FB15k-237 | 0.4084 ± 0.0014 | *0.411* | −0.0026 |
| NBFNet WN18RR | 0.5466 ± 0.0015 | *0.551* | −0.0044 |
| A\*Net WN18RR | 0.5431 ± 0.0013 | *0.549* | −0.0059 |

⚠️ La semilla 1025 de NBFNet FB15k-237 se recuperó de un checkpoint de la **época 8** (el job
murió por disco lleno en la ~13). Es defendible porque NBFNet satura temprano (la 1024 alcanzó
su mejor valid en la **época 4** y no mejoró en 16), pero **hay que declararlo**.

### 2. GT completo, 20 épocas — gana en las CINCO métricas

| FB15k-237 transductivo | MRR | H@1 | H@3 | H@10 | MR |
|---|---:|---:|---:|---:|---:|
| **GT (20 ép., n=1)** | **0.41827** | **0.3219** | **0.4615** | **0.6079** | **109.1** |
| NBFNet (n=3) | 0.4147 ± 0.0010 | 0.3210 | 0.4547 | 0.5982 | 115.9 |
| A\*Net (n=3) | 0.4084 ± 0.0014 | 0.3192 | 0.4483 | 0.5834 | 497.8 |

Δ sobre NBFNet: MRR **+0.0035**, H@3 +0.0068, H@10 **+0.0097**, MR **−6.7**.

⚠️ **Asimetría de convergencia que hay que reportar, no explotar**: el mejor checkpoint del GT
cae en la **época 18 de 20** con el valid aún subiendo (0.420 → 0.422 → 0.424); NBFNet satura en
la **época 4**. ⇒ **nuestro número es un piso y el suyo su techo, a igual presupuesto de épocas**
(que es el protocolo de ellos). Extender épocas solo para nosotros sería desviarse.
⚠️ **n=1 contra n=3.** El margen (+0.0035) es 3.6× la σ de NBFNet, pero **ésa es la varianza de
ELLOS**. Sin semillas propias no es publicable como victoria.

### 3. 🔍 La poda v1 falló — diagnóstico por eliminación, sin re-entrenar

**Síntoma**: valid plateau en **~0.35** (oscilando 0.32–0.356) contra **0.424** del GT completo.
La predicción falsable escrita en el sbatch ("como la poda de A\*Net es sin pérdida, el GT podado
debería quedar dentro del ruido") **queda REFUTADA**.

Tres sospechosos, todos medidos sobre el mismo checkpoint:

| sospechoso | medición | veredicto |
|---|---|---|
| El fallback de no-visitados compite con los scores reales | 93.7 % de nodos visitados · respuesta visitada **99.6 %** · mandar no-visitados al fondo da **Δ −0.0000** · `unvisited_score` aprendió a −3.74 | **descartado** |
| El gate `sigmoid(readout)` colapsa los mensajes | gate medio **0.955** en capas 1+ | **descartado** |
| Mi capa nueva tiene un bug | cargando los pesos del **GT completo entrenado** con `edge_ratio=1.0`: **max\|diff\| 8.8e-05, corr 1.000000** | **descartado** |

⚠️ **CORREGIDO 2026-08-26**: esta conclusión resultó **FALSA**. Arreglada la frontera (y el
bug de `scatter_` con duplicados que la rompía), la v2 midió **igual que la v1** (0.339/0.327/
0.331 vs 0.331/0.347/0.353) ⇒ **la conectividad NO era la causa**. La causa real era que la
selección estaba mal implementada en tres puntos respecto de `AStarNet/reasoning/model.py`
(criterio por fuente en vez de destino, una etapa en vez de dos, prioridad sin condicionar a la
query). Ver entrada 2026-08-26. Y peor: **A\*Net NO impone frontera conectada** — todos los
nodos reciben score inicial ⇒ la frontera es una desviación nuestra, no una corrección.
Lo que sigue se escribió antes de saberlo.

⇒ ~~**La causa es ESTRUCTURAL**~~: el top-L global por score de la fuente **no garantiza que las
aristas estén conectadas al head**. El dato que lo señala: con un selector malo nuestro modelo
colapsa a **MRR 0.0001**, mientras **A\*Net con selección ALEATORIA todavía da 0.378** — porque
ellos expanden una frontera desde el head **por construcción** (seleccionan nodos y luego aristas
en *su* vecindario). Nosotros le pedíamos al selector que además descubriera la conectividad.

⚠️ **Salvedad del tercer diagnóstico**: usar el readout del GT completo como función de
prioridad está **fuera de distribución** (nunca se entrenó para seleccionar), así que el 0.0001
mide sobre todo "selector malo", no "podar destruye el cómputo". No citarlo como lo segundo.

### 4. FIX implementado (v2, job 92307)

Solo son candidatas las aristas cuya **fuente ya fue alcanzada**; el top-L opera dentro de esa
frontera. Y un detalle que la habría anulado: cuando la frontera tiene **menos de L** aristas el
`topk` rellena con score −inf ⇒ **solo las aristas REALES amplían `visited`**. Marcar también
los destinos del relleno habría inflado la frontera hasta cubrir el grafo en 2-3 capas.

**Smoke test, con la dependencia del ratio que predice el mecanismo:**

| `edge_ratio` | sin frontera | **con frontera** |
|---|---:|---:|
| 0.05 | 0.048 | **0.196** (4×) |
| 0.20 | 0.176 | 0.179 |

Con presupuesto chico la conectividad es crítica; con presupuesto grande la frontera se cubre por
accidente. (1 época, 10 batches ⇒ ruidoso; lo que vale es el patrón, no los valores.)

**Decisión**

- **v2 lanzada (job 92307)**: 8.76 it/s, 3.82 GB. Si supera el plateau de 0.35 de la v1, la
  frontera era la causa; si no, hay algo más y **la poda queda como línea con problema abierto**.
- **Herramienta reutilizable**: `diag_equiv.py` verifica que cualquier reimplementación de una
  capa sea fiel cargándole los pesos de la original y comparando salidas. Es el diagnóstico que
  más rápido descartó la hipótesis equivocada; usarlo **antes** de entrenar, no después.
- Artefactos: `src/model.py::PrunedSparseGraphTransformer` (frontera), `diag_pruned.py`,
  `diag_gate.py`, `diag_equiv.py`, `diag_prune_inference.py`, `sbatch_gt_pruned_v2.sh`,
  `sbatch_gt_eval_final.sh`, `logs/diag_*.log`.

---

## 2026-08-25 (b) — 📌 **Kernels fusionados para atención en grafos: existen y están verificados, pero exigen torch ≥ 2.4.1 y estamos en 2.1.0 ⇒ DIFERIDO a después del envío.** Corroboración externa de que el backward por arista es un límite duro

**Contexto**: el usuario trajo sugerencias de dos asistentes externos sobre cómo acelerar la
parte de atención del GT, que es donde no podemos usar `rspmm`. Se verificaron con búsqueda web.

### Qué es real y qué no aplica

**REAL y verificado**: *"On Efficient Scaling of GNNs via IO-Aware Layers Implementations"*,
Yandex Research, **ICML 2026 Spotlight** (arXiv 2605.31500,
`github.com/yandex-research/On-Efficient-Scaling-Of-GNNs`). Kernels estilo FlashAttention-2 que
fusionan scoring + softmax + agregación en una pasada sobre CSR, **para Graph Transformer y
GATv2** — o sea exactamente nuestro caso, no el de `rspmm` (que es message passing por nodo).
Reportan **mediana 1.6× y hasta 3.9× en forward** de GT; variante Tensor Core hasta 7.3 %.
Soporta forward **y backward**.

| herramienta | requiere | tenemos |
|---|---|---|
| Kernels de Yandex | **torch ≥ 2.4.1** (hasta 2.10) | **2.1.0** ❌ |
| FlexAttention (`torch.nn.attention.flex_attention`) | torch ≥ 2.5 | no existe en 2.1 ❌ |
| Fused3S | — | **sin backward** ⇒ inútil para entrenar |
| `torch.compile` | 2.0+ | ✅ (inductor de 2.1 flojo para `scatter_reduce`) |

### Por qué actualizar torch NO es barato

- **Env `attention`**: usa **PL 1.9.1**, y `train.py` llama `validation_epoch_end` /
  `test_epoch_end`, **eliminados en PL 2.0** ⇒ subir torch arrastra subir PL ⇒ reescribir los
  hooks del LightningModule.
- **Env `astarnet`**: `rspmm` está compilado contra **torch 2.1 + CUDA 12.1 + GCC 11.4**
  (CUDA 12.1 no soporta GCC >12.2). Ese env produce **todos los baselines** y costó seis
  bloqueos hacerlo andar (2026-08-19 c).

### La aritmética del payoff

El README del repo dice que el autotuning **optimiza latencia de FORWARD**. En entrenamiento el
backward pesa ~2× el forward; si no mejora:

```
speedup total = 1 / (0.33/1.6 + 0.67) = 1.14×      30 h/semilla → 26 h
```

⇒ **4 h por semilla** a cambio de una migración que puede romper la cadena de baselines, con
seis semanas de plazo. **La velocidad no es el cuello de botella**: con la poda ya pasamos de
76 h a 30 h por semilla, y lo que bloquea el paper es n=1 en el brazo principal y que la corrida
de wikikg2 exista. Un 1.14× no mueve ninguna de las dos.

### ✅ Corroboración externa de un hallazgo propio — esto sí es material de paper

El límite que medimos el 2026-08-21 (c) —la atención no puede usar el kernel fusionado porque
sus pesos por arista requieren gradiente— **aparece también en el trabajo especializado más
reciente del área**: el backward del GT sigue siendo el punto débil (atómicas al formar la
transpuesta), y en varios grafos el speedup de backward es **<1×**. O sea **no es una limitación
de nuestra implementación: es estructural de la familia**, y hay literatura de 2026 que lo
sostiene. Citable.

**Decisión**

- **DIFERIDO a después del envío.** Si en algún momento se migra torch, los kernels de Yandex
  son la vía real; no re-proponerlos antes, y no tocar el env `astarnet` bajo ningún concepto
  mientras produzca baselines.
- **NO se probó `torch.compile`** (decisión del usuario): sería lo único sin riesgo de entorno,
  pero el inductor de 2.1 es débil para `scatter_reduce` y la expectativa era baja.
- ⚠️ **FlexAttention no es una opción en este entorno** — la sugerencia externa no verificó la
  versión de torch. Anotado para que no vuelva.

---

## 2026-08-25 — 🏆 **EL GT SUPERA A NBFNet EN TRANSDUCTIVO: test 0.41651 contra 0.4140, y el 91 % de su ventaja viene del estrato predicho.** Y implementado `--model sparse_pruned` (poda aprendida): **6.5× menos memoria y 2.5× más rápido**

**Contexto**: caída del cluster, gate del diseño, e implementación de la poda. Tres bloques.

### 0. ⚠️ CAÍDA POR DISCO LLENO — cómo se ve y qué hacer

`OSError: [Errno 28] No space left on device` en el **`/home` compartido**, ~04:00 del 24-ago.
**No fue nuestro**: el proyecto ocupa 12 G de 70 T y al revisar había 14 T libres — fue
transitorio y de otro usuario. Se llevó tres jobs:

| job | síntoma |
|---|---|
| 92064 (GT s42) | `FAILED` en la época 16, **sin traceback** (salida truncada) |
| 92147 seed 1025 | `RuntimeError: File model_epoch_14.pth cannot be opened` |
| 92147 seed 1026 | `OSError: [Errno 28] ... 'working_dir.tmp'` |

⚠️ **SLURM reportó el job de NBFNet como `COMPLETED`** porque el script bash terminó bien; las
semillas fallaron adentro. **`sacct` State no basta: hay que revisar los rc de cada semilla.**
⚠️ Un job que muere sin traceback y con la salida cortada a mitad de época es **sospechoso de
disco**, no de bug del modelo. Revisar `df -h` y buscar `Errno 28` en los otros logs.

**Todo se recuperó sin re-entrenar**: el ckpt del GT (época 15, valid 0.422) estaba intacto y se
copió a `experiments/gt_s42_ep15_SAFE.ckpt` ANTES de relanzar (el resume de PL con `save_top_k=1`
lo habría borrado). NBFNet 1025 se recuperó evaluando su checkpoint de la época 8.

### 1. 🏆 GATE SUPERADO — el GT gana en FB15k-237 transductivo

| FB15k-237 transductivo, BCE | MRR | H@1 | H@3 | H@10 | MR |
|---|---:|---:|---:|---:|---:|
| **GT (ép. 15/20, n=1, `--dependent --remove_one_hop`)** | **0.41651** | 0.3195 | **0.4605** | **0.6061** | **109.3** |
| NBFNet s1024 | 0.41402 | 0.3186 | 0.4545 | 0.6007 | 113.9 |
| NBFNet s1025 (recuperada, ép. 8) | 0.41430 | 0.3218 | 0.4533 | 0.5955 | 113.5 |
| A\*Net (n=3) | 0.4084 | 0.3192 | 0.4483 | 0.5834 | 497.8 |

**Primer número transductivo del proyecto por encima de un NBFNet propio**: +0.0023 en MRR, y
también mejor en H@3, H@10 y **MR**. Empata en H@1. Y **le faltaban 5 épocas**.

Los baselines son extremadamente estables: las dos semillas de NBFNet difieren en **0.0003**;
A\*Net tiene sd 0.0014 ⇒ **todo el ruido está de nuestro lado.**

### 2. El perfil estratificado se limpia al entrenar — y confirma el mecanismo

| estrato (grado fuente) | A\*Net | GT ép.5 | **GT ép.15** | Δ ép.15 |
|---|---:|---:|---:|---:|
| 0-14 | 0.5412 | 0.5244 | 0.5394 | −0.0018 |
| 15-27 | 0.4956 | 0.4811 | 0.4974 | +0.0018 |
| 28-43 | 0.4896 | 0.4734 | 0.4913 | +0.0017 |
| 44-99 | 0.4443 | 0.4197 | 0.4417 | −0.0026 |
| 100-249 | 0.2946 | 0.2833 | 0.2991 | +0.0046 |
| **250+** | 0.1977 | 0.2126 | **0.2283** | **+0.0306** |

En la época 5 perdía de −0.011 a −0.024 en cinco estratos. **Entrenar cerró todo eso a ±0.003 y
la ventaja en el estrato de inundación CRECIÓ de +0.015 a +0.031.** Descomposición:

```
250+          : 0.199 × (+0.0306) = +0.0061   ← 91 % de la ventaja
otros 5 juntos                    = +0.0004
```

⇒ **El 91 % de la ventaja del GT sobre A\*Net viene del estrato que la hipótesis predijo.**
Captura el 23 % del techo disponible en ese régimen (+0.0265).

⚠️ **n=1.** El margen sobre NBFNet (+0.0023) es menor que la varianza histórica de la familia
sparse (σ 0.019–0.054). **No es publicable así.** Lo que sí es fuerte no es el agregado sino
**el patrón por estrato**, que es una predicción cumplida, no un número suelto.
⚠️ Semillas 43/44 canceladas por el usuario (checkpoints en épocas 8 y 5). Decisión razonada:
si el modelo definitivo es GT+poda, las semillas van ahí.

### 3. 🔧 IMPLEMENTADO `--model sparse_pruned --edge_ratio R`

**Cambio de plan respecto a `DISENO_GT_PODA.md`**: el atajo de reusar el subgrafo de A\*Net **no
sirve**, por dos razones que aparecieron al implementarlo:
1. el env `astarnet` **no tiene `pytorch_lightning`** ⇒ `train.py` no corre ahí, y meter PL ahí
   (o torchdrug en `attention`) arriesga romper un setup que costó seis bloqueos;
2. en FB15k-237 A\*Net poda al **42 % de los nodos por iteración** ⇒ la unión de 6 iteraciones es
   buena parte del grafo y **no validaría el argumento de memoria**. El 0.2 % es de wikikg2.

⇒ **Selección nativa**, y resultó más simple de lo estimado: **no hace falta el top-k sin padding
del Apéndice C de A\*Net** (existe porque ellos empaquetan subgrafos de tamaño variable). Con
**top-L de tamaño FIJO por query** es `topk` + `gather`, y da la misma reducción: `O(B·L·d)`.

Piezas: scoring de nodos con **weight sharing** contra el readout · selección **por capa** (la
frontera crece; estática no serviría) · gate por mensaje · **score de fallback aprendido** para
no visitados (apunta al MR malo de la poda: 4.4× FB237, 9.0× WN18RR).

**⚠️ DOS BUGS ATRAPADOS ANTES DE GASTAR GPU — anotados porque son fáciles de reintroducir:**

1. **Señal que se desvanece.** Portar `layer_input = sigmoid(score) * hidden`
   (`AStarNet/reasoning/model.py:319`) **literalmente** escala el stream residual por ~0.5 en
   cada capa ⇒ **0.5⁶ ≈ 0.016 después de seis**. A\*Net no lo sufre porque tiene `short_cut`;
   nuestro residual pre-LN no lo restaura. **La semántica correcta de su Ec. 12 es que el score
   pesa el MENSAJE, no el estado persistente.** En el smoke test el fix llevó `edge_ratio=0.2`
   de test_mrr **0.050 → 0.176**.
2. **`visited` calculado solo con la última capa** ⇒ nodos alcanzados en la capa 3 pero no en la
   6 iban al fallback pese a tener estado válido. Ahora se acumula sobre todas.

**Costo medido al arrancar (FB15k-237 transductivo, `edge_ratio 0.1`):**

| | GT completo | **GT podado** |
|---|---:|---:|
| memoria | 24.4 GB | **3.78 GB** (6.5×) |
| velocidad | 2.9 it/s | **7.3 it/s** (2.5×) |
| h/época | 3.70 | **~1.5** |
| 20 épocas | ~76 h | **~30 h** |

⇒ **La poda no es solo la ruta a wikikg2: hace toda esta línea 2.5× más barata**, lo que cambia
cuántas semillas caben en el plazo.

⚠️ **Límite conocido para wikikg2**: el scoring construye `(B, E)` para el topk. Con E = 32.2 M y
B = 32 son ~4 GB solo en esa tabla. Para FB15k-237 (E = 544 K ⇒ 70 MB) no molesta. A escala hay
que puntuar solo la **frontera**. Documentado en el código; fuera de esta versión a propósito.

**Decisión / EN CURSO**

- **`DISENO_GT_PODA.md` pasa su gate**: el GT es competitivo (de hecho superior) ⇒ se construye.
- **V1 lanzado (job 92253)**, con predicción escrita: como la poda de A\*Net es sin pérdida en
  inferencia, el GT podado debería quedar **dentro del ruido** del GT completo (0.41651). Si cae
  mucho, la poda **no transfiere** a una agregación por atención — resultado en sí mismo.
- En curso: **92253** GT podado (~30 h) · **92223** GT completo, resume desde la época 16 ·
  **92225** NBFNet semilla 1026 (~18 h).
- Volcado de ranks de NBFNet listo (`nbfnet_ranks_fb237_s1024_test.pkl`) para estratificar el GT
  contra NBFNet, no solo contra A\*Net.
- Artefactos: `src/model.py::{PrunedSparseAttentionLayer,PrunedSparseGraphTransformer}`,
  `train.py::--edge_ratio`, `sbatch_gt_pruned_v1.sh`, `sbatch_gt_eval_ep15.sh`,
  `recover_nbfnet_1025.sh`, `run_dump_nbfnet_ranks.sh`, `gt_ranks_fb237_s42_ep15.pt`,
  `experiments/gt_s42_ep15_SAFE.ckpt`.

---

## 2026-08-24 — 📐 **Análisis cruzado fuente×respuesta: los dos ejes del fallo son SEPARABLES y se componen (peor celda MRR 0.154). El techo de neutralizar la inundación es +0.027 ⇒ ~0.435 contra 0.414 de NBFNet.** Y escrito el diseño `DISENO_GT_PODA.md` (GT sobre subgrafo podado por A\*Net)

**Contexto**: pregunta del usuario — sabiendo dónde falla A\*Net, ¿se puede acertar ahí y
mantener donde funciona? Y después: si el modelo definitivo va a ser GT+A\* por el requisito de
escala, ¿no conviene diseñarlo ya? Dos análisis en CPU sobre volcados existentes y una
especificación. **Sin GPU nueva consumida.**

### 1. Dentro de la cola larga, qué separa acierto de fallo

Estrato respuesta de grado ≤27 (12 597 queries, MRR 0.2179, H@10 0.3610), mediana de cada
feature según si acierta (rank ≤10) o falla:

| | acierta | falla | ratio |
|---|---:|---:|---:|
| vecinos comunes (caminos de largo 2) | 2 | 1 | 2.00 |
| **grado de la FUENTE** | **42** | **83** | **0.51** |
| grado de la respuesta | 17 | 18 | 0.94 |
| frecuencia de la relación | 1908 | 2727 | 0.70 |

H@10 dentro de la cola por vecinos comunes: **0 → 0.272 · 1 → 0.291 · 2-3 → 0.402 ·
4-9 → 0.527 · 10+ → 0.696**.

Dos lecturas: (i) la cantidad de evidencia predice fuerte incluso **dentro** de la cola;
(ii) pero aun con **cero** caminos de largo 2 el modelo acierta el 27 % ⇒ hay señal más allá
de los 2 saltos, la cola no es ruido puro.

### 2. Cruce fuente × respuesta — **los dos ejes NO son el mismo problema**

MRR por celda (n entre paréntesis):

| fuente \ resp | 0-27 | 28-99 | 100-249 | 250+ |
|---|---:|---:|---:|---:|
| **0-27** | 0.266 (3684) | 0.373 (3658) | 0.595 (1395) | **0.849** (3860) |
| 28-99 | 0.240 (3658) | 0.369 (6122) | 0.558 (2232) | 0.768 (4079) |
| 100-249 | 0.210 (1395) | 0.272 (2232) | 0.585 (284) | 0.805 (170) |
| **250+** | **0.154** (3860) | 0.216 (4079) | 0.613 (170) | 0.644 (54) |

⚠️ **CORRIGE una afirmación de la sesión**: se dijo que "fuente hub → respuesta periférica es
la misma query, un solo problema". **Falso**: los dos efectos son **separables y se componen**.
- Por FILA (respuesta más conectada): 0.266 → 0.849, factor **3.2×**
- Por COLUMNA (fuente más hub): 0.266 → 0.154, factor **1.7×**

⇒ **El eje de la RESPUESTA es más fuerte, pero es el limitado por información. El eje de la
FUENTE es más débil y es el atacable** — concentra el **27 % del déficit total** (celdas con
fuente ≥250: 13.5 % + 13.2 %).

### 3. Techo de atacar la inundación, y corrección de una cuenta anterior

Si las filas de fuente ≥250 rindieran como las de fuente 0-27:

```
(250+, 0-27)   3860 × (0.266−0.154) = +432
(250+, 28-99)  4079 × (0.373−0.216) = +640
(250+, 250+)     54 × (0.849−0.644) = +11
                       +1083 / 40932 = +0.0265
A*Net 0.4084 → ~0.435        NBFNet 0.4140
```

⇒ **Eso sería ganar, no empatar.**

⚠️ **Corrige la cuenta pesimista dada horas antes en la misma sesión** (+0.0048 ⇒ ~0.413,
"paridad"). Aquélla tomaba la ventaja del GT **actual** (subentrenado, época 5, n=1) como si
fuera el techo del mecanismo; es apenas el **18 %** de lo disponible. La cuenta correcta mide
el techo del **mecanismo**, no el de una implementación parcial. La cifra vigente es **+0.027**.

**Objetivo de diseño que queda preciso**: no "un GT mejor", sino **capturar la ganancia de la
inundación sin pagar la pérdida en los otros cinco estratos**. Hoy el GT captura ~18 % y pierde
más que eso en el resto ⇒ agregado negativo.

### 4. Diseño escrito: `DISENO_GT_PODA.md`

Especificación de **GT disperso sobre el subgrafo podado por A\*Net** (161 líneas, nada
implementado). Tres piezas: selector A\*Net **congelado** → GT sobre el subgrafo → **fallback
para nodos no visitados**. Dos etapas a propósito (se pierde el weight sharing) para que quepa
en el plazo.

Tres cosas que aparecieron al escribirlo y que no estaban en la discusión:

- ⚠️ **Los negativos fuera del subgrafo pueden degenerar la loss.** Con `num_negative 32` sobre
  14 541 entidades la mayoría cae fuera del subgrafo y recibe el mismo score de fallback ⇒ la
  BCE deja de discriminar. **Medir la fracción antes de entrenar**; si es alta, muestrear
  negativos DENTRO del subgrafo.
- ✅ **El fallback es una oportunidad, no un detalle**: es la causa del MR malo de A\*Net
  (4.4× FB15k-237, 9.0× WN18RR). Un fallback sensato **conserva la eficiencia y arregla el MR**
  — resultado chico pero concreto y salido de medición propia.
- ⚠️ **El selector podría no transferir**: su prioridad se entrenó con weight sharing contra
  *su* predictor. Que la poda sea sin pérdida para A\*Net no garantiza que lo sea para otra
  agregación. Es la predicción falsable de V1.

**Decisión**

- **GATE, hoy**: si el GT completo (job 92064) no es competitivo con NBFNet (**0.4140**) ni
  muestra ventaja por estrato, **`DISENO_GT_PODA.md` NO se construye** — sería A\*Net con pasos
  de más y peores números.
- **Secuencia acordada**: poda **sola** primero (V1: GT-sobre-subgrafo vs GT-completo en
  FB15k-237), y la **compuerta atención/suma condicionada al grado DESPUÉS**, como cambio
  separado. Juntas harían imposible atribuir un fallo.
- ⚠️ **El eje de la respuesta (el más fuerte) no se ataca con arquitectura** en este plazo. La
  vía sería traer información de otro grafo (Etapa 2 de la tesis, ULTRA). Queda fuera del paper.
- Estado de jobs: **92064** GT semilla 42 en época 15/20 (valid 0.420), termina ~18:30 de hoy ·
  **92147** NBFNet 1 de 3 semillas completa (test **0.4140**), termina ~25-ago 06:00.
  **Semillas 43/44 del GT canceladas por el usuario** (checkpoints en épocas 8 y 5, reanudables)
  ⇒ el lado GT queda en **n=1** y la varianza de semilla del estrato 250+ no se va a poder medir.
- Artefactos: `DISENO_GT_PODA.md`, `astarnet_ranks_bias_s102{4,5,6}_test.pkl`,
  `logs/dump_astar_seeds_92188.log`.

---

## 2026-08-23 — 🌱 **PRIMERA SEÑAL A FAVOR DE LA ATENCIÓN EN 2 MESES, y con predicción escrita ANTES de medir: el GT le gana a A\*Net SOLO en el estrato de fuentes de grado ≥250 (+0.0149 ± 0.0052, t=2.85) y pierde en los otros cinco.** ⚠️ Todavía NO es una medición: n=1 y el GT está en la época 5 de 20

**Contexto**: la entrada 2026-08-22 (b) dejó como único eje vivo la **evaluación estratificada
por grado**, con esta hipótesis derivada del modo de fallo medido (fuente hub → respuesta
long-tail):

> *La atención ayuda donde la FUENTE tiene grado alto —donde inunda la propagación y la
> agregación fija diluye— y no aporta donde el vecindario es chico.*

Se escribió **antes** de medir. Las cinco predicciones previas sobre atención en este proyecto
(opción A sigmoid/degree, opción C anchor, `--attn rel`, `--rel_param lowrank`, expander)
fallaron todas.

### Infraestructura nueva: `--dump_ranks` en el harness propio

`src/metric.py` acumula **sumas** (`rank_sum`, `total`), no ranks por query ⇒ era imposible
estratificar por ninguna covariable. Agregado `--dump_ranks <path>` en `train.py`: guarda
`(h, r, t, rank)` de cada query de test más el `edge_index` del grafo.

**Verificado**: el MRR reconstruido desde los ranks volcados coincide **exactamente** con el
`test_mrr` del harness (0.0996570736 vs 0.09965697675) ⇒ el volcado es fiel a la métrica.

⚠️ **TRAMPA QUE COSTÓ UN JOB**: la primera pasada escribió a `/tmp` y **el archivo desapareció**
— `/tmp` NO se comparte con el nodo de cómputo (ya estaba en `CLAUDE.md`). El dump imprimió su
mensaje y `rc=0`, así que **parecía exitoso**. **`--dump_ranks` debe apuntar SIEMPRE a `$HOME`.**

**Comparabilidad verificada antes de medir**: `src/data.py::encode_triplets` agrega ambas
direcciones ⇒ nuestro harness evalúa **40 932 queries sobre 20 466 hechos de test**, idéntico a
A\*Net. Y los tamaños de estrato coinciden uno a uno entre ambos volcados (4583 / 8014 / 8041 /
8050 / 4081 / 8163) ⇒ mismos grafos y mismas queries, pese a indexaciones de entidad distintas.

### Resultado — GT (parcial, época 5/20, semilla 42) vs A\*Net (final, semilla 1024)

Agregado: A\*Net **0.4100** vs GT **0.3991**. ⚠️ El GT está subentrenado ⇒ su agregado es un
PISO y **no es citable**. Dato lateral: aun así el GT tiene **H@10 más alto** (0.5865 vs 0.5833)
con MRR más bajo — reparte la probabilidad distinto.

**Por grado de la FUENTE** (cortes fijos, no cuantiles, para que los estratos sean idénticos):

| estrato | n | A\*Net | GT | **Δ** | SE | **t** |
|---|---:|---:|---:|---:|---:|---:|
| 0-14 | 4583 | 0.5412 | 0.5244 | −0.0167 | 0.0092 | −1.82 |
| 15-27 | 8014 | 0.4956 | 0.4811 | −0.0145 | 0.0068 | −2.14 |
| 28-43 | 8041 | 0.4896 | 0.4734 | −0.0162 | 0.0067 | −2.41 |
| 44-99 | 8050 | 0.4443 | 0.4197 | −0.0246 | 0.0067 | −3.68 |
| 100-249 | 4081 | 0.2946 | 0.2833 | −0.0113 | 0.0085 | −1.32 |
| **250+** | **8163** | 0.1977 | **0.2126** | **+0.0149** | 0.0052 | **+2.85** |

**Pierde en cinco estratos y gana en uno: exactamente el que la predicción señalaba.** Y es el
estrato donde ambos son peores (MRR ~0.20), o sea la parte difícil del problema.

**Por grado de la RESPUESTA**: el GT es peor o igual en todos los estratos (Δ de −0.039 a
+0.003) ⇒ **el cruce está en el grado de la FUENTE, no en el de la respuesta.** Importa para no
confundir los dos ejes del perfil de fallo.

**Análisis**

- **(A) ⚠️ ESTO NO ES UNA MEDICIÓN TODAVÍA. Tres razones, en orden de gravedad:**
  1. **El SE es solo ruido de muestreo de queries DENTRO de una corrida; no incluye varianza de
     semilla.** El protocolo del proyecto existe por esto: el sparse llegó a σ = 0.019–0.054 en
     agregado. Un t=2.85 sobre muestreo de queries puede quedar tapado por la varianza entre
     semillas. **No citar el t=2.85 como significancia del efecto.**
  2. **El GT está en la época 5 de 20.** Si entrenar lo sube de forma uniforme, el cruce se
     mueve: podría agrandar la ventaja o hacerla desaparecer.
  3. **n=1 semilla en los dos lados.**
- **(B) Por qué igual vale la pena**: es la primera predicción sobre atención escrita antes de
  medir que se cumple, y el patrón es **cualitativo, no de nivel** — el signo se invierte en un
  estrato concreto, no es un desplazamiento uniforme. Un GT simplemente subentrenado daría
  negativo en los seis.
- **(C) Si replica, la arquitectura que la realiza es un HÍBRIDO CON COMPUERTA**, y el proyecto
  ya sabe cuál es el error a evitar: la opción C (`--attn anchor`, 2026-07-22) interpoló contra
  **mean-pool** (`1/grado_in`) y dio 0.278, el peor resultado del proyecto. Su propia entrada
  diagnostica que el ancla estaba mal elegida y deja escrito que *"haría falta una agregación
  que SUME y conserve conteo/grado (estilo suma/PNA)"* ⇒ **la versión correcta nunca se probó**.
  Con la compuerta **condicionada al grado de la fuente** deja de ser un hiperparámetro
  arbitrario y pasa a ser un mecanismo derivado de datos propios.

**Decisión / EN CURSO**

- **Reanudadas las semillas 43 y 44 del GT** (jobs 92151/92152, `RESUME=1`), desde sus
  checkpoints de la **época 3** — no se reempezó de cero. Verificado en el log:
  `Restoring states` → `Restored all states`.
- **PENDIENTE Y BLOQUEADO POR QOS**: volcar las semillas **1025 y 1026 de A\*Net** (ya
  entrenadas, ~5 min c/u) para tener varianza de semilla del otro lado. `sbatch` devolvió
  **`QOSMaxSubmitJobPerUserLimit`**: la QOS `external` topa en **4 jobs** y ni siquiera deja
  encolar el quinto. Script listo: `run_dump_astar_seeds.sh`.
  ⚠️ **Se descartó a propósito correrlo con `srun --overlap`** (el patrón de la bitácora para no
  gastar slot): 92064 usa **24.4 GB de 40** y el eval de A\*Net pide **~14 GB** ⇒ 38.4/40 es
  demasiado justo, y un OOM tumbaría una corrida de 27 h para ahorrar 5 min.
- **Los 4 slots están tomados y todo lo que falta depende de esos resultados** ⇒ no hay nada más
  que lanzar hasta que termine alguno.

### Estado de los jobs al cierre (2026-08-23)

| job | qué | estado |
|---|---|---|
| **92064** | GT `--loss bce --dependent --remove_one_hop`, semilla 42 | época 8/20 (~27 h) |
| **92147** | **NBFNet FB15k-237**, 3 semillas en serie (bloqueador #1) | ~1.5 h de ~55 h |
| **92151** | GT semilla 43, reanudada en época 3 | recién lanzada |
| **92152** | GT semilla 44, reanudada en época 3 | recién lanzada |

### Qué hacer cuando terminen (en orden)

1. Volcar las semillas 1025/1026 de A\*Net (`run_dump_astar_seeds.sh`) apenas se libere un slot.
2. Volcar los ranks de las 3 semillas del GT ya entrenadas (`sbatch_dump_gt_ranks.sh`, apuntando
   al best ckpt de cada una) y de NBFNet.
3. Re-correr `compare_strata.py` con **n=3 por lado** ⇒ ahí sí se decide si el +0.0149 sobrevive.
4. Si sobrevive: híbrido con compuerta condicionada al grado de la fuente, anclado a **suma/PNA**
   (no mean-pool), y recién después la poda A\* para escalar.
   Si no sobrevive: no hay reclamo arquitectónico y el paper es el de mecanismo.

- Artefactos: `train.py::{--dump_ranks,_save_ranks}`, `compare_strata.py`,
  `sbatch_dump_gt_ranks.sh`, `run_dump_astar_seeds.sh`, `gt_ranks_fb237_s42_partial.pt`,
  `logs/gt_dump_ranks_92150.log`, `logs/smoke_dump_921{48,49}.log`.

---

## 2026-08-22 (b) — 🔬 **OPCIÓN B REFUTADA en sus DOS mecanismos, con predicción escrita antes de medir. Y caracterización del modo de fallo de A\*Net: NO es alcance (99.7 % de los fallos están dentro del horizonte), es ESCASEZ DE EVIDENCIA en la cola larga — sin sesgo de popularidad**

**Contexto**: se arrancó la opción B (reranking sobre A\*Net, entrada 2026-08-22). Cuatro
mediciones, todas sobre el mejor checkpoint de la semilla 1024 (época 18, test 0.4100), sin
entrenar nada. Las dos primeras son gates con predicción falsable escrita **antes** de ver los
números; las dos últimas responden la pregunta del usuario sobre qué comparten los fallos.

### 1. GATE de cardinalidad — la exclusión mutua queda REFUTADA, y al revés

Techo oráculo de un reranker **perfecto** sobre top-K (si el rank verdadero es ≤ K va a 1; si
no, no se puede tocar), desglosado por cardinalidad de relación (convención TransE, umbral 1.5):

| cardinalidad | queries | MRR | **margen@10** | margen@100 |
|---|---:|---:|---:|---:|
| 1-a-1 | 0.9 % | 0.5641 | **+0.0791** | +0.2386 |
| N-a-1 | 22.0 % | 0.4715 | +0.1076 | +0.2563 |
| 1-a-N | 6.3 % | 0.3065 | +0.1549 | +0.3598 |
| **N-a-N** | **70.7 %** | 0.3980 | **+0.2096** | +0.4274 |

La hipótesis predecía margen **grande donde el conjunto respuesta es chico**. Salió **al revés**:
+0.079 en 1-a-1 contra +0.210 en N-a-N. Causa mecánica: el margen es `H@K − MRR`, o sea mide
**dónde el modelo está flojo**, no dónde la restricción de conjunto es informativa.

**Cota dura del mecanismo**: un reranker de exclusión mutua **perfecto**, aplicado solo donde
aplica (1-a-1 y N-a-1 = 23 % de las queries), compra
`0.009×0.0791 + 0.220×0.1076 =` **+0.024 de MRR**. Y R15 midió pick accuracy real de 44 %.

⚠️ **Corrección a un argumento que circulaba**: "los candidatos no se comparan entre sí" es
**falso a nivel de objetivo** — la CE de grafo completo es un softmax sobre las N entidades y ese
acoplamiento vale **+0.167 de MRR** (2026-08-08 c). Lo puntual es la representación, no la loss.

### 2. GATE de poda — A\*Net NO descarta nada recuperable (Δ +0.0002)

Mismo checkpoint, distinto `test_node_ratio`. Es un parámetro de **eval**
(`AStarNet/reasoning/model.py:259`: `node_ratio if self.training else test_node_ratio`), y el
propio config de wikikg2 de los autores lo sube en test (0.002 → 0.01) ⇒ uso previsto:

| `test_node_ratio` | MRR | H@1 | H@10 |
|---|---:|---:|---:|
| 0.1 (el de entrenamiento) | 0.4100 | 0.3215 | 0.5833 |
| 0.5 | 0.4092 | 0.3195 | 0.5833 |
| **1.0 (sin podar)** | **0.4102** | 0.3206 | 0.5846 |

Comparación **pareada** (n=40 932): **26.1 % mejora / 48.1 % empata / 25.8 % empeora** —
simétrico ⇒ ruido de propagación, no recuperación. De las **10 716 queries con rank 2-10** (las
únicas que un reranker sobre top-K podría arreglar) solo el **5.0 %** pasa a rank 1 sin podar, y
el rank medio de ese grupo **empeora** (4.38 → 6.33).

⇒ **La hipótesis del reranker sobre el vecindario muere**: no hay evidencia descartada que
recuperar. **Con esto la opción B queda refutada en los dos mecanismos que se pudieron nombrar.**

✅ **Resultado lateral CITABLE**: **la poda de A\*Net es sin pérdida en inferencia** — propagar
sobre el 10 % de los nodos rankea igual que sobre el 100 % (0.4100 vs 0.4102). Es más fuerte que
lo que ellos reportan. ⚠️ Mide poda **en inferencia**; no dice que entrenar con 10 % equivalga a
entrenar con 100 %.

### 3. Modo de fallo — NO es el alcance

Distancia h–t en el grafo de train (no dirigida; el modelo agrega inversas), cruzada con el rank:

| distancia | queries | MRR | H@10 |
|---|---:|---:|---:|
| d=2 | 73.4 % | 0.4487 | 0.6264 |
| d=3 | 25.8 % | 0.2954 | 0.4610 |
| d=4-6 | 0.2 % | 0.0251 | 0.0556 |
| inalcanzable | 0.1 % | 0.0002 | 0.0000 |

**De las 17 056 queries que fallan fuera de top-10, solo el 0.3 % está fuera del horizonte de 6
saltos.** El 99.2 % de los pares está a 2-3 saltos. (Ningún par de test a distancia 1 ⇒ el
dataset no tiene fuga de un salto, como corresponde a FB15k-237.)

⇒ **Ni más capas, ni atajos, ni alcance global arreglan esto.** Cierra por la vía dura la lista
negra #8: el expander dio cero porque **la alcanzabilidad nunca fue el cuello de botella**.

### 4. Modo de fallo — SÍ es el grado, y NO por sesgo del modelo

| bucket | queries | **grado fuente** | **grado respuesta** | vec. comunes |
|---|---:|---:|---:|---:|
| rank 1 | 32.2 % | 33 | **156** | 3 |
| rank 2-10 | 26.2 % | 39 | 57 | 2 |
| rank 11-100 | 21.0 % | 64 | 32 | 1 |
| **rank >100** | 20.7 % | **102** | **26** | 1 |

Monótonas y en direcciones opuestas: el caso que se resuelve es *fuente periférica → respuesta
hub*; el que falla es *fuente hub → respuesta long-tail*.

**Test de sesgo de popularidad** (grado del mejor distractor vs grado de la respuesta correcta):

| | mediana | media |
|---|---:|---:|
| respuesta correcta | 43 | 335 |
| mejor distractor | 47 | 196 |
| entidad promedio del KG | 18 | 29 |

Globalmente el distractor top-1 supera en grado a la respuesta el **50.0 %** de las veces, ratio
mediano **1.01** ⇒ **NO hay sesgo de popularidad.**
⚠️ Los condicionales (59.5 % en rank 2-10, 60.6 % en rank >10) son **efecto de selección**, no
sesgo: condicionar sobre "falló" selecciona los casos donde otros quedaron arriba. **No citarlos
como evidencia de sesgo.**

**MRR por decila de grado de la respuesta:**

| decil (grado) | MRR | H@10 |
|---|---:|---:|
| 1 (0-14) | **0.2316** | 0.3871 |
| 5 (33-43) | 0.2799 | 0.4341 |
| 8 (99-249) | 0.5734 | 0.8511 |
| 10 (655-5984) | **0.9172** | **0.9901** |

**4× de diferencia en MRR según qué tan conectada esté la respuesta**, sin sesgo que lo explique.

**Análisis**

- **(A) El modo de fallo dominante es escasez de evidencia en la cola larga**, no alcance ni
  sesgo. Eso descarta de una vez: reranking sobre top-K (en el bucket >100 la respuesta ni
  está), más capas, atajos estructurales y debiasing de grado.
- **(B) ⚠️ Lo que NO queda establecido**: "limitado por evidencia" **no** es "en el techo de
  información". Lo medido es que el modelo rinde peor donde hay menos aristas; no prueba que
  otra arquitectura no extraiga más de esas pocas aristas. **Ésa es la pregunta que queda viva.**
- **(C) Eje nuevo para el paper, y sale gratis**: **evaluación estratificada por grado de la
  respuesta**. Casi ningún paper de KGC lo reporta y el pipeline ya existe. Si nuestro GT
  difiere de NBFNet/A\*Net **en la cola** aunque empate en MRR agregado, eso es contribución — y
  es lo que más importa a escala: en wikikg2 la cola larga es una fracción mucho mayor del
  dataset que en FB15k-237.

**Decisión**

- **Opción B cerrada** como estaba planteada (los dos mecanismos refutados). No se descarta el
  eje "cola larga", que es lo único que sale de estas mediciones con mecanismo detrás.
- **LANZADO NBFNet FB15k-237** (job 92147, 3 semillas en serie, ~55 h): cierra el bloqueador #1
  **y** ahora tiene un segundo motivo — es el brazo que dice si propagar sobre el grafo completo
  rescata algo en la cola de bajo grado.
- Artefactos: `dump_astarnet_ranks.py` (monkey-patch de la task; **no toca el repo**),
  `analyze_rerank_ceiling.py`, `analyze_prune_recovery.py`, `analyze_failures.py`,
  `analyze_degree_bias.py`, `sbatch_prune_ablation.sh`, `run_dump_ranks.sh`,
  `astarnet_ranks_*.pkl`, `logs/dump_ranks_921{45,46}.log`, `logs/prune_ablation_92100.log`.

---

## 2026-08-22 — 🎯 **REQUISITO DURO NUEVO: el paper WWW tiene que escalar a ogbl-wikikg2.** Se abren las opciones **A (poda aprendida dentro del GT)** y **B (reranking sobre A\*Net)**, ambas a intentar estos días

**Contexto**: decisión del usuario. *"Necesito que esto sea escalable sí o sí... necesito que escale
hasta ogbl-wiki para este paper"*, con disposición explícita a soltar el GT disperso si hace falta.
Esta entrada registra el reencuadre técnico y las dos rutas que se van a intentar.

### El reencuadre: escalar a wikikg2 = PODAR. Todo lo demás es río abajo

A 2.5 M entidades y 32.2 M aristas dirigidas, **cualquier método que toque el grafo completo por
query muere** — NBFNet incluido (OOM con batch 1, reportado en el paper de A\*Net). A\*Net es el
único path-based que lo logra, propagando el **0.2 % de nodos y aristas**.

**Cálculo que cambia el diagnóstico** (extrapolado del ancla medida: 24.4 GB con E = 544 K, batch 8,
L6/d32, job 92064; la memoria es `O(B·E·d·L)`, lineal en `B·E`):

| grafo sobre el que se propaga | aristas | memoria estimada | ¿entra en A100-40GB? |
|---|---:|---:|---|
| wikikg2 **completo**, batch 1 | 32.2 M | **~180 GB** | ❌ (4.5× por encima) |
| wikikg2 **podado al 0.2 %**, batch 32 | **~64 K** | **~5-15 GB** | ✅ |

⇒ **El muro no es del modelo, es de propagar sobre todo el grafo.** Sobre un subgrafo podado
nuestro GT entra sin problema. La pregunta deja de ser "qué arquitectura" y pasa a ser
**"de dónde sale la poda"**.

### ⚠️ Mamba NO ataca este cuello de botella — no re-proponerlo por escalabilidad

Surgió como candidato en una discusión externa. **No aplica**: el cuello en wikikg2 no es la
longitud de secuencia sino **cuántos nodos/aristas hay que tocar por query**. Mamba cambia
`O(L²)` por `O(L)` en la secuencia y no reduce ese conjunto. Tokenizando el grafo como secuencia
de nodos seguís tocando 2.5 M nodos; tokenizando caminos la secuencia son 6 hops y la ventaja es
nula. **Sin poda, Mamba no escala.** (Puede tener valor como encoder barato de caminos —el orden
de un camino es natural, así que no sufre el problema de ordenamiento que hunde a Graph-Mamba en
grafos generales— pero eso es eficiencia, no escala.)

### Las dos rutas que se van a intentar

**OPCIÓN A — poda aprendida DENTRO de nuestro GT.** Implementar selección top-K de nodos + top-L
de aristas con el top-k sin padding de A\*Net (Apéndice C: multi-key sort, dos sorts estables),
función de prioridad y weight sharing; validar que reproduce a escala FB15k-237 antes de escalar.
- ⚠️ **Riesgo alto**: 2-4 semanas realistas solo de implementación + validación, más corridas caras
  en wikikg2 (su config: 50 ép × 400 batches).
- ⚠️ **Problema de fondo a tener presente**: el resultado es estructuralmente *"A\*Net con atención
  en vez de agregación fija"*, y los datos propios dicen que la atención no le gana a la agregación
  fija; más el **3× estructural** de perder el kernel fusionado (entrada 2026-08-21 c).

**OPCIÓN B — reranking sobre A\*Net.** ⇒ **CERRADA el mismo día, ver entrada 2026-08-22 (b)**:
sus dos mecanismos quedaron refutados (exclusión mutua acotada en +0.024 y con el margen al
revés de lo predicho; poda sin pérdida, Δ +0.0002). Lo de abajo es el planteo original.
Escala **por construcción** hasta donde escala A\*Net. La
poda la aporta A\*Net (ya corre en nuestro env `astarnet`); la contribución sería la capa de
arbitraje entre candidatos.
- ⚠️ **Prior fuerte en contra, del proyecto viejo** (`SESSION_NOTES` HISTÓRICO, R14/R15):
  reranker listwise sobre top-K **dañó −0.08** (empeoró 99 queries, mejoró 2); con bias por par la
  señal era real (pair > shuffle) pero el techo calculado quedó **~baseline** con pick accuracy
  44 %. Y `sparse_nbfv` (backbone de caminos + atención encima) midió 0.421 vs 0.459 ⇒
  net-destructivo. **Son n≈1 y de otro codebase ⇒ priors, no pruebas** (regla vigente: nada se
  hereda sin re-medir).
- ✅ **Margen que sí existe**: el techo oráculo de un reranker sobre top-K es ≥ H@K. Para A\*Net en
  FB15k-237: **H@10 0.5834 contra MRR 0.4084 ⇒ +0.175 sin explotar.** El problema de R15 no era
  falta de techo sino pick accuracy.
- ✅ **Hipótesis mecánica que R14/R15 NO probaron**: cardinalidad y exclusión mutua. Es una
  restricción sobre el CONJUNTO que el scoring puntual no puede representar. ⚠️ Ojo con el
  argumento "los candidatos no se comparan": **la CE de grafo completo ya los acopla vía softmax**,
  y eso vale +0.167 de MRR (2026-08-08 c). Lo puntual es la representación, no el objetivo.

### GATE BARATO para B (hacer PRIMERO, decide entre A y B)

Con los checkpoints de A\*Net que ya están en disco: volcar ranks por query en FB15k-237 y calcular
el **techo oráculo de un reranker sobre top-K, desglosado por cardinalidad de relación**.
- Si el margen se concentra en **1-a-1 / N-a-1** ⇒ la hipótesis de restricciones de conjunto está
  viva, B tiene mecanismo y escala gratis.
- Si es **plano** ⇒ B queda muerta y la única ruta a wikikg2 es A, con su riesgo.

Costo: una GPU un rato para el dump de ranks; el resto es CPU.

**Decisión**

- **Se intentan A y B en los próximos días.** Prioridad: correr el gate de B primero porque cuesta
  horas y decide.
- ⚠️ **Evaluación de riesgo, registrada a propósito**: con ~6 semanas al deadline, partiendo de
  tres baselines, un brazo corriendo y cero poda implementada, **comprometerse a wikikg2 es de
  riesgo alto en las dos rutas**. B es la única con chance realista de entrar en plazo, y solo si
  el gate sale bien. Si ninguna funciona, el fallback declarado es YAGO3-10 (123 143 entidades,
  factible con batch 2, ~medio día por semilla) más el análisis honesto del muro de wikikg2, que
  es **compartido con NBFNet**.
- **NO cambia el camino crítico de hoy**: sigue siendo NBFNet FB15k-237 (4º brazo del bloqueador
  #1, sin lanzar) y el job 92064 (GT + BCE + `--dependent` + `--remove_one_hop`, semilla 42,
  época 5/20). Sin esos dos números, ni A ni B son interpretables.
- Los expander quedan **definitivamente fuera** (decisión del usuario de hoy, sobre la lista
  negra #8 ya establecida).

---

## 2026-08-21 (c) — 🔍 **POR QUÉ el GT es 4× más lento que NBFNet en transductivo, y por qué NO se arregla: el kernel `rspmm` se desactiva cuando los pesos de arista requieren gradiente — que es la definición de atención.** A\*Net lo esquiva pesando NODOS, no aristas

**Contexto**: pregunta del usuario sobre si el GT se va a demorar mucho más que NBFNet y si bajar
a L4 ayudaría. Al buscar la causa aparece un límite **estructural**, no de implementación.

### El mecanismo (leído del código de ellos)

`AStarNet/reasoning/layer.py:132` — y el mismo chequeo en `ULTRA/ultra/layers.py:91`:

```python
def message_and_aggregate(self, graph, input):
    if graph.requires_grad or self.message_func == "rotate":
        return super().message_and_aggregate(graph, input)   # ruta LENTA, no fusionada
```

⇒ **el kernel fusionado se apaga si los pesos de arista requieren gradiente.**

Y A\*Net **esquiva** eso (`AStarNet/reasoning/model.py:319`):

```python
layer_input = F.sigmoid(subgraph.score).unsqueeze(-1) * subgraph.hidden
```

Su función de prioridad aprendida se aplica a los **estados de NODO** `(B,N,d)` **antes** de la
capa, no como peso de arista ⇒ `edge_weight` queda sin gradiente y **conserva el kernel**.
La atención no puede hacer eso: sus pesos **son por arista y requieren gradiente por
definición** ⇒ la ruta fusionada le está vedada, no por implementación sino por la forma del
modelo.

### Descomposición del costo (todo medido)

| | h/época | h/semilla (20 ép) |
|---|---:|---:|
| NBFNet **con** kernel, batch 64 (2026-08-19 c) | 0.90 | **18.4** |
| NBFNet-PyG **sin** kernel, batch 16 (2026-08-19 b) | 2.67 | ~56 |
| **GT sparse L6/d32 b8 + `--dependent`** (job 92064) | **3.70** | **~76** |

⇒ **4.1× contra NBFNet-con-kernel, pero solo 1.39× contra el mismo NBFNet sin kernel.**
**~3× de la brecha es el kernel; ~1.4× es nuestro modelo.** Es la forma honesta de reportarlo,
y explica también por qué no podemos igualar su batch 64: su memoria es `O(B·V·d)` y la nuestra
`O(B·E·d)`.

### ¿Bajar a L4?

Ahorraría ~34 % (2.47 h/época ⇒ ~50 h/semilla). **NO se hace**, por dos razones:

- **L es el horizonte de razonamiento** en un modelo path-based (caminos de hasta L saltos), no
  un hiperparámetro cualquiera. Los 4 configs de A\*Net/NBFNet usan **6 capas en los 4
  datasets** y su ablación muestra MRR monótono con la profundidad, saturando en 6.
- ⚠️ **La evidencia que la bitácora usaba para justificar L4 en transductivo está CONFUNDIDA.**
  El sweep del 2026-07-20 comparó **L6/dim64 vs L4/dim32** — movió las dos variables a la vez y
  atribuyó el +0.006 a "capacidad". **Nunca se midió L4 vs L6 a dimensión fija en transductivo.**
  La frase "L4 da ~98 % del rendimiento a ~40 % del costo" no está sostenida por esa medición.

**Decisión**

- **L6/d32 confirmado** para el brazo transductivo. Las 3 semillas van como jobs independientes
  ⇒ cuando hay GPUs corren en paralelo y el wall es ~76 h, no 228 h.
- **La eficiencia del GT NO se ataca con menos capas.** La palanca real es un kernel fusionado
  con backward por los pesos de arista, que **habría que escribir** (torchdrug no lo tiene) ⇒
  trabajo futuro, no algo para octubre.
- **Material para el paper**: *"la atención paga un costo estructural que la agregación fija y la
  poda a nivel de nodo no pagan: pesos aprendidos por arista impiden fusionar message+aggregate"*.
  Es un límite de la familia, medido y con la línea de código que lo causa.

---

## 2026-08-21 (b) — 🔧 IMPLEMENTADO `--dependent`: representación relacional **generada desde la query**, el mecanismo de NBFNet / A\*Net. Es capacidad relacional **por QUERY, no por arista** ⇒ el eje barato, al revés que `--rel_param lowrank`

**Contexto**: pregunta del usuario sobre con qué capacidad relacional lanzar los experimentos de
GT con BCE. Al leer los cuatro códigos aparece que la pregunta estaba mal planteada en la
bitácora: el problema nunca fue *cuántos* parámetros por relación, sino **dónde se aplican**.

### Qué hace cada uno (leído del código, no de los papers)

| modelo | mecanismo | params/relación dirigida/capa (d=32) |
|---|---|---:|
| NBFNet / A\*Net `dependent: yes` | `Linear(d, R·d)(q) → (B,R,d)`, **una vez por query** | d²+d = **1056** |
| NBFNet / A\*Net `dependent: no` | `nn.Embedding(R, d)` por capa | d = **32** |
| ULTRA (GNN de entidades) | **ninguno**: las computa del grafo de relaciones (`project_relations`) | **0** |
| **Nuestro GT, tabla estática** | `rel_bias` (H,R) + `rel_value` (H,R,hd) | H+d = **40** |

⇒ **Nuestro modo estático YA ERA su `dependent: no`.** Lo que faltaba era el `yes`.

### Qué usan ELLOS en cada dataset (`AStarNet/config/transductive/`)

| WN18RR | FB15k-237 | YAGO3-10 | ogbl-wikikg2 |
|---|---|---|---|
| **`no`** | `yes` | `yes` | `yes` |

**WN18RR es la excepción, no la regla — y no es por tener pocas relaciones**: YAGO3-10 tiene ~37
y usa `yes`. Verificación de escala: en wikikg2, `relation_linear` sola son
6 × 1070 × (32²+32) = **6.78 M** parámetros contra los **6.83 M** que reporta el paper ⇒ **~99 %
de A\*Net en un grafo de 2.5 M entidades es la parametrización relacional, y 0 % son embeddings
de entidad.** Ése es el motivo de que escale, contra los ~250 M de ComplEx+RP.

### El hallazgo que reordena la línea de capacidad relacional

| | se aplica | costo | medido |
|---|---|---|---|
| `dependent` (nuevo) | una vez por **QUERY** | `O(B·R·d²)`, R ≪ E | — |
| `--rel_param lowrank` | una vez por **ARISTA** | gather `(E,H,hd,hd)` | **2.3× más lento** en transductivo |

Por eso NBFNet puede darle **26× más** capacidad relacional que nosotros y aun así ser **9× más
rápido** que nuestro full attention. Es la misma lección de la entrada 2026-08-08 (k) —*en este
harness se cuentan tensores por arista, no FLOPs*— aplicada al eje relacional. ⇒ **si alguna vez
se quiere cerrar la asimetría relacional de FB15k-237, la forma correcta es ésta, no `lowrank`.**

### Implementación (HECHA)

`--dependent` (store_true) en `train.py`; cableado en las dos capas de atención ⇒ lo heredan
`rfat`, `sparse` y `sparse_exp`. `rel = relation_linear(q_emb).view(B,R,H,hd)` una vez por capa
por batch, leído por arista con **`index_select`** (regla 2026-08-09). El `rel_bias` **NO** se
toca: sigue estático, así cambia **un solo canal** y el ablation es limpio.

**Init deliberadamente distinto al de ellos**: peso **0** y bias **1** ⇒ `rel == 1` para toda
relación, que es exactamente el init de `rel_value` (`torch.ones`). NBFNet usa el init default de
`nn.Linear` (bias 0). Se eligió el nuestro para que **el forward al arranque sea bit a bit
idéntico al modo estático** y el contraste tenga control exacto — misma convención que `lowrank`
(init 0) y que `exp_typing path ≡ single`.

**Verificación (smoke test CPU, los 3 modelos)**

| qué | resultado |
|---|---|
| estático vs `--dependent` al init, pesos compartidos igualados | **max\|diff\| = 0.000e+00** en los 3 |
| gradiente en `relation_linear` tras 2 pasos de optimizador | 6/6 tensores (L capas × weight+bias) |
| las relaciones se **diferencian** tras entrenar (control del fallo de `lowrank`) | dispersión 0.11–0.20 ⇒ vivo |
| guard `--dependent` + `--rel_param lowrank` | ✅ bloqueado |
| guard `--dependent` + `--exp_typing {ultra,path}` | ✅ bloqueado (recomponen `rel_value`, que acá no existe) |

**Parámetros con las configs del paper** (GT L4/d32/H8 vs NBFNet L6/d32):

| dataset | GT estático | **GT `--dependent`** | NBFNet |
|---|---:|---:|---:|
| FB15k-237 transd. (R=474) | 126 497 | **2 072 097** (16.4×) | 3 099 969 |
| WN18RR transd. (R=22) | 39 713 | **133 921** (3.4×) | 221 633 |

En los dos casos el GT con `--dependent` queda **por debajo** de NBFNet (0.67× y 0.60×) ⇒ si
mejora, no se puede atribuir a tener más parámetros que el baseline.

⚠️ **NO TRANSFIERE ZERO-SHOT**: `Linear(d, R·d)` tiene R horneado en la forma del parámetro.
Sirve para el paper transductivo; para las Etapas 2-3 de la tesis la ruta es ULTRA (cero
parámetros por relación). ULTRA lo desactiva a propósito.

**Decisión**

- `--dependent` disponible, verificado y **MEDIDO en GPU** (job 92055, A100, fwd+bwd con
  warmup, mismos batches y semilla en los dos brazos):

  | régimen | E | **tiempo** | **memoria** | params estático → dependent |
  |---|---:|---:|---:|---|
  | FB15k-237 ind v1, `sparse` L6/d64, b16 | 8 490 | **1.08×** | 1.19× (1.72 → 2.04 GB) | 384 K → **9.26 M** ⚠️ |
  | FB15k-237 ind v1, `rfat` L6/d64, b16 | 8 490 | **1.01×** | 1.02× (11.85 → 12.11 GB) | 384 K → **9.23 M** ⚠️ |
  | WN18RR **transd.**, `sparse` L4/d32, b32 | 173 670 | **1.04×** | 1.12× (26.85 → **30.02 GB**) ⚠️ | 40 K → 134 K |
  | FB15k-237 **transd.**, `sparse` L4/d32, b8 | 544 230 | **1.05×** | 1.13× (14.77 → 16.66 GB) | 126 K → 2.07 M |

  **CONFIRMADO: el eje por-query es barato.** 1.01–1.08× en tiempo contra el **2.33×** que
  cuesta `--rel_param lowrank` en el mismo dataset y config (5.71 → 2.45 it/s, entrada
  2026-08-08 k). El tensor `(B,E,H,hd)` extra por capa se paga en **memoria (1.02–1.19×)**, no
  en tiempo: estas capas ya materializan varios tensores por arista, así que uno más no cambia
  el régimen. En el full attention es literalmente gratis (1.01×) porque domina el `O(N²)`.

  ⚠️ **DOS AVISOS que salieron del benchmark y no estaban previstos:**
  1. **WN18RR transductivo queda a 30.0 GB de 40 (75 %)**, contra 26.9 GB del estático. El
     margen baja a 10 GB ⇒ **subir batch, `hidden_dim` o combinar con otra cosa puede dar OOM**
     ahí. Es el brazo más apretado de los cuatro, y ya lo era antes.
  2. **Con `hidden_dim 64` los parámetros explotan a ~9.2 M**, o sea **3.9× NBFNet (2.37 M)**,
     porque `R·d²` escala con d². Con la config transductiva (d=32) no pasa: 2.07 M contra
     3.10 M de NBFNet, o sea **queda por debajo**. ⇒ **En transductivo `--dependent` es
     parameter-matched-por-abajo y el argumento se sostiene; en inductivo con d=64 NO**, y
     habría que bajar a d=32 en ese brazo o declarar la diferencia. La propiedad "si mejora no
     es por tener más parámetros" **solo vale en transductivo**.
- **Regla del usuario (hoy)**: todos los experimentos van con `--loss bce` salvo que se pida lo
  contrario. Registrado en `CLAUDE.md`.
- Artefactos: `src/model.py` (bloque de docs `--dependent`, `_make_relation_linear`, las dos
  capas y los tres modelos), `train.py::--dependent`, smoke test en el scratchpad de la sesión.

---

## 2026-08-21 — 🎯 **TRES DE LOS CUATRO BRAZOS DEL BLOQUEADOR #1, CERRADOS con n=3: A\*Net FB15k-237 0.4084 ± 0.0014 y NBFNet WN18RR 0.5466 ± 0.0015 ⇒ los tres reproducen la literatura dentro de 1.1 %.** Falta solo NBFNet FB15k-237

**Contexto**: terminaron los jobs **91884** (NBFNet WN18RR, 20-ago 19:19) y **91894** (A\*Net
FB15k-237, 21-ago 17:47), ambos con las 3 semillas completas y `rc=0`. Con la entrada
2026-08-19 (d) (A\*Net WN18RR) quedan **tres de los cuatro brazos** de baselines externos del
paper WWW medidos con el protocolo vigente (n=3, media ± sd). No hay jobs en cola.

### Resultado — baselines transductivos, 20 ép, n=3 semillas 1024/1025/1026, full-filtered

| brazo | **test MRR** | H@1 | H@3 | H@10 | MR | **paper** | **Δ** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NBFNet WN18RR** | **0.5466 ± 0.0015** | 0.4906 | 0.5706 | 0.6619 | **652.9 ± 5.0** | *0.551* | **−0.0044** (−0.8 %) |
| **A\*Net WN18RR** | **0.5431 ± 0.0013** | 0.4890 | 0.5669 | 0.6523 | 5853.7 ± 18.5 | *0.549* | **−0.0059** (−1.1 %) |
| **A\*Net FB15k-237** | **0.4084 ± 0.0014** | 0.3192 | 0.4483 | 0.5834 | 497.8 ± 39.8 | *0.411* | **−0.0026** (−0.6 %) |
| **NBFNet FB15k-237** | **PENDIENTE** | — | — | — | — | *0.415* | — |

Mejores épocas por semilla (torchdrug valida cada 2 ⇒ 10 validaciones): A\*Net FB237 **18/16/18**,
NBFNet WN18RR **18/20/20**, A\*Net WN18RR 20/18/16.

### Contraste NBFNet vs A\*Net en WN18RR — el que aísla la poda A\*

| | NBFNet | A\*Net | Δ (Welch) |
|---|---:|---:|---:|
| **MRR** | 0.5466 ± 0.0015 | 0.5431 ± 0.0013 | **+0.0035 ± 0.0011 (t=3.11)** |
| **MR** | **652.9** | **5853.7** | **9.0×** |

Los dos brazos comparten codebase, pipeline de datos y kernel `rspmm` (ambos salen de
`astarnet_cfg/*_trans.yaml`, o sea de `AStarNet/config/transductive/`) ⇒ **la única diferencia
es la poda**, sin mezclar efectos de implementación.

**Análisis**

- **(A) Los tres brazos reproducen, y el sesgo es sistemático y del mismo signo: −0.6 % a
  −1.1 %.** Que las tres desviaciones sean negativas, chicas y de magnitud parecida es la firma
  de un pipeline correcto con una diferencia menor de entorno (versión de torch/CUDA, o el
  no-determinismo del kernel), no de tres errores independientes. **El setup externo queda
  validado** y las tablas del paper pueden citar estos números como "nuestra reproducción".
- **(B) ⚠️ Corrige una premisa que la bitácora usaba desde el 2026-08-08 (c): el harness NO
  corre "0.011–0.016 por encima" de la literatura en estos brazos — corre POR DEBAJO.** Ese
  +0.011/+0.016 es de **nuestro** harness con **CE de grafo completo** en inductivo; los repos
  originales con **BCE-k** en transductivo caen 0.003–0.006 **por debajo** del número publicado.
  Son cosas distintas y no hay que mezclarlas: el techo contra el que se compara el paper WWW
  es **el baseline propio medido acá**, no el de la tabla publicada ni el del harness inductivo.
- **(C) La ventaja de NBFNet sobre A\*Net en WN18RR es real pero minúscula (+0.0035, t=3.11), y
  reproduce la dirección del paper** (0.551 vs 0.549, +0.002). Con n=3 y σ≈0.0014 el contraste
  es significativo sin ser relevante: confirma la tesis de A\*Net —la poda no cuesta calidad—
  a un costo 5.8× menor de wall.
- **(D) El MR de A\*Net es 9.0× peor que el de NBFNet en el mismo dataset (5854 vs 653), con
  MRR/H@1/H@10 casi idénticos.** Es la propiedad de la poda ya anticipada en la entrada
  2026-08-19 (d), ahora **con el brazo de control en el mismo codebase**, que es lo que faltaba
  para poder afirmarlo: los nodos que la búsqueda A\* nunca visita quedan sin score y arruinan
  la cola del ranking. **Es material directo para el paper**: es exactamente la métrica donde un
  GT disperso que propaga por TODO el grafo debería ganar, y ahora la brecha está medida contra
  un baseline propio en vez de contra el ~636 de la tabla de NBFNet.
  ⚠️ Ojo: en FB15k-237 A\*Net da MR **497.8**, o sea el problema **no se hereda entre datasets**
  (regla vigente del proyecto). Habrá que ver el MR de NBFNet FB237 antes de generalizarlo.
- **(E) ⚠️ NBFNet WN18RR es un PISO: 2 de 3 semillas eligen la época 20 de 20.** El protocolo
  vigente pide verificar que el best-valid no caiga en las últimas épocas y acá cae en dos.
  El brazo está **subentrenado**, así que el 0.5466 solo puede subir. **No inflar la lectura del
  +0.0035 sobre A\*Net con eso** (A\*Net tiene el mismo problema en una semilla), pero sí
  declararlo si se cita el número absoluto. Es el protocolo de los autores (20 épocas), así que
  extenderlo sería desviarse de su receta — mejor reportarlo como está y anotar la salvedad.
- **(F) Señal preliminar, sigue SIN ser citable.** Nuestro GT sparse en WN18RR transductivo dio
  test **0.5536** (sparse) y **0.5566** (sparse_exp deg3) con **MR 1616/1497**, contra estos
  0.5466 (NBFNet) y 0.5431 (A\*Net) ⇒ **+0.007 / +0.010 de MRR y 4× mejor MR que A\*Net**.
  **Pero nuestros números son n=1, 20 ép, L4/d32 y con CE de grafo completo, y éstos con BCE-k**
  — el confounder de loss que motivó el cambio de protocolo del 2026-08-19. **No es una
  comparación válida hasta re-correr nuestro harness con `--loss bce` y n≥3.** Es ahora la
  medición más valiosa que falta del lado propio.

**Decisión**

- **Los tres brazos quedan registrados como baselines del paper WWW.** Números de la tabla de
  arriba, extraídos de los logs con un parser (no transcritos a mano).
- **Falta el cuarto y último: NBFNet FB15k-237** (~18.4 h/semilla ⇒ ~55 h las tres en serie).
  Es el que cierra el bloqueador #1 y el que hace interpretable el mejor número transductivo del
  proyecto (sparse + `--remove_one_hop`, test 0.42893, n=1).
  ⇒ **PREPARADO Y NO LANZADO (decisión del usuario, 2026-08-21)**: `sbatch_nbfnet_fb237_trans.sh`
  queda escrito y verificado (config a 2 líneas de diff del original, wall 68 h, semillas
  1024/1025/1026). **Hoy solo hay 1 GPU libre en el nodo** — las otras 7 están ocupadas por otros
  usuarios — así que el brazo se pospone a un día con más margen. El script acepta `SEEDS=<n>`
  para lanzar una semilla por job si alguna vez hay 3 GPUs libres (~18.4 h de wall en vez de 55).
  ⚠️ Nota operacional: `squeue -u $USER` lista solo los jobs PROPIOS ⇒ no sirve para saber si
  hay GPUs libres. Para eso, `sinfo`/`squeue -p AI` o `nvidia-smi` en el nodo.
- **Cola inmediata después de ese brazo**: (1) nuestro harness en transductivo con `--loss bce`,
  n≥3, en los dos datasets — sin eso ninguna comparación con estos baselines es válida;
  (2) `--edge_drop` y `--remove_one_hop` en transductivo como brazos separados y combinado.
- Artefactos: `logs/nbfnet_wn_trans_91884.log`, `logs/astarnet_fb237_trans_91894.log`,
  `experiments_astarnet/KnowledgeGraphCompletion/{WN18RR/NBFNet,FB15k237/AStarNet}/`.

---

## 2026-08-19 (d) — 🎯 **PRIMER BASELINE EXTERNO DEL PAPER, con barras de error: A\*Net en WN18RR transductivo mide 0.5431 ± 0.0013 (n=3) contra el 0.549 publicado ⇒ REPRODUCIDO.** Y cae la 4ª y última incompatibilidad de torchdrug (`is_meta`)

**Contexto**: primer lanzamiento real de la etapa transductiva (job 91869). Se eligió el brazo
**más barato de los cuatro** (~1.6 h/semilla) a propósito, porque además era el test barato del
último riesgo abierto — ver punto (B).

### Resultado (WN18RR transductivo, 20 ép, n=3 semillas 1024/1025/1026, full-filtered)

| | valid | **test** |
|---|---:|---:|
| **MRR** | 0.5402 | **0.5431 ± 0.0013** |
| Hits@1 | — | 0.4890 |
| Hits@3 | — | 0.5669 |
| Hits@10 | — | 0.6522 |
| MR | — | **5854** ⚠️ |

| referencia | MRR |
|---|---:|
| **A\*Net, nuestra corrida (n=3)** | **0.5431 ± 0.0013** |
| *A\*Net, paper (Tabla 1)* | *0.549* |
| *NBFNet, paper* | *0.551* |

Mejor época por semilla: 20 / 18 / 16 (curvas de valid monótonas hasta ~0.54, sin colapso).

**Análisis**

- **(A) REPRODUCIDO: Δ −0.006 contra el número publicado (1.1 % relativo), con σ = 0.0013.**
  Es el primer baseline externo del proyecto medido con el protocolo vigente (n≥3, media ± sd) y
  valida el pipeline completo — env, datos, config y kernel — de punta a punta.
- **(B) CAE LA 4ª Y ÚLTIMA INCOMPATIBILIDAD DE `SESSION_NOTES` 2026-08-08 (c).** El job entrenó
  las 20 épocas × 3 semillas **correctamente** y murió recién en `AStarNet/script/run.py:36`,
  `solver.load("model_epoch_%d.pth" % best_epoch)`, con
  `AttributeError: 'Graph' object has no attribute 'is_meta'`.
  **Causa**: torchdrug registra objetos `Graph` en el `state_dict`; `torch.nn.Module.
  _load_from_state_dict` (torch 2.1, `module.py:2024`) evalúa `param.is_meta` sobre cada entrada.
  `Graph` sí tiene `.shape` (pasa el chequeo previo de `module.py:2017`) pero no `is_meta`.
  **Arreglo**: `Graph.is_meta = False` en `run_astarnet.py` — `False` es el valor correcto, un
  `Graph` nunca está en meta device. ⇒ **las cuatro incompatibilidades de la entrada 2026-08-08
  (c) están resueltas; torchdrug ya no es un bloqueador.**
- **(C) Los resultados se RECUPERARON sin re-entrenar.** Los checkpoints (10 por semilla, cada 2
  épocas) estaban en disco. Se leyó la mejor época de la curva de valid del log y se evaluó ese
  checkpoint con `num_epoch: 0` + `checkpoint:` en el config ⇒ **3 min en vez de 5 h.**
  **Patrón operacional a reutilizar** (tercera vez que sirve en el proyecto, ver 2026-08-08 (i)
  y 2026-08-10 (a)): un job que muere DESPUÉS de entrenar casi nunca es trabajo perdido.
- **(D) Haber lanzado el brazo BARATO primero fue lo que evitó el costo.** El mismo crash en
  FB15k-237 se habría descubierto tras **18 h** de entrenamiento en vez de 1.6 h. La decisión
  estaba escrita en el encabezado de `sbatch_astarnet_wn_trans.sh` antes de lanzar.
- **(E) ⚠️ El MR es 5854, contra ~636 que reporta NBFNet en WN18RR.** Es **esperable de un método
  que poda**: los nodos que la búsqueda A\* nunca visita quedan sin score y arruinan la cola del
  ranking, mientras MRR/H@1/H@10 se mantienen. No es un error de la corrida — es una propiedad
  del enfoque, y conviene tenerla presente porque **es justo la métrica donde un GT disperso que
  propaga por TODO el grafo debería ganarle**.
- **(F) Señal preliminar, NO citable todavía.** Nuestro GT sparse en WN18RR transductivo dio
  test **0.5536** con **MR 1616** (n=1, 20 ép, L4/d32) contra estos 0.5431 y 5854 ⇒ **+0.010 de
  MRR y 3.6× mejor MR**. **Pero nuestro número es con CE de grafo completo y éste con BCE-k**
  (ver entrada 2026-08-19): es exactamente el confounder de loss que motivó el cambio de
  protocolo. **No es una comparación válida hasta re-correr nuestro harness con `--loss bce`.**

**Decisión**

- Resultado registrado como **primer baseline externo del paper WWW**.
- **Lanzado el segundo brazo: NBFNet en WN18RR transductivo**, 3 semillas, misma GPU
  (`sbatch_nbfnet_wn_trans.sh`, ~8.6 h) ⇒ completa el par de baselines en el dataset barato antes
  de ir a FB15k-237, que cuesta 2-3× más por brazo.
- Artefactos: `sbatch_astarnet_wn_trans.sh`, `recover_astarnet_wn.sh`,
  `logs/astarnet_wn_trans_91869.log`, `logs/recover_wn_10{24,25,26}.log`,
  `experiments_astarnet/KnowledgeGraphCompletion/WN18RR/AStarNet/2026-08-19-0{5-27-46,6-58-30,8-28-58}/`.

---

## 2026-08-19 (c) — ✅ **DESBLOQUEADOS LOS REPOS ORIGINALES: el kernel CUDA `rspmm` de torchdrug COMPILA.** Env aislado `astarnet` con `nvcc` propio ⇒ **A\*Net es viable y `NBFNet/` (torchdrug) deja de ser inusable**. Seis bloqueos, seis causas, todas de empaquetado

**Contexto**: la entrada (b) dejó A\*Net como *hard-blocked* por falta de `nvcc`. El usuario pidió
probar la instalación en un entorno aislado porque **necesita correr los códigos originales**.
Resultado: **funciona**. Ninguno de los seis problemas era del modelo ni del cluster; todos eran
de toolchain, y **cada uno tenía una causa distinta a la que aparentaba**.

### Los seis bloqueos, en orden de aparición

| # | síntoma | causa real | arreglo |
|---|---|---|---|
| 1 | `ModuleNotFoundError: torch_sparse` | faltan `torch_scatter/sparse/cluster`; pip intentaba **compilarlos** en un env de build aislado sin torch | wheels precompiladas de `data.pyg.org`, y `torchdrug` con `--no-build-isolation` |
| 2 | `TypeError: metaclass conflict` en `torch_geometric/data/dataset.py:20` | torchdrug inyecta su metaclase en `torch.utils.data.Dataset`; después PyG no puede derivar de ella + `ABC`. Lo dispara `reasoning/dataset.py:5` (`from ogb import linkproppred`) | **importar PyG ANTES que torchdrug** (`run_astarnet.py`) |
| 3 | no existe `nvcc` | el cluster nuevo no trae toolkit ni module system | `cuda-nvcc 12.1` en un env conda **aislado** |
| 4 | `fatal error: ATen/SparseTensorUtils.h` | PyTorch **movió** el header a `ATen/native/`; el namespace `at::sparse` es **idéntico** ⇒ cambio de RUTA, no de API | `patch_torchdrug_aten.sh` (sobre site-packages del env aislado, con backups `.orig`) |
| 5 | `fatal error: cublas_v2.h` | conda-forge deja los headers de math en `targets/<arch>/include`, y `cpp_extension` **solo** mira `$CUDA_HOME/include` | `CPATH` en `env_astarnet.sh` |
| 6 | errores de plantilla **dentro de `pybind11/cast.h`** | ⚠️ **engañoso**: no es pybind11. **CUDA 12.1 soporta GCC hasta 12.2** y conda-forge instaló **12.4**. No emite "unsupported GNU version", falla parseando plantillas | bajar el host compiler del env a **gcc 11.4** |

**Verificación**: `generalized_rspmm` compila en **62 s** y devuelve salida finita; los imports de
`reasoning` (A\*Net) pasan.

**Análisis**

- **(A) La entrada 2026-08-08 (c) queda DESACTUALIZADA en su conclusión, no en sus hechos.**
  Ahí se declaró el repo `NBFNet/` (torchdrug) *"NO USABLE en este cluster"* listando cuatro
  incompatibilidades: metaclase PyG/torchdrug, header `ATen/SparseTensorUtils.h`, `ninja` y el
  crash de `is_meta`. **Las dos primeras están resueltas acá** (#2 y #4) y `ninja` viene con el
  env. La causa raíz de todas era **una sola**: no había `nvcc`, así que nunca se llegó a
  diagnosticar el resto. ⇒ **`NBFNet/` (torchdrug) vuelve a estar disponible** (queda por ver el
  `is_meta` al cargar checkpoints, que es lo único no probado hoy).
- **(B) Beneficio colateral grande para el paper: se puede compilar `rspmm` también para
  NBFNet-PyG.** Hoy NBFNet-PyG corre con `NBFNET_PYG_NO_RSPMM=1`, que fuerza la ruta
  `message+aggregate` de PyG con memoria **O(B·E·d)** ⇒ es la razón de que FB15k-237 transductivo
  **no entre a batch 64** y tope en 16 (entrada (b)). Con el kernel fusionado la memoria pasa a
  O(B·V·d) ⇒ **debería entrar el batch de su config y bajar de las 56 h/semilla medidas.**
  Pendiente de medir; no citar mejora hasta tenerla.
- **(C) Lección de diagnóstico, la misma de la entrada 2026-08-09**: el síntoma #6 apuntaba a
  pybind11 y la causa era la versión de GCC. Dos de los seis (#5 y #6) mienten sobre su causa.
  **En problemas de toolchain, leer la primera línea del error no basta.**

### Tiempos MEDIDOS con el kernel activo (A100, 1 GPU, configs originales sin tocar)

| brazo | batch | it/s | triples/s | h/época | **20 ép (solo train)** |
|---|---:|---:|---:|---:|---:|
| **A\*Net FB15k-237** | 64 | 2.42 | 155 | 0.49 | **~9.8 h** |
| **NBFNet FB15k-237** | 64 | 1.32 | 84 | 0.90 | **~17.9 h** |
| **A\*Net WN18RR** | 64 | 5.13 | 328 | 0.07 | **~1.5 h** |
| **NBFNet WN18RR** | 32 | 1.77 | 57 | 0.43 | **~8.5 h** |

NBFNet se corrió con **`AStarNet/config/transductive/*_nbfnet.yaml`**, o sea el mismo codebase,
pipeline de datos y kernel que A\*Net ⇒ el cociente entre ambos aísla la poda A\*, sin mezclar
el efecto del kernel.

- **Reproducimos las cifras de eficiencia del paper**: A\*Net vs NBFNet **1.83× en FB15k-237**
  (ellos reportan 2.1× de reducción de tiempo) y **5.8× en WN18RR** (reportan 6.8×). Ambas cerca
  y en el orden correcto ⇒ el setup está bien armado.
- **La poda se comporta como describen**: FB15k-237 usa 7 % de las aristas y 42 % de los nodos;
  WN18RR **2.4 % de aristas y 5.5 % de nodos** (ellos reportan reducción de mensajes de 14.1× y
  42.9×).
- ⚠️ **Corrige el presupuesto de la entrada (b)**: NBFNet FB15k-237 pasa de **2.67 h/época (~56 h,
  batch 16 forzado por OOM, sin kernel)** a **0.90 h/época (~18 h) al batch 64 de su config**.
  **3× más rápido y sin desviarse del setup original** ⇒ los dos problemas de la entrada (b)
  quedan resueltos por el kernel.
### Costo de EVALUACIÓN medido, y TOTAL por semilla

Medido con `num_epoch: 0` (salta el train y va directo a `test(cfg, solver)`, que evalúa valid y
luego test). El modelo está sin entrenar ⇒ **las métricas no sirven** (MRR ~0.002, o sea azar,
que es el control de que efectivamente no entrenó), **el tiempo sí**: el costo del ranking
full-filtered no depende de los pesos.

| brazo | valid | test | eval total en 20 ép* | **TOTAL / semilla** |
|---|---:|---:|---:|---:|
| **A\*Net FB15k-237** | 2.4 min | 2.8 min | 0.49 h | **~10.3 h** |
| **NBFNet FB15k-237** | 2.5 min | 2.8 min | 0.50 h | **~18.4 h** |
| **A\*Net WN18RR** | 14 s | 12 s | 0.05 h | **~1.6 h** |
| **NBFNet WN18RR** | 41 s | 38 s | 0.14 h | **~8.6 h** |

\* torchdrug valida cada `ceil(20/10) = 2` épocas ⇒ **10 validaciones** + valid y test finales.

- **El eval resultó BARATO: 3-5 % del total**, contra lo que hacía temer la entrada (b). El
  kernel también lo acelera: NBFNet-PyG sin `rspmm` tardaba **~11 min** por validación de
  FB15k-237; con kernel son **2.5 min** (≈4.4×).
- **Presupuesto completo del bloqueador #1** (4 brazos × 3 semillas = 12 corridas):
  **~117 GPU-h**. Con los 4 slots de QOS y la corrida más larga en 18.4 h, el wall es de
  **~30 h ≈ 1.5 días**. ⚠️ **Corrige la estimación de "4-5 días" que se dio antes de medir el
  eval**: el conjunto es bastante más barato de lo previsto.

**Decisión**

- **Env `astarnet` es el entorno de los REPOS ORIGINALES** (A\*Net y NBFNet/torchdrug); `attention`
  sigue siendo el del harness propio y `venv_nbfnet` el de NBFNet-PyG. **No se tocó ninguno de los
  dos.** Punto de entrada: `source env_astarnet.sh` (NO `env.sh`).
- ⚠️ **`env.sh` y la sección Entorno de `CLAUDE.md` NO aplican a este env** — `CLAUDE.md` además
  menciona `module load gnu12/cuda 12.6`, que el cluster nuevo no tiene. Corregir.
- El parche de torchdrug es **sobre site-packages del env aislado**, nunca sobre los repos
  clonados, que siguen intactos (criterio de la entrada 2026-08-08 c).
- **NBFNet para el paper se corre desde el repo de A\*Net** (`astarnet_cfg/*_nbfnet_trans.yaml`),
  no desde NBFNet-PyG: mismo codebase, pipeline y kernel que A\*Net ⇒ el contraste entre ambos
  aísla la poda A\*. NBFNet-PyG queda como referencia secundaria.
- Artefactos: `setup_astarnet_env.sh`, `setup_astarnet_pip{,2}.sh`, `patch_torchdrug_aten.sh`,
  `env_astarnet.sh`, `run_astarnet.py`, `astarnet_cfg/` (4 configs transductivos + 4 `_evalonly`,
  4 líneas de diff contra los originales), `bench_originals.sh`, `bench_eval_originals.sh`,
  `probe_nbfnet.py`, `logs/astarnet_kernel_gcc11.log`, `logs/bench_*.log`.

### Cola inmediata (en orden)

1. **Costo de evaluación** de los 4 brazos (job 91868 en curso). Hasta tenerlo, los totales de
   arriba son un piso.
2. ⚠️ **Verificar el crash de `is_meta` al cargar el best checkpoint** — es el único de los cuatro
   problemas de la entrada 2026-08-08 (c) que NO se probó hoy, y **aparece al FINAL de un
   entrenamiento**. Hay que descartarlo **antes** de lanzar una corrida de 18 h, no después.
   Prueba barata: entrenar 1 época con `num_epoch: 1` y dejar que llegue a `test(cfg, solver)`.
3. Lanzar **NBFNet y A\*Net, n=3 semillas, en los dos datasets transductivos**. Con los tiempos
   medidos: FB15k-237 ~18 h y ~10 h por semilla, WN18RR ~8.5 h y ~1.5 h ⇒ con los 4 slots de QOS
   el conjunto cabe en **~4-5 días de reloj**, muy dentro del plazo de octubre.
4. Re-correr **nuestro** harness en transductivo con `--loss bce` (entrada 2026-08-19) para que
   los brazos propios sean comparables con estos baselines.

---

## 2026-08-19 (b) — ⏱️ **TIEMPOS MEDIDOS de los baselines transductivos**: NBFNet ~56 h/semilla en FB15k-237 y ~10.6 h en WN18RR. **FB237 NO corre al batch de los autores (OOM a 64, tope 16)** y **A\*Net está HARD-BLOCKED: exige el kernel CUDA de torchdrug y no hay `nvcc`**

**Contexto**: primera pregunta operativa del paper WWW — cuánto cuesta una semilla de cada
baseline y con cuántas épocas se replica. Las épocas salen de sus configs; los tiempos se
**midieron en A100** con un probe que replica el loop de `NBFNet-PyG/script/run.py:54-71`
operación por operación (`probe_nbfnet.py`), en vez de estimarse.

### Épocas y config de referencia — **20 épocas en los cuatro casos**

| repo / config | épocas | batch | dim/L | `dependent` | adv. temp | `remove_one_hop` |
|---|---:|---:|---|---|---:|---|
| NBFNet-PyG transd. FB15k-237 | **20** | 64 | 32 / 6 | yes | 0.5 | **yes** |
| NBFNet-PyG transd. WN18RR | **20** | **16** | 32 / 6 | **no** | **1** | **no** |
| A\*Net transd. FB15k-237 (`node_ratio 0.1`) | **20** | 64 | 32 / 6 | yes | 0.5 | **yes** |
| A\*Net transd. WN18RR (`node_ratio 0.1`) | **20** | 64 | 32 / 6 | **no** | **1** | **no** |

⚠️ **Los dos datasets NO comparten receta** — FB237 usa `dependent: yes`, adv. temp 0.5 y
`remove_one_hop`; WN18RR usa `dependent: no`, adv. temp 1 y **sin** `remove_one_hop`. Copiar
la config de uno al otro invalida la comparación con la literatura. `AStarNet/config/` trae
además su propio `*_nbfnet.yaml` transductivo ⇒ **NBFNet y A\*Net se pueden correr en el mismo
codebase**, que es la forma limpia de que sean apples-to-apples entre sí.

### Tiempos MEDIDOS (A100-40GB, 1 GPU, NBFNet-PyG, n=1 semilla)

| | triples train | batch usable | it/s | h/época | **20 ép + eval** | pico GPU |
|---|---:|---:|---:|---:|---:|---:|
| **FB15k-237 transd.** | 272 115 | **16** ⚠️ | 1.77 | 2.67 | **~56 h (2.3 días)** | 27.5 GB |
| **WN18RR transd.** | 86 835 | 16 (el de su config) | 2.91 | 0.52 | **~10.6 h** | 18.3 GB |

Desglose del eval: `run.py` valida cada `ceil(20/10) = 2` épocas ⇒ **10 validaciones**, más
valid+test finales. FB237 ~10.9 min por validación ⇒ ~2.2 h de eval; WN18RR ~1.0 min ⇒ ~0.2 h.

**El throughput por triple es invariante al batch** (FB237: 28 triples/s a batch 16 y a batch 8;
WN18RR: 46 y 45) ⇒ bajar el batch **no cambia el wall-clock**, solo la memoria. Confirma la
observación de la entrada 2026-08-08 (e) para nuestro harness, ahora también en el de ellos.

### 🚩 Bloqueo 1 — FB15k-237 transductivo NO corre al batch de los autores

Batch 64 da **OOM en A100-40GB** (pide 4.26 GB sobre 38.6 ya usados; necesitaría ~110 GB).
Causa: sin `nvcc` no se puede compilar el kernel `rspmm`, así que se cae a la ruta
`message+aggregate` de PyG, que materializa **O(B·E·d)** en vez de O(B·V·d) — y en transductivo
E = 544 230 contra las 19 478 aristas de ind v2, donde el fallback sí era barato. Tope medido:
**batch 16** (27.5 GB). Como el throughput no depende del batch, **no cuesta tiempo**, pero
**es una desviación de su config** que hay que declarar en el paper (o compensar con
acumulación de gradiente para igualar el batch efectivo de 64).

### 🚩 Bloqueo 2 — A\*Net NO CORRE en este cluster (y no tiene fallback)

Dos problemas, el segundo es duro:

1. Faltan `torch_sparse` y `torch_scatter` en el env (`reasoning/data.py:4`), y hay un
   **conflicto de metaclase torchdrug↔PyG**: `reasoning/dataset.py:5` importa `ogb`, que importa
   PyG, y si torchdrug se cargó antes, `torch_geometric/data/dataset.py:20` explota con
   `metaclass conflict`. **Solución conocida: importar PyG ANTES que torchdrug** (verificado: con
   ese orden el error desaparece y aparece solo el de `torch_sparse`). Se arregla con un wrapper,
   sin tocar el repo — el mismo patrón de `run_nbfnet_pyg.py`.
2. **A\*Net llama `functional.generalized_rspmm` de torchdrug en 9 sitios de
   `reasoning/layer.py`** (líneas 55, 163-175, 334) y **no tiene rama alternativa**. Ese es el
   kernel CUDA JIT de torchdrug ⇒ **requiere `nvcc`**. NBFNet-PyG se salva porque su propio
   código define la salida (`layers.py:69-72`); **A\*Net no la tiene**.

**Verificado hoy que la premisa de la bitácora sigue siendo cierta**: no hay `nvcc`, no hay
module system, `CUDA_HOME` vacío. (Nota: la sección Entorno de `CLAUDE.md` menciona `module load
gnu12, cuda/12.6`; **`env.sh` ya no hace eso** y el cluster nuevo no tiene módulos — está
desactualizada.)

⇒ **Sin `nvcc`, A\*Net no se puede correr, punto.** La vía de desbloqueo es instalar el CUDA
toolkit por conda (`cuda-nvcc` 12.1, a juego con torch 2.1.0+cu121), idealmente en un env nuevo
para no tocar `attention`. **Beneficio doble**: además de habilitar A\*Net, permitiría compilar
`rspmm` para NBFNet-PyG ⇒ batch 64 entraría y FB237 sería bastante más rápido que las 56 h de
hoy. **No se intentó: requiere aprobación (cambia el entorno).**

**Decisión**

- **Épocas: 20**, las de sus configs, en los cuatro casos.
- **Presupuesto medido para NBFNet**: FB237 ~56 h/semilla, WN18RR ~10.6 h/semilla. n=3 semillas
  ⇒ **~7 días de FB237** y **~1.3 días de WN18RR**, paralelizables (4 jobs de QOS). El wall
  máximo de la partición AI es 3-12:00:00, así que **una semilla de FB237 cabe en un solo job**
  (pedir `--time=60:00:00`).
- **A\*Net queda bloqueado** hasta decidir sobre `nvcc`. Es baseline obligatorio para WWW ⇒ es
  decisión de camino crítico, no un detalle.
- Artefactos: `probe_nbfnet.py`, `nbfnet_pyg_cfg/{fb15k237,wn18rr}_trans.yaml` (idénticos a los
  suyos salvo `output_dir` y `root`, verificado con `diff`), datos transductivos descargados en
  `~/datasets/knowledge_graphs/{FB15k-237,WN18RR}` (FB237: 14 541 nodos / 272 115 train;
  WN18RR: 40 943 / 86 835 — coinciden con la literatura).

---

## 2026-08-19 — 🔁 **CAMBIO DE LOSS PRINCIPAL: BCE-k pasa a ser el objetivo primario y la CE de grafo completo queda como alternativa.** Implementado `--loss bce` bit a bit idéntico a la receta de NBFNet. Y A\*Net entra como baseline obligatorio + camino de escalamiento a 1M nodos

**Contexto**: al preparar la comparación transductiva quedó claro que **los cuatro papers de
referencia entrenan con BCE + k negativos muestreados** — NBFNet, ULTRA, A\*Net y TRIX — y que
nuestros números salen de CE de grafo completo, medida el 2026-08-08 (c) como **+0.167 de MRR**
sobre la misma arquitectura y los mismos datos. **Decisión del usuario: la BCE pasa a ser la
loss principal**, y la CE de grafo completo se reporta como alternativa, aunque implique
re-correr experimentos.

### Por qué la decisión es correcta (y no solo conservadora)

- **El riesgo que evita es de credibilidad, no de justicia.** Nadie objeta usar una receta de
  entrenamiento mejor si los baselines están igualados. Lo que sí pasa es que un revisor compara
  contra la tabla publicada, ve 0.52 donde la literatura dice 0.415, y sospecha diferencia de
  protocolo o bug. Con BCE-k estamos **sobre el protocolo publicado**, y las tablas son
  directamente comparables.
- **En TRANSDUCTIVO no hay motivo para esperar el desastre del 0.359.** Ese número es de
  **inductivo v2** (9 739 triples de train, best-valid en la época 2-4 ⇒ sobreajuste rápido).
  En transductivo, NBFNet mide **0.415** y A\*Net **0.411** de MRR full-filtered entrenando con
  BCE-32 sobre 272 115 triples. Es decir: **en el régimen que nos importa, BCE-k + MRR
  full-filtered es un protocolo coherente y sano.**
  ⚠️ Aplicando la regla vigente del proyecto: el +0.167 está medido en ind v2 y **no se hereda**
  al transductivo. Hay que medir el Δ de loss ahí, no asumirlo — y ese Δ es en sí un resultado.
- **La BCE NO ahorra cómputo en este harness.** La propagación calcula los N scores igual; los
  32 negativos solo se indexan. Re-correr es costo puro, sin compensación en velocidad. Dicho de
  otro modo: BCE-k nunca nos compró tiempo, solo nos costaba MRR.
- **Reportar las dos convierte el confounder en contribución**: *sobre la misma arquitectura,
  datos e implementación, la loss vale X de MRR* — el ablation que ninguno de los cuatro papers
  hizo.

### Riesgo a vigilar: la varianza

Con BCE-k nuestro NBFNet-PyG midió **σ = 0.046** (rango 0.307–0.394, n=3) contra **σ = 0.004**
con CE, y el best-valid cayó en la **época 2-4 de 20**. Si eso replica en transductivo, el
protocolo de medición se pone más caro (más semillas, model selection delicado). **Otra vez: es
un dato de ind v2, hay que re-medirlo, no heredarlo.**

### Implementación (HECHA): `--loss {ce,bce} --num_negative --adversarial_temperature`

Port fiel de `NBFNet-PyG/script/run.py:57-68` (loss) y `nbfnet/tasks.py:41-91` (muestreo), en
`train.py::GTLightningModule._bce_loss`:

- **Strict negative sampling**: los negativos nunca son cola verdadera de (h,r). El
  `filter_mask` que `train_collate_fn` ya construía desde `data.train_filters` (solo triplets
  de train) **es exactamente** su `strict_negative_mask` ⇒ no hubo que construir nada nuevo, y
  no hay fuga de val/test.
- Muestreo uniforme **con reemplazo** (su `rand * num_candidate` lo es; usamos `multinomial`).
- **Self-adversarial weighting de RotatE**: negativos pesados por `softmax(score_neg / T)` sin
  gradiente, `T = 0.5`; con `T <= 0`, peso uniforme `1/k` (su rama `else`).
- Defaults `num_negative=32`, `adversarial_temperature=0.5` — los de NBFNet.

**Verificación**

| qué | resultado |
|---|---|
| loss nuestra vs `run.py:57-68` literal, mismos negativos | **\|diff\| = 0.000e+00** (bit a bit) |
| ningún negativo es cola verdadera de train | 0 violaciones |
| gradiente finito y no-cero en el score | ✅ |
| `T=0` cae a peso uniforme y difiere de `T=0.5` | ✅ |
| subir el score del gold baja la loss | ✅ |
| end-to-end en ind v1 (1 ép, GPU): train + valid + test | ✅ `train_loss` 0.707 ≈ ln 2 |
| regresión de la rama CE (default sin tocar) | ✅ `train_loss` 7.42 ≈ ln(1594) |

Las dos losses arrancan exactamente en su valor teórico de inicialización, que es el chequeo
más barato de que ninguna está mal normalizada.

⚠️ **El default de `--loss` quedó en `ce`**, no en `bce`, **solo por compatibilidad**: hay ~40
`sbatch_*.sh` y `run_*.sh` en el repo y cambiar el default los haría cambiar de objetivo en
silencio — el modo de fallo que esta bitácora ya corrigió tres veces. **Todos los scripts
nuevos del paper WWW deben pasar `--loss bce` EXPLÍCITAMENTE**, baselines incluidos, para que
además quede registrado en el log de cada corrida.

### A\*Net (leído hoy de `resumen_papers_KGC.md`) — dos consecuencias

1. **Es baseline obligatorio para un paper transductivo en WWW.** FB15k-237 MRR **0.411**,
   WN18RR **0.549** — comparable a NBFNet propagando solo el 10 % de nodos y aristas. Es *el*
   método path-based escalable; un revisor de WWW va a preguntar por él.
2. **Es la respuesta al escalamiento a >1M nodos, y confirma que el cuello NO es la loss.** En
   ogbl-wikikg2 (2.5 M entidades, 16 M triples) **NBFNet falla por OOM incluso con batch 1**;
   A\*Net propaga el **0.2 %** de nodos/aristas y logra SOTA (MRR 0.6767). Nuestro GT disperso
   tiene la misma estructura de costo que NBFNet (`O(B·E·d)` por query, sin cómputo compartido
   entre queries) ⇒ **hereda exactamente el mismo muro**. La poda por query es una línea de
   trabajo propia, no un ajuste de la loss.
3. **Corroboración externa de la lista negra #8, y conviene citarla.** Su ablation de la función
   de prioridad mide: neuronal **0.411** > **Random 0.378** > Degree 0.347 > PPR 0.266. O sea,
   seleccionar subgrafos **al azar** rinde claramente peor que seleccionarlos de forma aprendida
   y condicionada a la query — evidencia independiente, de otro grupo y en otro setting, de que
   la conectividad aleatoria no aporta en KGC. Es justo el argumento que sostiene nuestro
   resultado del expander ("una arista aleatoria es un no-hecho").

**Decisión**

- **BCE-k es la loss principal del paper WWW; CE de grafo completo se reporta como alternativa**
  en una tabla de ablation de loss. Ambos brazos, y **los baselines con la misma loss**.
- **Todos los resultados previos del proyecto son con CE de grafo completo** y quedan como la
  columna "alternativa". No se borran ni se re-etiquetan.
- **A\*Net se agrega a la lista de baselines** junto a NBFNet.
- Artefactos: `train.py::{_bce_loss,training_step}` y los 3 flags nuevos; smoke test en el
  scratchpad de la sesión.

---

## 2026-08-18 — ✅ **DEFENSA DE CANDIDATURA APROBADA** ⇒ **cambio de prioridad: manda el TRANSDUCTIVO, con paper a ACM WWW (deadline primera semana de octubre de 2026). El inductivo se pospone hasta después del envío.**

**Contexto**: el usuario rindió hoy la defensa de candidatura y **salió aprobada**. En esa
instancia se acordó reorientar la prioridad del proyecto. Esta entrada registra el acuerdo y
hace inventario de lo que hay en transductivo, que es el régimen que pasa a mandar.

### El acuerdo

| | |
|---|---|
| **Régimen prioritario** | **TRANSDUCTIVO** |
| **Entregable** | publicación enviada a **ACM WWW** |
| **Deadline** | **primera semana de octubre de 2026** (~6 semanas desde hoy) |
| **Inductivo** | **pospuesto — se retoma DESPUÉS del envío** |

Registrado también en el bloque de prioridad de `CLAUDE.md` y `GOALS.md`. **La pregunta
central de `GOALS.md` ("¿el full attention supera a NBFNet en ind v1?") queda como registro
de la etapa anterior y deja de fijar la prioridad de cómputo.** El marco de la tesis de la
entrada 2026-08-08 (f) no cambia; lo que cambia es qué régimen se ataca primero.

### Inventario transductivo (todo lo que hay, al 2026-08-18)

Config transductiva estándar del proyecto: **L4 / dim 32** (elegida por costo el 2026-07-20:
L6/dim64 cuesta 2.4× por época y 2.7× en memoria para comprar +0.006 de MRR). ⚠️ **No
confundir con el inductivo, que corre L6 / dim 64.** Todo n=1, 20 épocas, seed 42.

| dataset | brazo | batch global | valid | **test_mrr** | H@10 | MR |
|---|---|---:|---:|---:|---:|---:|
| FB15k-237 | **sparse + `--remove_one_hop`** | 8 | 0.4335 | **0.42893** | 0.606 | 130.0 |
| FB15k-237 | sparse softmax L6/d64 (2.4× más caro) | 16 | 0.407 | 0.4028 | 0.595 | 116.4 |
| FB15k-237 | sparse_exp deg3 | 8 | 0.4012 | 0.3973 | 0.589 | 126.8 |
| FB15k-237 | sparse softmax (baseline directo) | 8 | 0.402 | 0.3965 | 0.591 | 128.4 |
| FB15k-237 | sparse `--attn degree` | 32 | 0.4023 | 0.3957 | 0.592 | 126.1 |
| FB15k-237 | **NBFNet** | — | *0.376 (ép 1/20)* | **SIN MEDIR** | — | — |
| WN18RR | sparse_exp deg3 | 32 | 0.5531 | **0.5566** | 0.647 | 1497 |
| WN18RR | sparse softmax | 32 | 0.5481 | 0.5536 | 0.644 | 1616 |
| WN18RR | sparse `--attn degree` | 32 | 0.5468 | 0.5533 | 0.642 | 1610 |
| WN18RR | **NBFNet** | — | — | **NO CORRIDO** | — | — |

### Qué falta, en orden de bloqueo

1. 🚩 **BLOQUEADOR #1 — NBFNet transductivo, en los dos datasets.** No existe baseline propio
   en **ninguno**. Sin él el 0.42893 no es interpretable, y **el ~0.415 de literatura no sirve
   como techo**: este harness corre NBFNet 0.011–0.016 **por encima** de las re-evaluaciones
   publicadas (ver 2026-08-08 (c)), así que el baseline propio podría estar en ~0.43 y el
   "supera a NBFNet" evaporarse. FB15k-237 tiene checkpoint **reanudable** en la época 1/20
   (`experiments/nbfnet_trans_fb237/last.ckpt`, `sbatch_nbfnet_trans.sh` ya encadenado,
   ~2.5 h/época × 18 restantes); WN18RR hay que lanzarlo de cero y es **barato** (~14 min/época).
2. **El 0.42893 es n=1 y sin brazo de control con el código de hoy.** El protocolo vigente
   pide n≥3; con σ del sparse en 0.019–0.054 un |Δ| < ~0.05 no es distinguible de ruido, y el
   +0.0324 sobre su baseline directo cae justo en esa zona. `sbatch sbatch_onehop_trans.sh
   control` está implementado y **sin lanzar**.
3. **`--edge_drop` NUNCA se probó en transductivo.** Es el **único efecto positivo del
   proyecto que sobrevivió una réplica con barras de error** (+0.057 ± 0.013 en ind v1,
   +0.060 ± 0.014 en ind v2, t>4 en ambos; exactamente 0.000 en WN18RR). Y `--remove_one_hop`
   es la versión **dirigida** de la misma idea (ver 2026-08-08 (d)) ⇒ **hay que medirlos como
   brazos separados y combinado**, porque es probable que sean parcialmente redundantes.
   Barato en comparación con lo demás y directamente en el camino del paper.
4. **Explicar la asimetría de `remove_one_hop`** (mejora fuerte en transductivo denso,
   −0.236 en ind v1 con el control ya hecho ⇒ el efecto es real). Es material de paper: la
   hipótesis registrada es que en un grafo denso casi siempre hay camino alternativo y quitar
   el atajo fuerza composición, mientras que en uno disperso destruye la evidencia.

### Notas de planificación (el tiempo es el recurso escaso)

- **Las corridas transductivas cuestan DÍAS, no horas**: FB15k-237 sparse ~2 h/época a batch
  global 8 (~40 h las 20 épocas); NBFNet FB15k-237 ~45–63 h; WN18RR sparse_exp llegó a 38
  min/época y **25.7 GB/GPU** (necesitó DDP para no dar OOM en la A100-40GB). Con **6 semanas
  y 4 jobs simultáneos de QOS**, la cola de arriba es del orden de lo que cabe — no hay margen
  para barridos exploratorios grandes ni para gastar GPU en inductivo.
- Vale para el transductivo el mismo protocolo (n≥3, ≥50 ép, media ± sd). ⚠️ Ojo: **20 épocas
  es lo que se usó en todo el transductivo hasta ahora**; el subentrenamiento a 20 ép está
  medido como propiedad **del sparse en ind v1**, y por la regla vigente (ninguna propiedad se
  hereda entre brazos ni splits sin re-medirla) **no se sabe si aplica acá**.
- Lo que **NO** hay que volver a probar en transductivo, ya cerrado ahí: expander (Δ +0.0008
  FB237, +0.0030 WN18RR ⇒ lista negra #8), `--attn degree` (−0.0002 en WN18RR
  apples-to-apples), `--rel_param lowrank` (−0.0001 / −0.0004 a épocas igualadas).

**Decisión**

- **Prioridad registrada en `CLAUDE.md` (bloque al inicio) y `GOALS.md` (sección al inicio)**
  para que ninguna sesión futura vuelva a priorizar el inductivo por leer la pregunta central
  vieja.
- **Nada lanzado todavía en esta sesión** — esta entrada es el acuerdo y el inventario, no
  resultados.

---

## 2026-08-10 (b) — `--rel_param lowrank` **CERRADO: nulo en 4 de 5 regímenes**. Se cancelan los dos jobs transductivos (~230 GPU-h ahorradas)

**Contexto**: cierra la línea abierta en la entrada (e) del 2026-08-08. Tras la no-réplica en v2,
se midió en WN18RR ind v1 (job 91488, n=3) y se comparó la curva de valid en transductivo a
épocas igualadas (jobs 91478/91479 contra 91458).

### Cuadro completo de `lowrank k=8`

| régimen | modelo | diag | lowrank k=8 | **Δ (Welch)** |
|---|---|---:|---:|---:|
| **FB15k-237 ind v1** | full (n=3) | 0.3641 ± 0.0189 | 0.4164 ± 0.0115 | **+0.052 (t=4.09)** ✅ |
| FB15k-237 ind v1 | sparse+edrop (n=3) | 0.3689 ± 0.0365 | 0.4218 ± 0.0076 | +0.053 (t=2.46) ✅ |
| FB15k-237 ind v2 | sparse+edrop (n=3) | 0.5157 ± 0.0088 | 0.5191 ± 0.0044 | +0.003 (t=0.59) ❌ |
| **WN18RR ind v1** | sparse+edrop (n=3) | 0.7339 ± 0.0051 | 0.7384 ± 0.0019 | **+0.0045 (t=1.42)** ❌ |
| **WN18RR ind v1** | sparse_exp+edrop (n=3) | 0.7277 ± 0.0092 | 0.7323 ± 0.0030 | **+0.0046 (t=0.82)** ❌ |
| **FB15k-237 transductivo** | sparse+onehop (n=1) | — | — | **−0.0001** ❌ |
| **FB15k-237 transductivo** | sparse_exp+onehop (n=1) | — | — | **−0.0004** ❌ |

**Uno de cinco regímenes.** Y ese uno (v1) es donde el sparse está **más lejos** de NBFNet.

### Evidencia transductiva: curvas de valid a épocas igualadas

| época | 91458 (onehop) | 91478 (+lr8) | Δ | 91479 (exp+lr8) | Δ |
|---|---:|---:|---:|---:|---:|
| 2 | 0.41870 | 0.42178 | +0.0031 | 0.41712 | −0.0016 |
| 4 | 0.42184 | 0.42178 | **−0.0001** | 0.42142 | **−0.0004** |
| 7 | 0.42184 | 0.42178 | −0.0001 | 0.42142 | −0.0004 |

Los tres brazos quedan pegados **en la cuarta cifra decimal**, y los dos con `lowrank` **por
debajo**. Además ninguno de los dos mejoró su valid desde la época 4 (3-4 épocas planos),
mientras 91458 siguió subiendo hasta **0.43346** en la época 19.

**Análisis**

- **(A) `lowrank` compra margen SOLO donde el modelo está lejos del techo.** Funciona en v1
  (sparse 0.397 vs NBFNet 0.458, brecha 0.061) y es nulo en v2 (0.5157 vs 0.5245, ya empata),
  en WN18RR (0.7339 vs 0.7398, a 0.006) y en transductivo. **No es una propiedad de la
  arquitectura**: es margen que existe donde hay margen. Refuerza el punto (C) de la entrada
  (h): a parámetros igualados NBFNet seguía ganando en v1.
- **(B) Se une a LapPE y a sparse+RWSE como efecto dependiente de split.** Tercer caso del
  proyecto. La diferencia es que esta vez se detectó **en la réplica**, no meses después.
- **(C) Matiz que conviene no perder**: en WN18RR el mejor brazo absoluto de la tabla es
  `sparse+edrop+lowrank` con **0.7384 ± 0.0019** — a **−0.0014** de NBFNet, con **H@1 (0.691) y
  H@10 (0.828) mejores** que él, y la **σ más baja de toda la tabla**. No es significativo
  contra su propio `diag`, pero como número absoluto es el más fuerte que tiene el proyecto en
  ese split.

**Decisión**

- **`--rel_param lowrank` NO se adopta.** Queda en el código, documentado y optimizado
  (entradas (k) y 2026-08-09), pero fuera de la best config.
  ⇒ **POSDATA 2026-08-21 (b)**: al leer el código de NBFNet/A\*Net/ULTRA aparece que esta línea
  atacaba el **eje equivocado**. La capacidad relacional de ellos se aplica **una vez por
  query** (`Linear(d, R·d)(q)`), no por arista; por eso NBFNet puede dar 26× más capacidad por
  relación y ser 9× más rápido que nuestro full attention. `lowrank` era la versión **cara** de
  la misma idea (1.05× vs 2.33× de sobrecosto, medido). El nulo en 4 de 5 regímenes sigue en
  pie como resultado; lo que cambia es que **no refuta la hipótesis de capacidad relacional**,
  solo la implementación que se probó.
- **Cancelados los jobs 91478 (`onehop+lr8`) y 91479 (`exp_onehop+lr8`)**: iban en las épocas
  6/20 y 5/20 y necesitaban ~50 h y ~64 h más a 2 GPUs ⇒ **~230 GPU-horas** para confirmar un
  nulo ya visible en cinco regímenes. **Los checkpoints se conservan** (`epoch=2` y `epoch=3` +
  `last.ckpt`); `sbatch_onehop_trans.sh` es encadenable si alguna vez se retoman.
- **Ninguno de los dos estaba en el camino crítico del paper transductivo.** Lo que sí lo está y
  sigue **sin lanzar**: (1) **NBFNet transductivo** (~63 h, checkpoint reanudable en la época
  1/20) — el bloqueador para comparar contra el estado del arte; (2) el **brazo de control** de
  `remove_one_hop` con el código de hoy y en DDP; (3) **semillas adicionales** de ese brazo.
- Artefactos: `logs/relparam_wn18rr_v1_91488.log`, `experiments/relparam_wn18rr_v1_*`.

---

## 2026-08-10 (a) — TABLA COMPLETA de WN18RR ind v1 (7 brazos, n=3) · full attention en v2 rescatado de un job cancelado · `remove_one_hop` transductivo **CERRADO: test 0.4289 en 20 épocas, +0.032 sobre el baseline** · documento `RESULTADOS_DEFENSA.md`

### 1. WN18RR inductivo v1 — 50 épocas, n=3, L6/dim64, 1 GPU

| brazo | **MRR** | H@1 | H@3 | H@10 | MR | s/ép |
|---|---:|---:|---:|---:|---:|---:|
| *NBFNet (n=1, 20 ép)* | *0.7398* | *0.689* | ***0.774*** | *0.822* | *29.6* | — |
| **sparse + edrop + lowrank k8** | **0.7384 ± 0.0019** | **0.691** | 0.757 | **0.828** | 31.8 | 22 |
| sparse | 0.7339 ± 0.0072 | 0.681 | 0.763 | **0.828** | **29.1** | 19 |
| sparse + edrop 0.2 | 0.7339 ± 0.0051 | 0.683 | 0.762 | 0.823 | 35.6 | 18 |
| sparse_exp + edrop + lowrank k8 | 0.7323 ± 0.0030 | 0.685 | 0.751 | 0.821 | 32.9 | 39 |
| sparse_exp deg3 | 0.7317 ± 0.0069 | 0.678 | 0.762 | 0.824 | 32.9 | 36 |
| sparse_exp + edrop 0.2 | 0.7277 ± 0.0092 | 0.683 | 0.745 | 0.815 | 36.4 | 35 |
| *sparse_exp `ultra` ed=0.0 / 0.2* | *0.7196 / 0.7036* | — | — | — | *68.7 / 79.4* | *37* |
| *full attention (n=1, 20 ép)* | *0.673* | *0.638* | *0.691* | *0.739* | *97.4* | — |

Lecturas: el **edge dropout es exactamente 0.000** acá (0.7339 vs 0.7339), confirmado ahora con
las cuatro métricas · el **expander resta** (−0.002 sin edrop, −0.006 con) · **`ultra` es lo más
dañino** y dispara el MR a 68.7-79.4 contra 29-36 · el **full attention es el peor** (−0.067).

### 2. Full attention en FB15k-237 ind v2 — rescatado de un job cancelado (n=1)

| | MRR | H@1 | H@3 | H@10 | MR |
|---|---:|---:|---:|---:|---:|
| **full (ckpt ép 47/50, n=1)** | **0.4973** | 0.406 | 0.545 | 0.664 | 101.3 |
| *registro histórico (n=1, 20 ép)* | *0.491* | *0.389* | *0.544* | *0.686* | *76.6* |

El job 91480 se canceló a las 9 h; se evaluó su mejor checkpoint antes de perderlo. **Δ +0.006
contra el histórico de 20 ép ⇒ el full tampoco estaba subentrenado en v2** (igual que en v1).
Nota: el checkpoint se llama `epoch=47` pero **el mejor valid fue en la época 12** (0.45295) y no
mejoró en 35 épocas ⇒ el modelo ya había convergido; el número es sólido pese a no completar 50.

⇒ **El full es el peor de los cuatro modelos en los DOS splits de FB15k-237**: −0.081 bajo NBFNet
en v1, −0.027 en v2. Coherente con la entrada (g): manda el 99.4 % de su atención a pares sin
relación.

### 3. `remove_one_hop` transductivo (job 91458) — evaluaciones sucesivas del mejor checkpoint

| checkpoint | valid | **test_mrr** | H@1 | H@3 | H@10 | MR |
|---|---:|---:|---:|---:|---:|---:|
| época 2/20 | 0.4187 | 0.4143 | 0.325 | 0.451 | 0.591 | 146.0 |
| época 4/20 | 0.4218 | 0.4169 | 0.328 | 0.455 | 0.589 | 143.5 |
| época 10/20 | 0.4330 | 0.4288 | 0.338 | 0.468 | 0.607 | 130.0 |
| **FINAL (20/20, best ép 13)** | **0.43346** | **0.42893** | **0.338** | **0.468** | **0.606** | **130.0** |
| *baseline sparse L4/d32 batch 8 (ép 19/20)* | *0.402* | *0.3965* | *0.300* | *0.437* | *0.591* | *128.4* |

**RESULTADO FINAL: test_mrr 0.42893** (20/20 épocas completas, best-valid en la época 13,
`trainer.test` automático sobre el mejor checkpoint) ⇒ **+0.0324 sobre su baseline directo**
(0.3965, misma config y mismo batch global 8) y **+0.0261** sobre la config grande L6/dim64 que
cuesta 2.4× más (0.4028). Las **cuatro métricas de ranking superan al baseline a la vez**
(H@1 +0.038, H@3 +0.031, H@10 +0.015) y el **MR se recuperó solo** a lo largo del entrenamiento
(146.0 → 143.5 → 130.0): en las primeras épocas subía la cabeza del ranking y empeoraba la cola,
y al converger quedó a 1.6 puntos del baseline (130.0 vs 128.4).
El best-valid cae en la **época 13 de 20** ⇒ convergido, no cortado por el tope de épocas.

⚠️ Sigue siendo **n=1**, sin brazo de control con el código de hoy, y **superar el ~0.415 de
literatura de NBFNet NO es superar a NBFNet** (nuestro harness lo corre 0.011-0.016 por encima de
las re-evaluaciones publicadas ⇒ el baseline propio podría estar en ~0.43).

**Decisión / artefactos**

- Creado **`RESULTADOS_DEFENSA.md`**: 12 tablas ordenadas por importancia, con MRR/H@1/H@3/H@10,
  MR, capas/dim, GPUs y s/época, **extraídas de los logs** (no copiadas de esta bitácora) para no
  arrastrar errores de transcripción. Separa arriba lo que tiene n≥3 y 50 ép, abajo lo n=1.
- **Patrón operacional confirmado dos veces**: se puede evaluar el mejor checkpoint de un job en
  curso —o de uno cancelado— con `--eval_only --eval_ckpt`, con `srun --overlap` (sin gastar slot
  de QOS) o en GPU dedicada. Rescató 9 h de cómputo del 91480.
- **Correcciones a scripts**: `sbatch_relparam_v1.sh` tenía la ruta de FB15k-237 fija (no podía
  correr WN18RR) y **no pasaba `--exp_degree`**, así que `sparse_exp` habría corrido con el
  default **4** en vez del 3 de todo el proyecto — habría hecho la fila incomparable.
  `sbatch_onehop_trans.sh` pasó a brazos **componibles** (`exp_onehop_lr8`, etc.).
- Artefactos: `RESULTADOS_DEFENSA.md`, `logs/relparam_wn18rr_v1_91488.log`,
  `logs/onehop_trans_eval_ep{2,4,10}.log`, `experiments/edrop_v2_rfat_p00_ep50_s42/`.

---

## 2026-08-09 (b) — `--rel_param lowrank` **NO REPLICA en v2**: el +0.052 (t=4.09) de ind v1 se cae a +0.003 (t=0.59) en ind v2 ⇒ **mismo destino que LapPE**. Y queda CERRADO el contraste del full en v1 con control de código propio

**Contexto**: cierra el barrido lanzado en la entrada (e) y marcado como provisional en la (h).
Jobs 91460 (full en v1: `diag` y `lr4`, completo) y 91466 (v2: `sparse` completo, `sparse_exp`
**detenido** por costo — ver abajo).

### 1. v1, FULL attention — contraste CERRADO con control del código de hoy

| ind v1, 50 ép, n=3 | **test_mrr** | H@1 | H@10 | MR | best ép |
|---|---:|---:|---:|---:|---:|
| diag (control de hoy) | 0.3641 ± 0.0189 | 0.308 | 0.448 | 193.2 | 47.0 |
| lowrank **k=4** | 0.4148 ± 0.0125 | 0.353 | 0.515 | 184.9 | 37.7 |
| lowrank **k=8** | 0.4164 ± 0.0115 | 0.356 | 0.509 | 170.5 | 47.3 |

**k=4: +0.0507 ± 0.0131 (t=3.87)** · **k=8: +0.0523 ± 0.0128 (t=4.09)**. El efecto en v1 **ya no
depende del control histórico** (esto corrige el "⚠️ provisional" de la entrada (h)), es
**monótono y saturante**: casi todo el salto está en k=4.

### 2. v2, SPARSE — **el efecto NO replica**

| ind v2, 50 ép, n=3 | **test_mrr** | H@1 | H@10 | MR |
|---|---:|---:|---:|---:|
| sparse + edrop, lowrank **k=8** | 0.5191 ± 0.0044 | 0.419 | 0.697 | 82.6 |
| sparse + edrop, `diag` | 0.5157 ± 0.0088 | 0.417 | 0.694 | 79.9 |
| *NBFNet* | *0.5245 ± 0.0042* | *0.423* | *0.705* | *49.9* |

**+0.0034 ± 0.0057 (t=0.59) ⇒ indistinguible de cero.** Contra **+0.052 (t=4.09)** en v1.

### 3. v2, SPARSE_EXP — parcial (n=1, época 26 de 50), corrida DETENIDA

| | test_mrr | H@1 | H@3 | H@10 | MR |
|---|---:|---:|---:|---:|---:|
| sparse_exp + edrop + lr8, **ép 26/50, n=1** | 0.5234 | 0.426 | 0.586 | 0.696 | 78.3 |
| *sparse_exp + edrop, `diag` (n=3, 50 ép)* | *0.5275 ± 0.0032* | *0.426* | *0.590* | *0.706* | *52.5* |

Evaluada sobre el mejor checkpoint antes de cancelar. **No sugiere mejora** — está por debajo
del `diag` completo, aunque es n=1 y sin converger (el brazo `diag` de v2 llegaba a su mejor
valid en las épocas 39-48, así que a la 26 todavía subiría).

**Análisis**

- **(A) LA HIPÓTESIS DE CAPACIDAD RELACIONAL NO SOBREVIVE EL CAMBIO DE SPLIT.** Es exactamente
  el patrón de **LapPE** (+0.035 en v1 → +0.005 en v2, entrada 2026-07-08) y de **sparse+RWSE**
  (que invertía el signo). El proyecto ya se quemó dos veces con efectos de un solo split; éste
  es el tercero. **En v1 el efecto es grande y significativo (t=4.09); en v2 es cero (t=0.59).**
- **(B) Por qué es coherente que muera en v2, y refuerza la lectura de fondo.** En v1 el sparse
  está **lejos** de NBFNet (0.397 vs 0.458) y ahí más capacidad relacional compra 0.05; en v2 ya
  lo **iguala** (0.5157 vs 0.5245, t=−1.56) y no queda nada que comprar. O sea la capacidad
  relacional no era el cuello de botella **de la arquitectura**, sino un margen que solo existe
  donde el modelo está lejos del techo. Consistente con el punto (C) de la entrada (h): a
  parámetros igualados NBFNet seguía ganando.
- **(C) Lo que SÍ queda establecido**: en v1, `lowrank` es la única intervención del proyecto
  que mueve al **full attention** (+0.052, t=4.09, con control propio), y lleva al sparse a
  **0.4218**, el mejor número de atención registrado en ese split. Es un resultado de v1, no una
  propiedad general.

**Decisión**

- **NO adoptar `--rel_param lowrank` como parte de la best config.** Queda como efecto
  **dependiente de split**, junto a LapPE y sparse+RWSE. ⇒ **CONFIRMADO 2026-08-10 (b)**: nulo
  también en WN18RR ind v1 (t=1.42 y t=0.82) y en transductivo (−0.0001 / −0.0004 a épocas
  igualadas) ⇒ **1 de 5 regímenes**. Línea cerrada. La entrada (e) escribió la predicción
  falsable ("si es plano dentro del ruido, la hipótesis de capacidad queda refutada"); **en v2 es
  plano**. El flag se conserva en el código con su documentación y la optimización de la entrada
  (k), pero no se usa por defecto.
- **Corrida detenida por costo**: el brazo `sparse_exp` de v2 iba a 3.96 it/s ⇒ ~2.6 h por
  semilla ⇒ ~6 h más, para completar un contraste cuya versión `sparse` ya dio cero y cuyo
  parcial n=1 tampoco sugiere mejora. No se justifica.
  ⚠️ **Nota de mezcla**: las corridas de 91466 arrancaron con la implementación VIEJA de
  `lowrank`; como el sbatch lanza un proceso por corrida, las últimas ya tomaron la optimizada.
  Los **resultados son mezclables** (bit a bit idénticos), **los tiempos no**.
- **Nota de proceso**: se reportó al usuario "faltan ~40 min" para el brazo `sparse` y no se
  volvió a verificar en ~5 h. La estimación salió de los 19.3 it/s del modo `diag`, pero
  `lowrank k=8` en la implementación vieja era mucho más lento. **Estimar el ETA con el ritmo
  del brazo que se está corriendo, no con el del control.**
- Artefactos: `logs/relparam_v1_91460.log`, `logs/relparam_v2_91466.log`,
  `experiments/relparam_v{1,2}_*`.

---

## 2026-08-09 — ⚡ **ENCONTRADA la causa del sobrecosto del expander: no era el expander, era `table[:, rel]`**. El profiler la ubica en el BACKWARD de la indexación avanzada; cambiar a `index_select` da **2.08× en sparse_exp** y baja el sobrecosto del expander de **1.76× a 1.08×**. Bit a bit idéntico.

**Contexto**: la entrada (l) dejó el sobrecosto 1.75× del expander **sin explicar** tras descartar
la hipótesis de cache misses, y anotó como decisión "antes de cualquier próximo intento de
optimización, correr un profiler". Se hizo eso. El usuario necesitaba correr
`sparse_exp + remove_one_hop + rel_param k=8` en FB15k-237 transductivo, que a la velocidad
anterior daba ~6.5 días.

### Paso 1 — profiler (job 91473, FB15k-237 transductivo, L4/d32, batch 4)

| op | sparse | sparse_exp | Δ | % del extra |
|----|-------:|-----------:|--:|------------:|
| `aten::_index_put_impl_` | 90.4 ms | **209.1 ms** | +118.7 | **49.8 %** |
| `indexing_backward` (kernel) | 77.9 ms | **160.2 ms** | +82.3 | **34.5 %** |
| `indexing_backward` (2º) | 11.1 ms | 47.5 ms | +36.4 | **15.2 %** |
| todo lo demás (mul, sum, scatter, index_add…) | — | — | ~+3 | ~1 % |
| **total CUDA** | **357.9 ms** | **596.4 ms** | | **1.67×** |

**El 99.5 % del sobrecosto está en el BACKWARD de los gathers por indexación avanzada.** Y el
contraste que da la pista: `aten::index_add_` cuesta **5.06 → 5.84 ms** (lineal, casi gratis)
mientras `_index_put_impl_` cuesta **90 → 209 ms**. Son la misma operación lógica por caminos
distintos: `x[:, idx]` ⇒ backward `index_put_(accumulate=True)`; `x.index_select(1, idx)` ⇒
backward `index_add_`.

### Paso 2 — aislar la causa (job 91475)

El backward de `table[:, rel]` acumula un gradiente **por arista** en la fila de su relación,
con atómicos. Las **6·N = 87 030 aristas expander comparten TODAS el índice `R_exp`** ⇒ 87 mil
acumulaciones atómicas sobre **UNA sola fila**:

| variante (gather de `rel_value`, fwd+bwd) | ms/iter |
|---|---:|
| solo aristas reales (E=573 240) | 2.77 |
| real + expander (87 K aristas con el mismo índice) | **31.43** (11.35×) |
| real por gather + expander por `expand` | 2.58 (0.93×) |

### Paso 3 — el arreglo y su atribución (job 91477)

Se probaron tres variantes end-to-end. **Todo el efecto viene de `index_select`; el broadcast
de los tramos de relación única aporta 0** (por eso NO se conservó).

| FB15k-237 transductivo, A100 | `x[:, rel]` | `index_select` | + broadcast | **total** |
|---|---:|---:|---:|---:|
| **sparse_exp deg3 + lowrank k=8** | 1.39 it/s | **2.58** | 2.59 | **1.87×** |
| sparse_exp deg3 | 3.33 it/s | **6.83** | 6.92 | **2.08×** |
| sparse | 5.87 it/s | **7.52** | 7.49 | **1.28×** |
| WN18RR transductivo, sparse | 1.93 it/s | 2.23 | 2.24 | 1.16× |
| FB15k-237 ind v1, sparse | 61.77 it/s | 75.35 | 73.86 | 1.20× |

**EFECTO PRINCIPAL: el sobrecosto del expander cae de 1.76× (5.87/3.33) a 1.08× (7.49/6.92).**
Prácticamente deja de costar.

**Verificación**: `equal=True`, `max|diff| = 0.0e+00` en `sparse`, `sparse + attn=rel`,
`sparse_exp` y `sparse_exp + lowrank` ⇒ **bit a bit idéntico**, no solo dentro de tolerancia.
fwd+bwd OK en los 5 modelos.

**Análisis**

- **(A) CIERRA la causa que la entrada (l) dejó abierta, y la respuesta no tenía nada que ver
  con el expander.** No era localidad de memoria, ni el grafo aleatorio, ni el conteo de
  aristas: era un **gotcha de PyTorch**. El expander solo lo hacía visible, porque concentra
  87 K aristas en un único índice de relación. La explicación de julio ("cache misses por la
  aleatoriedad") queda definitivamente descartada.
- **(B) También acelera todo lo demás.** `sparse` gana 1.28× y el inductivo v1 1.20 % — porque
  los **N self-loops** sufren la misma patología en menor escala (todos comparten `self_rel`).
- **(C) ⚠️ El speedup end-to-end EN EL JOB REAL es menor que el del benchmark aislado.** Job
  91469 (código viejo, 2 GPUs DDP) medía **2.45 it/s**; job 91478 (código nuevo, misma config)
  mide **2.88 it/s** ⇒ **1.18×**, no el 1.87× del benchmark de 1 GPU. Época: 4.35 h → 3.7 h;
  20 épocas: ~87 h → **~74 h**. **No tengo explicación completa de la diferencia** (candidatos:
  overhead de DDP que no encoge, data loading, u otro cuello que pasa a dominar cuando el
  gather deja de serlo). **Citar el 1.18×, no el 1.87×, para planificar corridas DDP.**
- **(D) Cuarta iteración de la misma lección, ahora con el método correcto.** Esta optimización
  es exactamente la que la entrada (l) descartó estimándola en **~1 %** — la estimé sobre el
  tráfico del *forward* cuando el costo estaba en el *backward*. Las tres estimaciones por
  inspección de esa entrada fallaron; **el profiler encontró la causa en veinte minutos**. La
  regla de la entrada (k) queda confirmada por la vía dura.

**Decisión**

- ⚠️ **REGLA: los lookups relacionales van con `index_select`, NUNCA con `table[:, rel]`.** Está
  documentado con las mediciones en `SparseRelationalAttentionLayer._rel_lookup`, con aviso
  explícito de no "simplificarlo" de vuelta. Cuesta hasta 2×.
- **NO se conservan** el broadcast de tramos de relación única (1.00×) ni el ordenamiento por
  `dst` de la entrada (l) (1.00×).
- **Los tiempos previos al 2026-08-09 no son comparables**; los resultados sí (bit a bit).
- Relanzado el brazo combinado transductivo con el código nuevo: **job 91478**
  (`--remove_one_hop --rel_param lowrank --rel_rank 8`), ~74 h. Se canceló 91469 sin pérdida
  (no había completado la época 0).
- **Pendiente**: el mismo patrón `x[:, idx]` existe en los gathers de NODO (`q[:, dst]`,
  `k[:, src]`, `v[:, src]`) y en el full attention (`attn[:, :, dst, src]`, doble indexación).
  Ahí `index_select` medido aislado dio **0.84–0.91×** (peor), así que **no** se cambió — pero
  el full attention nunca se perfiló y podría tener su propia versión del problema.
- Artefactos: `prof_exp.sh`, `bench_gather.sh`, `bench_relgather.sh`, `bench_attrib.sh`,
  `logs/prof_exp_91473.log`, `logs/bench_relg_91475.log`, `logs/bench_attrib_91477.log`.

---

## 2026-08-08 (l) — INTENTO FALLIDO de optimizar el costo del expander: ordenar aristas por `dst` da **1.00×** (medido en 5 casos, revertido) ⇒ **la causa mecánica registrada el 2026-07-20 ("cache misses por la aleatoriedad") NO se sostiene**; el sobrecosto 1.75× es real pero queda SIN EXPLICAR

**Contexto**: pregunta del usuario sobre si el costo del expander se puede optimizar. Se
propusieron tres cambios; se implementó y midió el más prometedor, y los otros dos se
descartaron cuantificándolos antes de escribir código. **Se registran los tres para que nadie
—incluido un asistente futuro— los vuelva a proponer.**

### Intento 1 (implementado y medido): ordenar el edge_index por `dst`

**Hipótesis**: las capas están dominadas por accesos indexados y **`dst` aparece 5 veces por
capa** (`q[:, dst]`, `scatter_reduce_(amax)`, el `gather` del max, el `index_add_` del
denominador y el `index_add_` del mensaje) contra 2 de `src`. El `edge_index` de `src/data.py`
**no viene ordenado** (verificado en ind v1 y transductivo) ⇒ ordenar por `dst` debería
coalescer esos accesos y reducir conflictos atómicos. Implementado como `sort_edges_by_dst`,
una vez por forward, con guard para no romper `exp_typing` ultra/path (que exigen que las
expander sean el tramo final contiguo). **Equivalencia verificada**: max|diff| 2–3e−07 en los
5 modelos (puro orden de suma en punto flotante).

**Resultado (A100, benchmark con warmup, 40 iters, mismo grafo y config):**

| caso | sin ordenar | ordenado | speedup |
|------|------------:|---------:|--------:|
| WN18RR TRANSDUCTIVO, sparse | 1.93 it/s | 1.95 it/s | **1.01×** |
| FB15k-237 transductivo, sparse | 5.84 it/s | 5.82 it/s | 1.00× |
| FB15k-237 transductivo, sparse_exp | 3.34 it/s | 3.28 it/s | 0.98× |
| FB15k-237 transductivo, sparse_exp + lr8 | 1.38 it/s | 1.38 it/s | 1.00× |
| FB15k-237 ind v1, sparse | 61.80 it/s | 61.93 it/s | 1.00× |

**REVERTIDO** (helper y los 4 call sites eliminados; forward re-verificado en los 3 modelos).

**Por qué no funciona**: los tensores de nodo **entran en la L2 de la A100 (40 MB)**, así que
el orden de acceso da igual — todos los gathers pegan en cache igual.

| dataset | q/k/v (c/u) | los 3 | ¿cabe en L2? |
|---|---:|---:|---|
| FB15k-237 ind v1 (N=1594, d64, B16) | 6.5 MB | 19.6 MB | sí |
| FB15k-237 transductivo (N=14505, d32, B4) | 7.4 MB | 22.3 MB | sí |
| **WN18RR transductivo (N=40943, d32, B32)** | 167.7 MB | **503 MB** | **NO** |

Se incluyó WN18RR transductivo **a propósito**: es el único caso donde los tensores desbordan
L2 por 12×, o sea donde la hipótesis del cache tenía que morder. Dio **1.01×**.

### Intento 2 (descartado sin implementar): evitar el gather relacional del tramo expander

Todas las aristas expander comparten `R_exp`, así que `rel_bias[:, rel]` y `rel_value[:, rel, :]`
devuelven la misma fila para ese tramo y podrían reemplazarse por un `expand`. Cuantificado en
FB237 transductivo: ahorra **~14 MB por capa** (2.8 del bias + 11.2 del valor) ⇒ ~56 MB por
forward. Contra eso, **cada** tensor de activaciones por arista `(B,E,H,hd)` pesa **330 MB** y
hay varios por capa más el backward ⇒ del orden de 1–2 GB por capa. **El ahorro es ~1 %**, por
debajo del piso que el intento 1 demostró no resolver. Encima obliga a partir y concatenar
tensores, lo que asigna memoria nueva y podría anular el ahorro. **No se implementa.**

### Intento 3 (descartado, y basado en una lectura ERRÓNEA del código)

Se afirmó que `_expander` hace `.to(device)` "en cada capa, en cada batch". **Falso**: se llama
en `src/model.py:1267`, en el `forward` del modelo, **antes del loop de capas** ⇒ **una vez por
batch**. Son 2 tensores de 87 030 int64 = **1.4 MB por forward**, o 4.6 MB/s a 3.34 it/s, contra
los GB/s del bus. **Ruido. No se implementa.**

**Análisis**

- **(A) CORRECCIÓN a la entrada del 2026-07-20.** Esa entrada explica el sobrecosto del expander
  por **cache misses** debidos a la aleatoriedad de la permutación d-regular ("filas `q/k/v` de
  los hubs quedan en L2 y se reusan; las expander caen en filas impredecibles"). **Esa
  explicación no sobrevive esta prueba**: (i) reordenar por `dst` —que cambia el patrón de
  acceso de los cinco gathers/scatters dominantes— no recupera nada; (ii) en los grafos donde
  todo cabe en L2 no puede haber miss para empezar, y ahí el sobrecosto igual aparece.
- **(B) El sobrecosto SÍ es real y se reprodujo**: 5.84 → 3.34 it/s por **+15.6 %** de aristas =
  **1.75×** (julio midió 1.67×). Despejando, cada arista expander cuesta **~4.8×** una real
  (julio: 4.3×). El número está bien; ~~la causa queda ABIERTA~~ ⇒ **RESUELTA el 2026-08-09**:
  no era el expander sino el backward de `table[:, rel]` (indexación avanzada), donde las 87 K
  aristas expander comparten el índice `R_exp` y disparan 87 K atómicos sobre una fila. Con
  `index_select` el sobrecosto cae a **1.08×**.
- **(C) Candidatos NO descartados** para explicar el 1.75×: contención de atómicos en
  `index_add_`, el cambio en la distribución de grado de entrada que introduce el expander, o
  algo del path de `SparseExpanderGraphTransformer` que no escala con el número de aristas.
- **(D) LECCIÓN METODOLÓGICA, y es la misma que la entrada (k) ya había dejado escrita como
  regla.** Las tres optimizaciones se estimaron **razonando sobre el código** en vez de medir
  dónde se va el tiempo, y las tres fallaron: una midió 1.00×, otra tiene techo de 1 %, la
  tercera partía de leer mal una línea. **En este harness no se optimiza por inspección: se
  perfila primero.** La herramienta correcta es `torch.profiler` con `record_shapes=True` sobre
  unos pasos de `sparse` vs `sparse_exp`, que dice qué kernel concreto se lleva el tiempo extra.

**Decisión**

- **NO re-proponer**: ordenar por `dst` (medido 1.00×), evitar el gather del tramo expander
  (~1 %), ni cachear el expander en GPU (1.4 MB/forward). Los tres están cerrados.
- **Antes de cualquier próximo intento de optimización, correr un profiler.** Sin eso son
  conjeturas.
- **Prioridad: baja.** El expander aporta ~cero en los 4 regímenes (lista negra #8); el único
  motivo para optimizarlo es que `sparse_exp+edrop` es el brazo que empata a NBFNet en v2.
  Agendado para después de la defensa.
- Artefactos: `bench_sort.sh`, `logs/bench_sort_914{70,71}.log`. El cambio de `src/model.py`
  fue revertido; no queda código de esta línea.

---

## 2026-08-08 (k) — OPTIMIZACIÓN de `--rel_param lowrank`: la matriz de valor se precomputa **por RELACIÓN** en vez de por arista ⇒ **1.7× más rápido, 1.8× menos memoria, y el costo deja de depender de k**. Verificado bit a bit idéntico. **REGLA: de acá en adelante todos los experimentos usan esta versión.**

**Contexto**: al llevar `lowrank` a FB15k-237 **transductivo** (E = 558 735) el costo se volvió
prohibitivo, y la predicción de la entrada (e) —"el término nuevo es `O(B·E·hd·k)`, lineal en k"—
resultó **falsa en la práctica**: bajar de k=8 a k=2 ahorró solo **8 %**.

### Medición que disparó el cambio (FB15k-237 transductivo, sparse L4/d32, 2×A100 DDP, batch global 8)

| brazo | it/s | h/época | 20 épocas | GPU |
|-------|-----:|--------:|----------:|----:|
| sin lowrank (referencia) | 5.71 | 1.9 h | ~37 h | 7.5 GB |
| lowrank k=2, versión original | 1.54 | 6.9 h | ~139 h | 9.1 GB |
| lowrank k=8, versión original | 1.42 | 7.5 h | ~150 h | 14.7 GB |
| **lowrank k=8, OPTIMIZADO** | **2.45** | **4.4 h** | **~87 h** | **8.4 GB** |

Sobrecosto contra el modo `diag`: **4.0× → 2.3×**. Y **k=8 optimizado corre más rápido que k=2
sin optimizar** ⇒ ya no hay razón para bajar el rango por costo.

**Análisis (causa mecánica)**

- **(A) El cuello NO eran los FLOPs, era mover tensores por arista.** Por eso el costo apenas
  bajaba con k: la versión original hacía **tres tensores por arista y por capa** — gather de
  `U[:, rel]` y de `W[:, rel]`, ambos `(E,H,hd,k)` y **con un `permute` que los volvía no
  contiguos**, más el intermedio `(B,E,H,k)` de la primera contracción. A E=558 735 eso es del
  orden de 1.7 GB de tráfico por capa. Los FLOPs sí escalan con k; el tráfico dominante no.
- **(B) La optimización es puramente algebraica.**
  `v ⊙ g[r] + (v @ U[r]) @ W[r]ᵀ  ==  v @ (diag(g[r]) + U[r] W[r]ᵀ)`.
  La matriz `M[r] = diag(g[r]) + U[r]W[r]ᵀ` se precomputa **una vez por forward, por relación**:
  cuesta `O(H·R·hd²·k)`, **independiente de E**. Con **R ≈ 475 contra E = 558 735** el
  precómputo es gratis. El forward pasa a hacer **un solo gather** `(E,H,hd,hd)` y **un solo
  einsum**. Además se guarda `M` con layout `(R,H,hd,hd)` para que `M[rel]` sea un gather sobre
  la dimensión 0 y contiguo (adiós al `permute`), y en modo lowrank **ya no se materializa** el
  gather diagonal `g_e`, que era otro `(1,E,H,hd)` inútil.
- **(C) Corolario: el costo ya no depende de k** ⇒ usar **k=8 siempre**; k=2/k=4 solo tienen
  sentido como ablation de capacidad, no como ahorro.
- **(D) Lección metodológica**: la complejidad asintótica escrita en la entrada (e) era correcta
  pero **irrelevante** para el wall-clock. En estas capas el régimen es memory-bandwidth-bound —
  el mismo diagnóstico que ya había aparecido con el expander el 2026-07-20 (una arista aleatoria
  cuesta 4.3× una real **por localidad**, no por cómputo). **En este harness, estimar costo por
  FLOPs lleva a error; hay que contar tensores por arista.**

**Verificación (hecha ANTES de relanzar, no asumida)**

| qué | resultado |
|---|---|
| álgebra `M[rel]` vs la fórmula anterior, sobre pesos reales | **max \|diff\| = 0.000e+00** |
| `lowrank` ≡ `diag` en la inicialización (los 3 modelos) | ✅ |
| gradiente en `rel_u` y `rel_w` tras un paso de optimizador | 6/6 capas |
| modo `diag` sin regresión | ✅ |

Es **el mismo modelo bit a bit**; solo cambió cómo se calcula.

**Decisión**

- ⚠️ **REGLA VIGENTE: todos los experimentos de acá en adelante usan esta versión.** No revertir
  a la implementación por arista. `src/model.py::{RelationalAttentionLayer,
  SparseRelationalAttentionLayer}._rel_matrix`.
- **Los TIEMPOS medidos antes de esta entrada no son comparables** con los de después. Los
  **resultados sí**, porque la matemática es idéntica (verificado). En particular el job 91466
  (`rel_param` en v2) arrancó con la versión vieja, pero como el sbatch lanza **un proceso por
  corrida**, sus semillas posteriores toman ya el código optimizado: **sus números son válidos y
  mezclables, sus tiempos no.**
- Relanzado el brazo combinado transductivo con k=8 optimizado: **job 91469**
  (`--remove_one_hop --rel_param lowrank --rel_rank 8`), ~87 h ⇒ un encadenado con el wall de
  48 h (antes hubieran hecho falta 4).
- Se cancelaron 91467 (k=8 original) y 91468 (k=2 original); sus directorios de checkpoint se
  borraron para que no disparen `--resume_from`.

---

## 2026-08-08 (j) — 🎯 **NBFNet v2 re-medido (0.5245 ± 0.0042) ⇒ el GT DISPERSO IGUALA a NBFNet en FB15k-237 ind v2** (+0.0030 ± 0.0031, t=0.97) — primer empate del proyecto sin heredar el V de NBFNet. Y el control exonera al sbatch: el −0.236 de `remove_one_hop` era REAL

**Contexto**: cierra las dos incógnitas abiertas en las entradas (a), (f), (h) e (i). Job 91462
(NBFNet v2, 50 ép, n=3) y job 91464 (control del script generalizado).

### 1. Head-to-head en FB15k-237 ind v2 — 50 ép, n=3 semillas, TODO con el protocolo vigente

| brazo | **test_mrr** | H@1 | H@3 | H@10 | MR | best ép |
|-------|-------------:|----:|----:|-----:|---:|--------:|
| **sparse_exp deg3 + edge_drop 0.2** | **0.5275 ± 0.0032** | **0.426** | **0.590** | **0.706** | 52.5 | 44.0 ⚠️ |
| **NBFNet** | **0.5245 ± 0.0042** | 0.423 | 0.583 | 0.705 | **49.9** | 4.3 |
| sparse + edge_drop 0.2 | 0.5157 ± 0.0088 | 0.417 | 0.571 | 0.694 | 79.9 | 18.7 |
| sparse_exp deg3 (ed=0.0) | 0.4736 ± 0.0120 | 0.381 | 0.524 | 0.632 | 103.4 | 10.3 |
| sparse (ed=0.0) | 0.4556 ± 0.0225 | 0.365 | 0.505 | 0.606 | 103.8 | 10.7 |

Contrastes (Welch, mismas semillas 42/43/44, todo a 50 ép):
- **sparse_exp+edrop vs NBFNet: +0.0030 ± 0.0031 (t=0.97) ⇒ INDISTINGUIBLES.**
- sparse+edrop vs NBFNet: −0.0088 ± 0.0056 (t=−1.56) ⇒ **tampoco significativo**.
- NBFNet re-medido vs el histórico n=1 de 20 ép (0.526): Δ **−0.0015** ⇒ el número registrado
  era correcto, y **NBFNet tampoco estaba subentrenado en v2** (best-valid en la época 4.3 de 50).

### 2. Control del script generalizado (job 91464, NBFNet ind v1, sin el flag, seed 42)

**test_mrr 0.4579** contra el histórico **0.458 ± 0.004** ⇒ **reproduce exactamente**.

**Análisis**

- **(A) PRIMER EMPATE CON NBFNet EN FB15k-237 SIN HEREDAR SU V.** `sparse_nbfv` ya empataba
  (0.527 en v2, 0.740 en WN18RR) pero **por construcción**: su V son las representaciones de un
  NBFNet corrido aparte (lista negra #6). Éste es un GT disperso puro, con labeling trick y
  embeddings de relación como únicos parámetros estructurales. Las tres métricas de ranking
  acompañan (H@1 0.426 vs 0.423, H@3 0.590 vs 0.583, H@10 0.706 vs 0.705) ⇒ no es un empate en
  MRR con lo demás en contra.
- **(B) Es exactamente el entregable prometido de la Etapa 1 del manuscrito** (ver entrada (f)):
  *"un modelo Exphormer condicionado a la query que sea **competitivo** con NBFNet, con
  complejidad lineal, y que funcione en el setting inductivo sin embeddings de entidad"*.
  Complejidad lineal: demostrada (`O(B·E·d)`, sin término `N²`). Sin embeddings de entidad:
  por construcción. Competitivo: **establecido con barras de error en v2**.
- **(C) El margen es un PISO, por dos razones independientes.** (i) El brazo del expander está
  **subentrenado**: best-valid en las épocas 39/45/48 de 50, dos de tres todavía mejorando.
  (ii) NBFNet, en cambio, satura en la época 4.3 ⇒ está en su techo. Más épocas solo pueden
  ampliar la diferencia a favor del GT.
- **(D) Ojo con no sobre-leerlo: el empate es de v2, y en v1 la brecha sigue viva** (sparse+edrop
  0.397 ± 0.019 vs NBFNet 0.458 ± 0.004). Sumado a WN18RR v1 (0.738 vs 0.740, empate), el cuadro
  es: **empata en 2 de 3 splits, pierde en v1**. v1 es el split más chico y composicional.
- **(E) Nota metodológica que vale para la defensa**: el 0.5245 medido acá es **más alto** que el
  0.514 que reporta la re-evaluación de terceros (tabla de KnowFormer), y muy por encima del
  0.359 que da la receta liberada de los autores con BCE-k (entrada (c)). O sea el empate se
  afirma contra **la versión más fuerte** del baseline, no contra la más conveniente.
- **(F) El control cierra la duda del script: `--remove_one_hop` SÍ destruye a NBFNet en ind v1.**
  0.4579 sin el flag vs 0.2216 ± 0.0074 con él, mismo script. El efecto es **genuino**, aunque el
  mecanismo sigue sin explicación (0.13 % más de aristas quitadas, 2.2 % de queries sin camino).
  Lo que queda establecido es una **asimetría de régimen muy fuerte**: el mismo flag mejora en el
  transductivo denso (E=544 K) y destruye en el inductivo chico (E=8 490). Hipótesis a testear:
  en un grafo denso casi siempre hay camino alternativo y quitar el atajo fuerza composición; en
  uno disperso destruye la evidencia sin dejar alternativa.

**Decisión**

- **Resultado titular disponible para la defensa**, con la formulación exacta: *"en FB15k-237
  inductivo v2 el Graph Transformer disperso con aristas expander iguala a NBFNet
  (0.5275 ± 0.0032 vs 0.5245 ± 0.0042, t=0.97, n=3, 50 épocas), con complejidad lineal y sin
  embeddings de entidad"*. **No decir "supera"**: t=0.97 es empate, no ventaja.
- **La conclusión global del proyecto NO cambia**: ninguna atención supera a NBFNet; ahora hay un
  split donde **iguala**. La lista negra no se toca.
- Prioridades siguientes: (1) re-correr el brazo `sparse_exp+edrop` de v2 a **100 épocas** para
  saber dónde converge (está subentrenado); (2) `--rel_param lowrank k=8` en v2 (en v1 dio el
  mejor número de atención del proyecto, 0.4218); (3) explicar la asimetría de `remove_one_hop`.
- Artefactos: `logs/edrop_v2_9146{2,4}.log`, `experiments/edrop_v2_nbfnet_p00_ep50_s{42,43,44}/`.

---

## 2026-08-08 (i) — `--remove_one_hop`: resultado MUY BUENO en transductivo (test 0.4143 en la época 2 de 20, supera todo lo registrado) y CAÍDA de −0.236 en NBFNet ind v1 ⇒ **asimetría de régimen real** (el control descartó bug de script, ver entrada (j))

**Contexto**: primeros resultados del flag implementado en la entrada (d). Dos regímenes, dos
respuestas incompatibles.

### 1. Transductivo (job 91458, sparse, 2×A100 DDP, batch global 8) — PARCIAL, época 2 de 20

Checkpoint del mejor valid evaluado **sin detener el entrenamiento**, con `srun --jobid=91458
--overlap` (no consume slot de QOS) y `--devices 1` ⇒ **el test NO está afectado por DDP**.

| | valid_mrr | **test_mrr** | H@1 | H@3 | H@10 | MR |
|---|---:|---:|---:|---:|---:|---:|
| **one_hop, época 2/20** | 0.411 | **0.4143** | 0.325 | 0.451 | 0.591 | 146.0 |
| *sparse softmax L4/d32 batch 8 (baseline directo, ép 19/20)* | *0.402* | *0.3965* | *0.300* | *0.437* | *0.591* | *128.4* |
| *sparse softmax L6/d64 batch 16 (2.4× más caro)* | *0.407* | *0.4028* | *0.306* | *0.443* | *0.595* | *116.4* |
| *NBFNet transductivo (literatura)* | — | *~0.415* | — | — | — | — |

**En la época 2 de 20 ya supera todo lo transductivo registrado del proyecto**: +0.0178 sobre su
baseline a config y batch idénticos, +0.0115 sobre la config grande, y queda a **0.0007** del
~0.415 de literatura de NBFNet. Trayectoria de valid: 0.4072 (ép 0) → 0.4105 (ép 1), contra
0.353-0.374 y 0.366-0.375 de **tres** corridas baseline en las mismas épocas; el baseline alcanzó
su mejor valid de todo el entrenamiento (0.40157) recién en la **época 17**.

Matiz: el **MR empeora** (146.0 vs 128.4) mientras H@1 sube +0.025 ⇒ mejora la cabeza del ranking
y empeora la cola. Coherente con "aprendió composición en vez del atajo", pero es n=1 a 2 épocas.

### 2. Inductivo v1 — NBFNet + `--remove_one_hop` (job 91463, 50 ép, n=3): **NO CREÍBLE**

| brazo | test_mrr | H@1 | H@10 | MR | valid | best ép |
|---|---:|---:|---:|---:|---:|---:|
| NBFNet **+ one_hop** | **0.2216 ± 0.0074** | 0.140 | 0.382 | 199.5 | 0.327 | **3.0 de 50** |
| *NBFNet sin flag (ref, n=6)* | *0.458 ± 0.004* | *0.371* | *0.605* | *117* | *0.492* | — |

Δ **−0.236**, con el best-valid en la época 3 de 50 y `train_loss` bajando a 1.16 ⇒ firma de
colapso, no de regularización.

**Dos verificaciones, y las dos lo desmienten como efecto genuino:**
- El flag quita **10.7 aristas extra por batch sobre 8490** ⇒ **0.13 % más del grafo** (medido
  sobre 20 batches). Una intervención del 0.13 % no puede producir una caída del 51 % relativo.
- Solo el **2.2 %** de las queries de train pasan a tener el target **inalcanzable** en 6 saltos
  (7.2 % → 9.5 %). Esa era la hipótesis mecánica obvia y no alcanza ni de lejos.

Argumento externo adicional: **el config inductivo de NBFNet trae `remove_one_hop: yes`**. Si
fuera catastrófico en inductivo, sus autores no lo usarían.

**Análisis**

- **(A) La asimetría entre el tamaño de la intervención y el del efecto es el dato central.**
  0.13 % del grafo ⇒ −0.236 de MRR no cierra por ningún mecanismo que se me ocurra. Cuando eso
  pasa, el sospechoso #1 es el código propio: el flag y el sbatch generalizado se escribieron
  **hoy**.
- **(B) Lanzado el control decisivo (job 91464)**: mismo script generalizado, misma config,
  **sin** el flag, 1 semilla. Si reproduce ~0.458, el flag es el culpable y falta el mecanismo;
  si da ~0.22, **el bug está en el sbatch generalizado** y el −0.236 no significa nada — y
  entonces habría que revisar también si contamina algo del job 91462 (NBFNet v2), que usa el
  mismo script.
- **(C) Los dos resultados son incompatibles con una lectura simple.** El mismo flag mejora
  fuerte en transductivo y destruye en inductivo. Puede ser real (grafo denso vs disperso: en
  FB237 transductivo E=544 K y casi siempre hay camino alternativo; en ind v1 E=8490) pero la
  magnitud inductiva sigue sin explicarse.

**Decisión**

- ~~**NO usar ninguno de los dos números** hasta que vuelva 91464.~~ ⇒ **RESUELTO: el control
  91464 dio 0.4579 contra el histórico 0.458 ⇒ el script generalizado está BIEN y el −0.236 es
  REAL.** Ver entrada (j) punto (F). Queda una asimetría de régimen sin explicar (mejora en
  transductivo denso, destruye en inductivo chico), así que **el número transductivo sigue siendo
  n=1 a 2 épocas y no se cita en la defensa**; el inductivo sí está establecido.
- Sigue pendiente el brazo de control transductivo (`sbatch sbatch_onehop_trans.sh control`),
  que además cerraría la duda del artefacto DDP en las curvas de valid.
- **Patrón operacional útil, reutilizable**: se puede evaluar el mejor checkpoint de un job en
  curso con `srun --jobid=<N> --overlap ... --eval_only --eval_ckpt <ckpt>`, sin detener el
  entrenamiento y **sin consumir slot de QOS**. Costó ~14 min compartiendo GPU.
- Artefactos: `logs/edrop_v2_91463.log`, `logs/onehop_trans_eval_ep2.log`,
  `experiments/onehop_trans_fb237/epoch=2-step=102045.ckpt`.

---

## 2026-08-08 (h) — RESULTADOS del barrido `--rel_param lowrank`: **la capacidad relacional SÍ mueve la aguja — es la PRIMERA intervención del proyecto que mejora al full attention (+0.040, t≈5.9)** — pero a parámetros igualados NBFNet sigue ganando

**Contexto**: resultados del barrido lanzado en la entrada (e), que dejó escrita la predicción
falsable. Jobs 91459 (sparse, 4 rangos × 3 semillas) y 91461 (full, k=8) completos; **91460
(full: diag y k=4) todavía en curso**.

### Resultado (FB15k-237 ind v1, 50 ép, n=3 semillas 42/43/44)

**SPARSE + edge_drop 0.2**

| brazo | params | **test_mrr** | H@1 | H@10 | MR | semillas |
|-------|-------:|-------------:|----:|-----:|---:|----------|
| diag (control de hoy) | 384 K | 0.3689 ± 0.0365 | 0.302 | 0.486 | 162.8 | 0.3293 / 0.3761 / 0.4014 |
| lowrank k=2 | 936 K | 0.4061 ± 0.0072 | 0.341 | 0.503 | 173.8 | — |
| lowrank k=4 | 1.49 M | 0.3968 ± 0.0237 | 0.330 | 0.504 | 173.3 | — |
| **lowrank k=8** | 2.60 M | **0.4218 ± 0.0076** | **0.357** | **0.523** | **162.4** | 0.4132 / 0.4248 / 0.4276 |

**FULL attention**

| brazo | params | **test_mrr** | H@1 | H@10 | MR | semillas |
|-------|-------:|-------------:|----:|-----:|---:|----------|
| *diag (histórico 2026-08-07)* | 384 K | *0.3764 ± 0.0021* | *0.315* | *0.489* | *192.9* | — |
| **lowrank k=8** | 2.60 M | **0.4164 ± 0.0115** | 0.356 | 0.509 | 170.5 | 0.4071 / 0.4126 / 0.4293 |

Contrastes (Welch):
- **FULL k=8 vs full diag: +0.0400 ± 0.0068 (t=5.93)** ⚠️ contra el control HISTÓRICO.
- SPARSE k=8 vs diag de hoy: +0.0529 ± 0.0215 (t=2.46) · vs diag histórico n=6 (0.397 ± 0.019):
  +0.0248 ± 0.0089 (t=2.78).
- **NBFNet vs full k=8: +0.0416 ± 0.0068 (t=6.08)** — a parámetros comparables.

**Análisis (causa mecánica)**

- **(A) Es la PRIMERA intervención del proyecto que mueve al full attention.** RWSE (−0.009),
  LapPE (+0.035 en v1 pero no replica en v2), source_rw (−0.062), edge_drop (−0.007, nulo): todas
  neutras o negativas. Ésta da **+0.040 con t≈5.9** sobre un modelo cuya σ es 0.0021. La hipótesis
  de capacidad relacional de la entrada (e) **se confirma parcialmente**.
- **(B) La predicción escrita en (e) se cumple a medias.** Predije "MRR monótono con k, salto
  grande entre diag y k=2". El salto diag→k=2 existe y es grande (**+0.037** en el sparse), pero
  **NO es monótono**: k=4 (0.3968) < k=2 (0.4061), aunque dentro del ruido entre sí. El máximo
  está en k=8. Lectura honesta: **la capacidad relacional importa, pero la curva en k no es
  limpia** con n=3.
- **(C) EL RESULTADO QUE MÁS IMPORTA: a parámetros igualados, NBFNet sigue ganando.** Full k=8
  tiene **2.60 M** contra **2.37 M** de NBFNet — o sea el GT ahora tiene *más* parámetros — y
  pierde **0.4164 vs 0.458 (+0.0416 ± 0.0068, t=6.08)**. ⇒ **La brecha NO era de capacidad
  relacional.** Era parte de ella (~40 % del gap del full: 0.082 → 0.042), pero el resto no.
  Esto refuerza el diagnóstico acumulado: lo que falla es la agregación aprendida, no cuántos
  parámetros hay por relación.
- **(D) El sparse alcanza 0.4218, el mejor número de atención registrado en ind v1** (contra
  0.397 del sparse+edrop y 0.3764 del full). Sigue **−0.036 debajo de NBFNet**.
- **(E) ⚠️ El control `diag` de hoy NO reprodujo el histórico**: 0.3689 ± 0.0365 (n=3) contra
  0.397 ± 0.019 (n=6). Una semilla dio 0.3293. Puede ser la varianza conocida del sparse (σ hasta
  0.054 sin regularizar) o deriva de código. **Es exactamente por esto que el brazo `diag` estaba
  en el barrido**; hasta entender la diferencia, los contrastes del sparse quedan con dos lecturas
  (+0.053 o +0.025) según qué control se use. **El del full está pendiente (job 91460)** y es el
  que decide, porque ahí σ=0.0021.

**Decisión**

- ⚠️ **CERRADA el 2026-08-09 (b): NO replica en v2** (+0.0034 ± 0.0057, t=0.59, contra +0.052
  en v1). Lo de abajo se escribió antes de esa réplica y quedó superado.
- ~~**Línea NO cerrada — es la primera hipótesis viva desde que se agotaron agregación y encodings.**~~
  Pero la conclusión central del proyecto **no cambia**: a parámetros igualados NBFNet sigue
  ganando con t=6.08.
- ~~**Esperar 91460** antes de fijar el número del full.~~ ⇒ **HECHO, ver entrada 2026-08-09 (b)**:
  con el control del código de hoy (diag 0.3641 ± 0.0189) el contraste del full es
  **k=8 +0.0523 ± 0.0128 (t=4.09)** y **k=4 +0.0507 (t=3.87)** ⇒ confirmado en v1.
  ⚠️ **PERO NO REPLICA EN v2** (+0.0034, t=0.59) ⇒ efecto dependiente de split, como LapPE.
- Pendiente barato y con valor: `k=8` en **v2** y en **WN18RR v1**, para ver si replica fuera de
  v1 (LapPE murió exactamente en ese test).
- Artefactos: `sbatch_relparam_v1.sh`, `experiments/relparam_v1_*`, `logs/relparam_v1_914{59,61}.log`.

---

## 2026-08-08 (g) — DÓNDE CAE LA ATENCIÓN del full GT (medido en los 3 checkpoints entrenados): **el 99.4 % de la masa va a pares SIN arista**, o sea sin relación ⇒ su canal de composición relacional está atenuado ~300× y el modelo NUNCA aprende a concentrarse en las aristas

**Contexto**: pregunta del usuario — ¿el Graph Transformer captura composición relacional?, y en
el full, que conecta todos con todos, ¿cómo la captura si entre dos nodos sin arista no hay
relación? La bitácora **no tenía ninguna medición de dónde cae la atención**. Se instrumentó el
`RelationalAttentionLayer` (copia del forward que guarda `attn`) y se midió sobre los tres
checkpoints de `full_v1_p00_ep50_s{42,43,44}` (50 ép, best-valid), en el grafo de test inductivo
**y** en el de train.

### Dónde vive la composición relacional en cada modelo

- **NBFNet**: mensaje `h[u] ⊙ w_r`; apilar L capas produce `x⁰[head] ⊙ w_{r1} ⊙ … ⊙ w_{rL}` a lo
  largo de cada camino. Eso **es** la composición.
- **GT (full y sparse)**: el término (ii), `out[dst] += α[dst,src] · (v[src] ⊙ g[rel])`, tiene la
  **misma forma DistMult**. Pero `a_e = attn[:, :, dst, src]` indexa **solo aristas** ⇒ la
  composición vive únicamente ahí, en los dos modelos.
- **Par (i,j) SIN arista en el full**: el peso es `softmax(q_i·k_j/√d)` —similitud de contenido
  pura—, **no** se suma `b[cabeza,rel]` (solo va sobre aristas) y **no** se aplica `⊙ g[rel]`.
  Aporta `α_ij · v_j` sin tipar. Es literalmente el **caso (a)** de `transformer_vs_nbfnet.tex`
  (atención ciega a la estructura) operando **dentro** del modelo diseñado como caso (b).
  ⇒ **Conectar directo NO captura el camino entre el par**: la conexión directa no lleva
  información relacional. Los caminos entran solo por el término restringido a aristas, iterado
  por capas — o sea **por exactamente el mismo mecanismo que el sparse**.

### Resultado (n=3 semillas, FB15k-237 ind v1, grado de entrada medio 3.6, N=1093/1594)

| capa | **masa en aristas — grafo de test (ind. disjunto)** | ×uniforme | **grafo de train** | ×uniforme |
|---|---:|---:|---:|---:|
| 0 | 0.30 % ± 0.01 | 0.9× | 0.31 % ± 0.01 | 0.9× |
| 1 | 1.06 % ± 0.32 | 3.2× | 0.26 % ± 0.06 | 0.8× |
| 2 | 0.54 % ± 0.02 | 1.6× | 0.73 % ± 0.38 | 2.2× |
| 3 | 0.53 % ± 0.14 | 1.6× | 1.05 % ± 0.20 | 3.2× |
| 4 | 0.52 % ± 0.12 | 1.6× | 0.92 % ± 0.27 | 2.8× |
| 5 | 0.43 % ± 0.05 | 1.3× | 0.32 % ± 0.03 | 1.0× |
| **todas** | **0.57 % ± 0.28** | **1.7×** | **0.60 % ± 0.37** | **1.8×** |

`|rel_bias|` medio tras 50 épocas: **0.142**. Si la atención fuera uniforme, la masa en aristas
sería **0.33 %**.

**Análisis (causa mecánica)**

- **(A) El canal composicional recibe menos del 1 % del peso.** **99.4 %** de la masa de atención
  cae en pares sin arista, donde no hay relación ninguna. El término (ii) —el único que compone
  relaciones— está multiplicado por esas α diminutas.
- **(B) El modelo PODRÍA arreglarlo y no lo hace.** Si `rel_bias` creciera, el softmax se
  concentraría en los vecinos y el full recuperaría al sparse. Tras 50 épocas está entre **0.8× y
  3.2× de uniforme** con `|b| = 0.142`: **no aprende a mirar las aristas.** Esto convierte el
  argumento "el full es el techo del sparse" en algo más preciso: *lo es en capacidad, pero el
  entrenamiento no lo lleva ahí.*
- **(C) Es igual en train y en test (0.60 % vs 0.57 %) ⇒ NO es un problema de transferencia.**
  Es una propiedad de la solución que encuentra el optimizador, no del cambio de grafo. Descarta
  leerlo como una firma más de overfit estructural.
- **(D) Dos mecanismos que explican por qué.** (i) **Dilución en el origen**: el softmax normaliza
  sobre las N claves, así que al arranque `α ≈ 1/N ≈ 0.0009` contra `1/grado ≈ 0.28` del sparse —
  el mensaje composicional entra **~300× más débil**. (ii) **Corrupción de los caminos con la
  profundidad**: tras la capa 1, `x[v]` ya mezcla aportes de todos los nodos, así que en la capa 2
  `v[src]` deja de ser "evidencia que llegó a src por caminos reales desde el head" ⇒ el producto
  `g[r1] ⊙ g[r2]` **ya no corresponde a un camino que existe**. La profundidad deja de significar
  largo del camino.
- **(E) EXPLICACIÓN MECÁNICA del resultado más contraintuitivo del proyecto.** La entrada del
  2026-08-07 (b) registra que el techo de expresividad (full 0.3764) rinde **peor** que su propia
  restricción (sparse+edrop 0.397) y lo atribuye a que el sparse estaba subentrenado. Hay además
  una razón de principio: **en el sparse el soporte de la atención ES el conjunto de aristas, así
  que el 100 % de su agregación está relacionalmente tipada y es consistente con caminos reales;
  en el full lo está el 0.6 %.**
- **(F) Matiz honesto que no hay que omitir**: sin regularizar, el full (0.3764 ± 0.0021) le gana
  al sparse (0.340 ± 0.027). El canal denso no es puramente dañino — es un sesgo inductivo
  distinto que sobreajusta menos la topología del train (gap val→test 0.056 vs 0.108). Pero con
  el sparse bien regularizado gana el sparse. Lectura: la lectura global por contenido vale algo,
  **menos que un canal composicional limpio**.

**Decisión**

- **Argumento nuevo y mejor para la Etapa 1 de la tesis** (ver entrada (f)): el GT disperso no es
  "el que cabe en memoria" sino **el que hace coincidir el soporte de la atención con el lugar
  donde vive la composición relacional**. Es una justificación mucho más fuerte que la
  escalabilidad, y sale de datos propios.
- Material directo para la defensa: la tabla de arriba responde "¿el GT captura composición
  relacional?" con un número, no con una intuición.
- **Sin medir todavía**: la misma sonda sobre el **sparse** (donde por construcción debería dar
  100 %) como control positivo, y sobre WN18RR ind v1 (donde el full es el peor de los tres,
  0.673 vs 0.740 — la predicción es que ahí la masa en aristas sea aún menor por N mayor).
- Artefactos: sonda en el scratchpad de la sesión; checkpoints `experiments/full_v1_p00_ep50_s{42,43,44}/`.

---

## 2026-08-08 (f) — MARCO DE LA TESIS (de `manuscrito_candidatura.md`) y mapeo de lo medido contra sus objetivos: el objeto central es el modelo **DISPERSO**, el full attention es un CONTROL, y **H1/H2 están en tensión con los datos** — con la reformulación que sí sostienen

**Contexto**: el usuario corrigió una premisa que esta bitácora tenía mal encuadrada. Defensa
de candidatura la semana del 2026-08-10. Se registra el marco real para que ninguna sesión
futura vuelva a razonar desde el encuadre equivocado.

### Qué es la tesis (manuscrito_candidatura.md)

**Objetivo general**: diseñar y validar una arquitectura **Graph Transformer con atención
DISPERSA basada en grafos expander**, que integre **codificación relacional composicional y
transferible** (lógica de ULTRA), capaz de inferencia de enlaces con generalización
**ZERO-SHOT** en KGs no vistos.

**Hipótesis**:
- **H1**: atención dispersa guiada por expander + representaciones relacionales transferibles ⇒
  captura dependencias globales con complejidad **lineal**, sin sacrificar el poder expresivo
  necesario para **superar a las GNN de paso de mensajes**.
- **H2**: un GT disperso con codificación relacional composicional generaliza **zero-shot** a
  relaciones/entidades no vistas, **superando a NBFNet** en KGs no vistos, **gracias a que la
  estructura expander facilita la propagación de señales estructurales globales**.

**Objetivos específicos** y su estado real:

| | objetivo | estado |
|---|---|---|
| **OE1** | Comparar estrategias de atención dispersa (base Exphormer) para **explicar cómo** el patrón expander afecta la captura de dependencias globales | **EJECUTADO** — resultado negativo con mecanismo. ⚠️ **nodos virtuales SIN probar** |
| **OE2** | Diseñar codificación relacional composicional transferible (ULTRA) compatible con atención dispersa | **Parcial** — `build_relation_graph` + `RelationGNN` implementados (port de `ULTRA/ultra/tasks.py:144`) pero usados solo para **tipar aristas expander**, no como encoder relacional del modelo |
| **OE3** | Integrar el mecanismo en Exphormer, prototipo inductivo multigrafo | **No iniciado** |
| **OE4** | Evaluar zero-shot en KGs no vistos vs NBFNet, KnowFormer, ULTRA | **No iniciado** |
| **OE5** | Qué componentes son estrictamente necesarios (eficiencia vs expresividad) | **Parcial, y es donde más evidencia hay** (todo el diagnóstico por eliminación) |

**Etapas**: (1) adaptar Exphormer a KGC condicionado a la query `(u,q)`, sin embeddings de
entidad — entregable esperado *"competitivo con NBFNet, complejidad lineal, inductivo"*,
publicable en ICLR/LoG; (2) codificación relacional transferible; (3) entrenamiento multigrafo
(FB15k-237 + WN18RR + NELL-995) y evaluación zero-shot.

**Análisis (qué implica para lo ya medido)**

- **(A) El objeto central es el modelo SPARSE, no el full.** `GOALS.md` y `CLAUDE.md` ponen como
  pregunta central "¿el full attention supera a NBFNet?". Eso es el **sanity check de la etapa**,
  no la tesis: el full es el **control que descarta la alternativa densa** (y de paso responde a
  la objeción "te faltó alcance global"). ⚠️ **`GOALS.md` está desalineado con el manuscrito y
  habría que corregirlo.**
- **(B) H1 y H2 están en TENSIÓN con los datos, y hay que llegar a la defensa con eso resuelto.**
  Ambas atribuyen el mecanismo **al expander**. Medido en **cuatro regímenes**: Δ +0.0008
  (FB237 transd.), +0.011 y +0.012 con edge dropout (ind v1, v2), −0.006 (WN18RR ind v1),
  +0.0030 (WN18RR transd.). Y con causa: **el expander operaba como REGULARIZADOR, no como
  atajo estructural** — su beneficio aparece y desaparece exactamente donde aparece y desaparece
  el del `--edge_drop`, que es más barato. Refuerzo: en el grafo inductivo de test **el 91.1 %
  de las aristas expander no tiene camino real ≤3 saltos** ⇒ los pares tipables
  composicionalmente son justo los que el MP ya alcanza.
- **(C) La reformulación que SÍ sostienen los datos, y no es un parche.** **OE1 no pide que el
  expander funcione: pide EXPLICAR cómo afecta.** Está redactado como *"para explicar cómo el
  patrón de conectividad expander afecta la captura de dependencias globales"*. Un negativo con
  causa mecánica **cumple OE1 completo**. Y el mecanismo es una contribución teórica sobre las
  **condiciones de aplicabilidad de Exphormer**: en KGC **toda arista es una afirmación de
  evidencia**, así que una arista aleatoria es un **no-hecho** que fabrica caminos sin soporte
  relacional; donde los expander sí funcionan (moleculares, tareas a nivel de grafo) la arista
  extra solo **mueve** información hacia un pooling global y **no afirma** nada. El manuscrito ya
  promete "una contribución teórica sobre las propiedades de los grafos expander" (Etapa 3).
- **(D) El entregable de la Etapa 1 está PARCIALMENTE LOGRADO — y el manuscrito pide
  "competitivo", no "superior".** Estado por split:

  | | modelo disperso | NBFNet | ¿competitivo? |
  |---|---:|---:|---|
  | WN18RR ind v1 | 0.738 | 0.740 | **sí** |
  | FB15k-237 ind v2 | **0.5275 ± 0.0032** (sparse_exp+edrop, n=3) | **0.5245 ± 0.0042** (n=3) | **SÍ — ESTABLECIDO** (t=0.97) |
  | FB15k-237 ind v1 | 0.397 ± 0.019 | 0.458 ± 0.004 | no |

  Complejidad lineal: **demostrada** (`O(B·E·d)`, sin término `N²`). Inductivo sin embeddings de
  entidad: cumplido por construcción. **Dos de tres splits dan "competitivo".**
- **(E) OE2 NO está refutado por el resultado de `--exp_typing ultra`.** Ese experimento midió
  ULTRA como **tipado de aristas expander** (−0.051 FB237 / −0.030 WN18RR, replicado). OE2 pide
  ULTRA como **encoder relacional del modelo**, que es otra cosa. No confundirlos: el daño
  medido es de inyectar una relación real prestada en una arista **falsa**, no de usar
  representaciones relacionales transferibles.
- **(F) La objeción "esto es un GAT" y las distinciones que SÍ resisten.** Atención dispersa
  sobre una máscara fija **es** message passing con agregación aprendida; el `SparseGraphTransformer`
  es, computacionalmente, un **GAT relacional en un bloque transformer**. Verificado además que
  Exphormer usa score **trilineal por arista con segment-softmax** (`Exphormer/graphgps/layer/Exphormer.py:43-63`),
  o sea la misma forma que GAT con otra función de score, y que distingue arista real de
  sintética por **tipado de arista** (`exp_edge_fixer.py:24,60-61`), **no por positional
  encodings**. Las distinciones reales de esta tesis frente a un GAT son otras dos, y son las
  que hay que defender:
  1. **Condicionamiento a la query `(u,q)`** — representaciones que surgen dinámicamente en la
     propagación, sin embeddings de entidad. Un GAT no tiene noción de query. Es el corazón de
     la Etapa 1 según el propio manuscrito.
  2. **Codificación relacional transferible** — un R-GAT representa la relación por identidad;
     acá por posición estructural en el grafo de relaciones. Es lo que habilita el zero-shot.
  **NO defender la distinción por el patrón de atención** (el expander): es formalmente válida
  pero empíricamente hueca según nuestros propios datos.

**Decisión**

- **Este marco manda sobre `GOALS.md`** mientras `GOALS.md` no se corrija. Toda propuesta de
  experimento se evalúa contra OE1-OE5, no contra "¿supera el full a NBFNet?".
- ~~**URGENTE: re-medir NUESTRO NBFNet en v2 a 50 ép, n≥3.**~~ ⇒ **HECHO (job 91462, entrada
  (j)): 0.5245 ± 0.0042.** El entregable de la Etapa 1 queda **establecido en v2**: el GT
  disperso **iguala** a NBFNet (t=0.97), con complejidad lineal y sin embeddings de entidad.
  Cuadro completo: **empata en v2 y en WN18RR v1, pierde en v1.**
- **Pendiente de OE1 nunca probado: los NODOS VIRTUALES.** El manuscrito pide ablación de los
  tres componentes del grafo de interacción (vecindario local, expander, nodos virtuales) y solo
  se probaron los dos primeros. Nota de riesgo: la bitácora ya registra que "WN18RR selecciona
  en contra de cualquier mecanismo global", así que la predicción es que aporten poco — pero es
  un hueco explícito del objetivo.
- **No decir en la defensa** que el expander captura dependencias globales (los datos dicen que
  no), ni apoyar la defensa en el full attention (no es el objeto de la tesis).

---

## 2026-08-08 (e) — CARACTERIZACIÓN de costo (complejidad, parámetros, velocidad) de los 4 modelos ⇒ nueva hipótesis: **la brecha podría ser de CAPACIDAD RELACIONAL** (NBFNet da 14× más parámetros por relación) ⇒ implementado `--rel_param lowrank` y lanzado el barrido

**Contexto**: pregunta del usuario sobre complejidad, conteo de parámetros y velocidad. Al
medirlos aparece una asimetría que la bitácora no tenía registrada y que sugiere la primera
hipótesis nueva desde que se agotaron las de agregación y las de encodings estructurales.

### Complejidad por capa (derivada del código)

Con **B** queries en paralelo, **N** nodos, **E** aristas dirigidas, **d** hidden, **H** cabezas,
**R** relaciones, **D** grado del expander. El labeling trick hace que `x⁰` dependa de la query
⇒ **cada query necesita su propia copia de los estados de nodo**: todo es `(B,N,d)` y `(B,E,d)`,
no hay cómputo compartido entre queries del batch.

| modelo | tiempo | espacio (activaciones) |
|--------|--------|------------------------|
| NBFNet | `O(B·E·d + B·N·d² + B·R·d²)` | `O(B·E·d + B·N·d + B·R·d)` |
| GT Full | `O(B·N²·d + B·E·d + B·N·d²)` | `O(B·H·N² + B·E·d + B·N·d)` |
| GT Sparse | `O(B·E·d + B·N·d²)` | `O(B·E·d + B·N·d)` |
| GT Sparse+exp | `O(B·(E + D·N)·d + B·N·d²)` | `O(B·(E + D·N)·d + B·N·d)` |

El cociente que decide full-vs-sparse es **H·N² contra E·d**: 31× en ind v1, 38× en ind v2,
69× en WN18RR v1 y **94× en FB15k-237 transductivo** (donde el full directamente no corre).
Corolario verificado: el costo por época es `O(|T|·L·E·d)` y **el batch se cancela** — el
throughput en "aristas×query/s" es estable a través de configs (27.3 M/s en FB237 transductivo
con B=8, 24.0 M/s en WN18RR transductivo con B=32, pese a 3.4× de diferencia en E y 4× en B).

### Parámetros (medidos instanciando los modelos)

| dataset | R | **NBFNet** (d32,L6,pna) | **GT Full** (d64,L6,H8) | **GT Sparse** | **GT Sparse+exp** |
|---|---:|---:|---:|---:|---:|
| FB15k-237 ind v1 | 360 | **2 374 017** | 383 745 | 384 177 | 384 609 |
| FB15k-237 ind v2 | 400 | **2 628 737** | 403 585 | 404 017 | 404 449 |
| FB15k-237 transd. | 474 | **3 099 969** | 440 289 | 440 721 | 441 153 |
| **WN18RR transd.** | **22** | **221 633** | **216 097** | 216 529 | 216 961 |

Fórmulas verificadas al parámetro exacto:
`NBFNet = L·(R·d² + R·d + 13d² + 3d) + R·d + (d²+d) + (d+1)` ·
`GT = L·(8d² + H·R + R·d + 7d) + R·d + 2d + (d²+d+1)`.
Los tres GT difieren entre sí por **432 = 6 capas × 72**, que es el costo de una relación extra
(`H+d`): el sparse reserva una para el self-loop y el expander otra para `R_exp`.

### Velocidad (misma A100, mismo batch 16, mismo protocolo, FB15k-237 ind v1)

| modelo | params | train | inferencia | 50 ép + test |
|--------|-------:|------:|-----------:|-------------:|
| **NBFNet** | 2 374 017 | **9.4 s/ép** | **150 it/s** | **7.8 min** |
| GT Sparse | 384 177 | 11.2 s/ép | 127 it/s | 9.3 min |
| GT Full | 383 745 | **88.7 s/ép** | **23.7 it/s** | **74.0 min** |

**Análisis (causa mecánica)**

- **(A) Menos parámetros NO es más rápido — es al revés.** NBFNet tiene **6.2× más parámetros**
  y es **9.4× más rápido en train y 6.3× en inferencia** que el full attention. Los 2.37 M de
  NBFNet viven en `relation_linear`, que se aplica **una vez por query**: cuesta `B·R·d²` ≈ 5.9 M
  FLOPs por capa, **440× menos** que el `B·N²·d` ≈ 2 602 M del full. *Los parámetros son memoria
  estática; el costo es memoria de activaciones.* Firma: 11.8 GB del full contra 1.0 GB de NBFNet.
- **(B) El full attention es peor en las TRES dimensiones a la vez**: 9.4× más lento, 12× más
  memoria, 6× menos parámetros, y pierde por 0.0816 ± 0.0020. Importa decirlo explícito en la
  tesis porque **desarma la objeción obvia** ("la atención tenía más capacidad y sobreajustó"):
  tiene *menos*. El sparse queda mejor parado (11.2 vs 9.4 s/ép) pero pierde por 0.061 ± 0.008.
- **(C) HIPÓTESIS NUEVA: la brecha podría ser de capacidad RELACIONAL.** NBFNet le da a cada
  relación una **matriz d×d por capa** (`dependent: yes` ⇒ `R·d²`, 1024 params por relación);
  los GT le dan un **escalar por cabeza + un vector diagonal DistMult** (`R·(H+d)`, 72). **14×.**
  Correlación sugerente: donde NBFNet tiene 6.2× más capacidad relacional (FB15k-237, R=360-474)
  gana por 0.061-0.082; donde los conteos casi coinciden (**WN18RR, R=22: 222 K vs 216 K**) el
  sparse **EMPATA** (0.738 vs 0.740). ⚠️ Son **dos datasets y está confundido con la estructura
  del grafo** (local/jerárquico vs composicional), que es la explicación que la bitácora ya
  sostiene. Es hipótesis, no hallazgo — pero es testeable y barata.
- **(D) Por qué esta hipótesis no estaba en la lista negra y merece probarse.** Todas las
  intervenciones previas atacaron la **agregación** (opciones A y C: sigmoid, degree, anchor,
  rel — las cuatro refutadas) o metieron estructura como **feature de nodo** (lista negra #7:
  RWSE, LapPE, source_rw — las tres refutadas). Ninguna tocó **cuánta capacidad tiene el modelo
  por relación**. Y es inductive-safe por construcción: las relaciones se comparten train/test,
  así que no roza la lista negra #3. Además el costo en tiempo es **casi nulo** (el término
  `B·R·d²` es marginal, como demuestra el propio NBFNet), o sea es barata de probar — al revés
  de lo que sugería la intuición de "más parámetros = más lento".

**Implementación (HECHA): `--rel_param {diag,lowrank} --rel_rank k`**

La modulación relacional del valor pasa de diagonal a diagonal + corrección de rango bajo:
`diag: msg = v[src] ⊙ g[rel]` · `lowrank: msg = v[src] ⊙ g[rel] + (v[src] @ U[rel]) @ W[rel]ᵀ`
con `U, W` de forma `(H, R, hd, k)`; añade `2·H·R·hd·k` params por capa; `k = hd` da rango
completo. Cableado en las dos capas de atención ⇒ lo heredan los tres GT. Bloqueado por assert
con `exp_typing` ultra/path (recomponen solo el término diagonal; mezclaría dos semánticas).

Params resultantes (ind v1): diag 384 K · k=2 936 K · **k=4 1.49 M** · k=8 2.60 M ·
*NBFNet 2.37 M*. **k=4 queda POR DEBAJO de NBFNet** ⇒ si cierra brecha ahí no se puede atribuir
a tener más parámetros que el baseline; **k=8 es el punto parameter-matched**.

**Dos bugs que atrapó el smoke test** (se registran porque el primero habría costado GPU y
producido una conclusión falsa):
- **La parametrización estaba MUERTA.** Con `U` y `W` ambos init en cero —lo natural para
  garantizar equivalencia al arranque— resulta `∂/∂U ∝ W = 0` y `∂/∂W ∝ vU = 0` ⇒ **gradiente
  cero en las 6 capas, para siempre**. Habría entrenado y devuelto exactamente `diag`,
  concluyendo "la capacidad relacional no aporta" sin haberla probado. Corregido con init
  estilo **LoRA** (U aleatorio σ=0.02, W en cero): el producto sigue siendo 0 al arranque pero
  ambos reciben gradiente. ⇒ **el test de gradiente hay que hacerlo DESPUÉS de un paso de
  optimizador**, no en el paso 0, donde `∂/∂U = 0` es correcto por construcción.
- Colisión de nombres: `rel_v` es substring de `rel_value` y contaminaba los conteos. Renombrado
  a `rel_w`.

**Verificado**: forward **idéntico bit a bit** a `diag` en la inicialización en los 3 modelos
(igualando los pesos compartidos, porque el `randn` de `rel_u` corre el stream del RNG);
`--rel_param diag` sin regresión; gradiente en `rel_u`/`rel_w` en 6/6 capas tras un paso;
guard de `exp_typing` activo.

**Decisión / EN CURSO**

- **Barrido lanzado** (FB15k-237 ind v1, 50 ép, n=3 semillas 42/43/44), jobs **91459** (sparse
  +edrop 0.2: diag, k=2, k=4, k=8 — 12 corridas, ~2 h), **91460** (full: diag, k=4) y **91461**
  (full: k=8) — 9 corridas de ~74 min. Script `sbatch_relparam_v1.sh`.
- **El brazo `diag` se re-corre** aunque ya existe (0.3764 ± 0.0021 y 0.397 ± 0.019): `model.py`
  y `train.py` cambiaron hoy, y comparar contra otra versión del harness es el error que esta
  bitácora ya corrigió tres veces. El control es del código de hoy.
- **Predicción falsable, escrita antes de ver los números**: si la brecha es de capacidad
  relacional, el MRR debe subir **monótonamente con k**, con el salto grande entre `diag` y
  `k=2`. Si sube y satura, la capacidad importaba pero no era el cuello de botella. **Si es
  plano dentro del ruido —lo más probable dado que sigmoid/degree/anchor/rel rindieron todos lo
  mismo— la hipótesis de capacidad queda refutada** y refuerza que el problema es la agregación
  aprendida, no cuántos parámetros hay por relación.
- Al leer: σ del sparse ≈ 0.019 ⇒ ahí solo un Δ > ~0.04 es interpretable con n=3. **La fila que
  decide es la del full** (σ = 0.0021), que resuelve efectos chicos.
- ⚠️ `lowrank` **no está pensado para transductivo** tal como está: el gather de U/W es
  `O(H·E·hd·k)` y con E = 558 K pesa ~1 GB por capa. Documentado en el código.
- Artefactos: `src/model.py` (bloque de docs `--rel_param` + las dos capas),
  `train.py` (`--rel_param`, `--rel_rank`), `sbatch_relparam_v1.sh`, `logs/relparam_v1_914{59,60,61}.log`.

---

## 2026-08-08 (d) — `remove_one_hop`: implementábamos la rama DÉBIL de NBFNet; el **31.6 %** de los triples de train en FB15k-237 transductivo conserva un atajo de 1 salto que su config sí elimina

**Contexto**: pregunta del usuario a partir del paper de NBFNet ("*during training, we drop out
edges that directly connect query node pairs*"). Al comparar los códigos aparece que hacemos
**algo**, pero no lo mismo.

### Qué hace cada uno

Semánticamente idénticos en todo lo demás (batch-wide, solo en train, ambas direcciones); la
diferencia está en **qué se matchea**:

| | qué quita |
|---|---|
| Nuestro `remove_edge` (`train.py`) | `(h,r,t)` y `(t,r⁻¹,h)` — **con la relación exacta** |
| NBFNet `remove_one_hop: False` | idéntico a lo nuestro |
| **NBFNet `remove_one_hop: True`** ← su config inductivo | **todas** las aristas entre h y t, sin mirar relación |

O sea implementábamos su rama `else` y su config inductivo usa la otra
(`NBFNet-PyG/nbfnet/models.py:54`; su propio comentario la llama *"dynamic edge dropout"*).

### Cuánto importa (medido sobre nuestros datos)

Fracción de triples de train que **conservan una arista directa h–t** bajo el modo débil:

| dataset | triples | con atajo residual |
|---|---:|---:|
| FB15k-237 ind v1 | 4 245 | **19.6 %** |
| FB15k-237 ind v2 | 9 739 | **23.1 %** |
| **FB15k-237 transductivo** | 272 115 | **31.6 %** |
| WN18RR ind v1 | 5 410 | 1.0 % |
| WN18RR transductivo | 86 835 | 0.5 % |

**Análisis**

- En FB15k-237 **una de cada cinco queries de train (una de cada tres en transductivo) se puede
  resolver sin razonar por caminos**: el par ya está conectado directo por otra relación. No es
  fuga de la respuesta exacta —la relación difiere— pero sí un atajo estructural.
- **Predicción**: debería pesar en FB15k-237 y ser nulo en WN18RR. Es la misma firma de
  `--edge_drop`, que mide +0.060 ± 0.014 en FB237 (replicado v1 y v2) y **exactamente 0.000
  (t=0.00)** en WN18RR. `remove_one_hop` es la versión **dirigida** de la misma idea (quita justo
  las aristas que trivializan la query) y `--edge_drop` la **aleatoria** ⇒ ojo, es probable que
  sean **parcialmente redundantes**: hay que medirlo como brazo aparte y combinado, no encima de
  la best config actual.
- Efecto diferencial esperado: el atajo de 1 salto es **máximamente explotable por el sparse**,
  que atiende sobre adyacencia con peso aprendido y puede privilegiar esa arista; NBFNet propaga
  por el mismo salto con agregación **fija**. Como el sparse es el modelo diagnosticado por
  sobreajuste topológico, quitarlo debería castigar su ajuste in-distribution y quizá mejorar su
  transferencia.
- ⚠️ **CORRECCIÓN a CLAUDE.md**: dice que nuestro NBFNet está *"alineado a
  `NBFNet/config/inductive/fb15k237.yaml`"*. **En este flag no lo está.** Todos nuestros números
  de NBFNet (0.458 v1, 0.526 v2) son NBFNet **sin** `remove_one_hop`, entrenado con ~20 % de
  queries trivializables que su config elimina. Dirección del efecto: desconocida.

**Implementación (HECHA)**: `--remove_one_hop` en `train.py`; el hash pasa de matchear el triple
`(src,rel,dst)` a matchear solo el par `(src,dst)`. Default `False` ⇒ comportamiento previo
intacto. Smoke test sobre un batch real de FB15k-237 transductivo: OFF quita exactamente **16**
aristas (2 × 8 queries, el histórico), ON quita **28** y es **superconjunto estricto**.

**Decisión / EN CURSO**

- Lanzado job **91458**: `--model sparse --remove_one_hop` en **FB15k-237 transductivo**, 2×A100
  DDP, 1 semilla (42), 20 ép, L4/dim32, **batch global 8** (4/GPU). Medido: 5.36 it/s ⇒ ~2 h/época
  ⇒ **~40 h**. Script `sbatch_onehop_trans.sh` (encadenable + watchdog).
- **Control**: `sparse softmax L4/d32 batch 8` = test **0.3965** (n=1, seed 42, jul-2026).
  ⚠️ Dos salvedades serias: (i) es n=1 y de julio, y con σ del sparse en 0.019-0.054 un
  |Δ| < ~0.05 no será distinguible de ruido; (ii) se corrió con el código de entonces. El brazo
  de control con el código de hoy está implementado (`sbatch sbatch_onehop_trans.sh control`) y
  **queda pendiente de lanzar**.
- El batch global 8 no es arbitrario: la bitácora corrigió dos veces (2026-08-03 y 2026-08-07)
  que ese baseline corrió a global 8 y no 16. Mantenerlo es lo único que hace comparable el
  resultado.
- Artefactos: `train.py::remove_edge`, `sbatch_onehop_trans.sh`, `logs/onehop_trans_91458.log`.

---

## 2026-08-08 (c) — NBFNet corrido desde el CÓDIGO DE LOS AUTORES en FB15k-237 ind v2: reproduce su número PUBLICADO (H@10_50 0.9472 vs ~0.941) pero su MRR full-filtered es 0.359, no 0.526 ⇒ **nuestro NBFNet con CE de grafo completo es MÁS FUERTE que la receta publicada, y la lista negra #5 queda respaldada con el código de ellos**

**Contexto**: la entrada (a) de hoy dejó el `sparse_exp+edge_drop` de v2 en 0.5275 ± 0.0032
rozando el 0.526 de NBFNet, pero ese 0.526 es n=1 a 20 ép de **nuestra** reimplementación.
Antes de re-medirla convenía una referencia **externa**: el código de los autores, su config,
sobre exactamente los mismos datos. Se corrió en 1 GPU, 3 semillas (1024/1025/1026, 1024 es
el default de su `run.py`), config sin tocar salvo `output_dir` (verificado con `diff`: 1
línea): input_dim 32, hidden_dims [32]×6 (6 capas), distmult, pna, short_cut, layer_norm,
dependent, remove_one_hop, lr 5e-3, batch 64, 20 épocas, **criterion BCE con 32 negativos**
y adversarial_temperature 0.5.

**Datos**: descargados de las mismas URLs de GraIL y verificados **byte-idénticos** a
`data/inductive/fb15k-237_v2{,_ind}/` (`diff -q`, los 4 archivos). El dataset procesado da
train graph N=2608 y test graph N=1660, que coincide con lo registrado en la entrada del
2026-06-23. La comparación es sobre los mismos datos, no sobre una copia parecida.

### Resultado (FB15k-237 ind v2, 20 ép, n=3, job 91457)

| seed | test_mrr | H@1 | H@10 | MR | **H@10_50** | best ép |
|------|---------:|----:|-----:|---:|------------:|--------:|
| 1024 | 0.3940 | 0.247 | 0.676 | 49.0 | 0.9567 | **2** |
| 1025 | 0.3069 | 0.181 | 0.566 | 83.3 | 0.9355 | **4** |
| 1026 | 0.3758 | 0.223 | 0.666 | 59.3 | 0.9494 | **2** |
| **media ± sd** | **0.3589 ± 0.0459** | 0.217 | 0.636 | 63.8 | **0.9472 ± 0.0108** | — |

| referencia | MRR full-filtered | H@10_50 |
|---|---:|---:|
| **NBFNet-PyG (autores, BCE-k)** | **0.3589 ± 0.0459** | **0.9472 ± 0.0108** |
| *paper NBFNet, v2, Hits@10(50 muestras)* | — | *~0.941* |
| **NBFNet nuestro (CE de grafo completo)** | **0.526** (n=1, 20 ép) | — |
| sparse_exp + edge_drop 0.2 (entrada (a)) | 0.5275 ± 0.0032 | — |

**Análisis (causa mecánica)**

- **(A) La corrida es CORRECTA: reproduce el número publicado, en la métrica que el paper
  reporta.** NBFNet publica los resultados inductivos de FB15k-237 como **Hits@10 contra 50
  negativos muestreados** (protocolo GraIL), no como MRR full-filtered. Ahí mide **0.9472 ±
  0.0108** contra el ~0.941 del paper. No hay bug: el setup está reproducido.
- **(B) Pero su MRR full-filtered es 0.359, un tercio por debajo de nuestro 0.526 — y la
  firma dice exactamente por qué.** Durante el entrenamiento la loss baja monótona (BCE 0.515
  → 0.331) mientras el **valid MRR BAJA** monótona (0.384 → 0.305); el best-valid cae en la
  **época 2-4 de 20**. El H@10_50, en cambio, se queda clavado en ~0.95 todo el entrenamiento.
  Mecanismo: entrenar BCE contra **32 negativos muestreados** y evaluar contra **50** es
  discriminar entre un puñado de candidatos — no exige ordenar bien contra las ~1660 entidades
  del grafo. El modelo mejora en su objetivo y se degrada en ranking completo.
- **(C) Replicado en DOS implementaciones independientes ⇒ no es un artefacto de entorno.**
  Se corrió primero el repo `NBFNet/` (torchdrug, job 91453) y dio **el mismo patrón**: loss
  0.519 → 0.356 y valid MRR 0.210 → 0.172. Esa corrida se descartó por rota —el stack de
  torchdrug 0.2.1 sobre torch 2.1 acumula cuatro incompatibilidades y encima crashea al cargar
  el best ckpt (`'Graph' object has no attribute 'is_meta'`), así que nunca llegó al test— pero
  **la dirección del valid MRR coincide** con la de NBFNet-PyG, que sí completó. Dos codebases
  distintas, misma receta, mismo comportamiento.
- **(D) CONSECUENCIA PRINCIPAL: nuestro NBFNet no está inflado — está entrenado mejor, y eso
  valida la lista negra #5 con el código de los autores.** El 0.526 de la bitácora sale de la
  MISMA arquitectura con **CE de grafo completo** en vez de BCE-k. La diferencia 0.526 vs 0.359
  **es la loss**, no la implementación. CLAUDE.md lista negra #5 ("NO BCE-k negative sampling
  como loss principal → usar CE de grafo completo") era hasta hoy una decisión de diseño **sin
  medición**; ahora tiene evidencia directa: sobre los mismos datos y la misma arquitectura, la
  receta publicada rinde **−0.167 de MRR** respecto a la nuestra. Corolario para la tesis: los
  números inductivos de la literatura reportados en H@10_50 **no son comparables** con MRR
  full-filtered, y usarlos como techo sería un error de protocolo.
- **(E) Su varianza en MRR es enorme: σ = 0.046 (rango 0.307–0.394), n=3.** Contra σ = 0.004 de
  nuestro NBFNet con CE. Coherente con (B): si el objetivo no es el ranking completo, dónde
  aterriza el MRR depende mucho de la semilla. En H@10_50 —su métrica— la σ cae a 0.011.

**Decisión**

- **La referencia externa NO reemplaza al brazo de comparación.** Para contrastar contra el
  0.5275 ± 0.0032 del `sparse_exp` sigue haciendo falta **nuestro NBFNet en v2 a 50 ép, n≥3**
  (CE de grafo completo). Sigue siendo la prioridad #1 y es barata (~1-2 h).
- **Lista negra #5 pasa de decisión de diseño a resultado medido** (ver D). Vale la pena
  registrarlo también en CLAUDE.md.
- **El repo `NBFNet/` (torchdrug) queda marcado como NO USABLE en este cluster**: sin CUDA
  toolkit (no hay nvcc en el login ni en nodeGPU01) y con torch 2.1, acumula conflicto de
  metaclase PyG/torchdrug, header `ATen/SparseTensorUtils.h` movido, `ninja` ausente y el crash
  de `is_meta`. **Usar `NBFNet-PyG/`** (reimplementación oficial del mismo autor, linkeada en el
  README del repo original), que corre sobre torch+PyG+torch-scatter, o sea el stack instalado.
- **Nota de infraestructura, relevante para cualquier corrida futura de NBFNet oficial**: ambos
  repos traen un kernel CUDA `rspmm` que se compila en JIT y acá **no se puede compilar**. Los
  dos definen su propia salida: `NBFNet-PyG/nbfnet/layers.py:69-72` rutea a
  `super().propagate()` (message+aggregate estándar de PyG) cuando no puede usar el kernel, y
  el comentario del propio repo (línea 154) describe el kernel como la *"fused computation of
  message and aggregate steps"* — misma matemática, más rápida y con menos memoria. Los
  wrappers fuerzan esa rama con `NBFNET_PYG_NO_RSPMM=1` / `NBFNET_NO_RSPMM=1`. **Equivalencia
  verificada**: con el modelo sin entrenar, CPU-con-kernel y GPU-con-fallback dan las mismas
  métricas a 6 decimales (valid mrr 0.00585632 en ambos; test 0.00691717 vs 0.00691718).
- Artefactos: `NBFNet-PyG/` (clonado), `run_nbfnet_pyg.py` y `run_nbfnet_orig.py` (wrappers;
  **ninguno modifica los repos originales**), `sbatch_nbfnet_pyg_v2.sh`,
  `sbatch_nbfnet_orig_v2.sh`, `nbfnet_pyg_cfg/`, `nbfnet_orig_cfg/`,
  `logs/nbfnet_pyg_v2_91457.log`, `logs/nbfnet_orig_v2_91453.log` (descartado),
  `experiments_nbfnet_pyg/`. Entorno: venv aislado `/home/jreutter/venv_nbfnet`
  (`--system-site-packages` sobre `attention` ⇒ **no se tocó el env de los jobs en curso**).

---

## 2026-08-08 (b) — RESULTADO de `sparse_exp` deg3 en WN18RR TRANSDUCTIVO: nulo (Δ +0.0030), cuarto régimen consecutivo ⇒ lista negra #8 confirmada en los cuatro

**Contexto**: es el job anunciado como "en curso" al cierre de la entrada (a). Cierra el
único régimen donde el expander no se había probado. 2×A100 DDP a batch 16/GPU (global 32,
apples-to-apples con el baseline sparse softmax), 20 épocas, seed 42, L4/dim32, lr 1e-3,
drop 0.0. Job 91434, 12 h de wall.

### Resultado (WN18RR transductivo, full-filtered, n=1)

| variante | valid_mrr | **test_mrr** | H@1 | H@3 | H@10 | MR | best ép |
|----------|----------:|-------------:|----:|----:|-----:|---:|--------:|
| **sparse_exp deg3** | 0.5531 | **0.5566** | 0.511 | 0.578 | 0.647 | 1497 | 18 |
| *sparse softmax (ref, 2026-08-07)* | *0.5481* | *0.5536* | *0.508* | *0.576* | *0.644* | *1616* | *15* |

**Δ test = +0.0030**; las 4 métricas de ranking coinciden en el segundo decimal.

**Análisis (causa mecánica)**

- **Misma firma de empate métrica-por-métrica** con la que se cerró el expander en los otros
  tres regímenes: FB15k-237 transductivo Δ +0.0008 (2026-08-03), FB15k-237 ind v1 +0.011 con
  edge dropout y WN18RR ind v1 **−0.006** (2026-08-05 b), FB15k-237 ind v2 +0.0118 (entrada
  (a) de hoy). **Cuatro regímenes, ningún efecto que sobreviva a su costo.**
- n=1 ⇒ el +0.0030 **no es distinguible de cero**; lo que sostiene la lectura no es la
  magnitud sino el patrón de coincidencia una a una, igual que en las entradas anteriores.
- **Costo medido, y es el dato transferible**: el expander cuesta **~4.0× por época** en
  WN18RR contra 1.67× en FB15k-237. La razón es el conteo, no la localidad: deg3 añade
  6·N = 245 658 aristas sobre solo 214 613 reales ⇒ **+114 % de aristas** (en FB237 era
  +15.6 %, porque tiene 2.8× menos nodos y 2.6× más aristas). Medido: 38 min/época y
  **25.7 GB/GPU**; a batch 32 en 1 GPU habrían sido ~50 GB ⇒ **OOM en la A100-40GB**, de ahí
  el DDP. Regla práctica: el costo del expander escala con **N/|E|**, así que es caro
  justo en los grafos dispersos, que son donde uno esperaría que los atajos sirvieran.

**Decisión**

- **Lista negra #8 confirmada en los cuatro regímenes** (FB237 transductivo e inductivo v1/v2,
  WN18RR inductivo v1 y transductivo). No se gasta más GPU en expander.
- Artefactos: `sbatch_sparse_exp_wn_trans.sh`, `experiments/sparse_exp_wn_trans/`,
  `logs/sparse_exp_wn_trans_91434.log`.

---

## 2026-08-08 (a) — RÉPLICA en FB15k-237 ind v2 del 2×2 {expander × edge dropout}: el EDGE DROPOUT replica limpio (+0.060, t=4.30), la hipótesis "el expander era un regularizador" SE DEBILITA, y el mejor brazo (0.5275 ± 0.0032) roza a NBFNet — pero la comparación NO está establecida

**Contexto**: dos huecos del protocolo. (i) El edge dropout es **la única intervención con
efecto positivo grande y significativo del proyecto** (+0.057 ± 0.013, t=4.23 en v1) y
**nunca se había probado fuera de v1** — y este proyecto ya se quemó dos veces con efectos de
un solo split (LapPE, sparse+RWSE). (ii) El hallazgo (A) del 2026-08-05 b —el expander
funciona como REGULARIZADOR, no como atajo estructural— se apoyaba en que su beneficio se
derrumba al añadir edge dropout (v1: +0.071 → +0.011) y en que desaparece donde el edge
dropout no sirve (WN18RR, ambos 0.000). v2 es el test que faltaba: FB237 composicional como
v1, pero split más grande y fácil. Se corrió el **2×2 completo** {expander sí/no} ×
{edge_drop 0.2/0.0}, 50 épocas, n=3 semillas 42/43/44, **mismas semillas y mismas épocas en
los 4 brazos**. Best config del sparse (L6, dim 64, 8 heads, drop 0.0, lr 1e-3, batch 16,
wd 1e-4); expander `deg3 typing=single`. Jobs 91435, 91436, 91445 — 12 corridas.

### Resultado (FB15k-237 ind v2, full-filtered, 50 ép, n=3)

| brazo | **test_mrr** | **sd** | H@1 | H@3 | H@10 | MR | valid | gap val→test | best ép |
|-------|-------------:|-------:|----:|----:|-----:|---:|------:|-------------:|--------:|
| **sparse_exp deg3 ed=0.2** | **0.5275** | **0.0032** | 0.426 | 0.590 | 0.706 | **52.5** | 0.484 | **−0.043** | **44.0** ⚠️ |
| sparse ed=0.2 | 0.5157 | 0.0088 | 0.417 | 0.571 | 0.694 | 79.9 | 0.485 | −0.030 | 18.7 |
| sparse_exp deg3 ed=0.0 | 0.4736 | 0.0120 | 0.381 | 0.524 | 0.632 | 103.4 | 0.475 | +0.001 | 10.3 |
| sparse ed=0.0 | 0.4556 | 0.0225 | 0.365 | 0.505 | 0.606 | 103.8 | 0.475 | +0.020 | 10.7 |
| *NBFNet (ref, n=1, **20 ép**)* | *0.526* | — | *0.416* | *0.595* | *0.727* | *49.3* | *0.483* | — | — |

Contrastes (Welch, mismas semillas, todo a 50 ép), con v1 al lado:

| contraste | **v2 (nuevo)** | *v1 (2026-08-05 b)* |
|---|---:|---:|
| edge dropout 0.2 vs 0.0, **sin** expander | **+0.0601 ± 0.0140 (t=4.30)** | *+0.091 ± 0.014 (t=6.59)* |
| edge dropout 0.2 vs 0.0, **con** expander | +0.0539 ± 0.0072 (t=7.52) | — |
| expander vs sin expander, a ed=0.0 | +0.0179 ± 0.0147 **(t=1.22, n.s.)** | *+0.071 (t=4.73)* |
| expander vs sin expander, a ed=0.2 | +0.0118 ± 0.0054 (t=2.19) | *+0.011 (t=2.43)* |

**Análisis (causa mecánica)**

- **(A) El EDGE DROPOUT REPLICA — es el primer efecto del proyecto que sobrevive un cambio de
  split con barras de error.** +0.0601 ± 0.0140 (t=4.30) en v2 contra +0.057 ± 0.013 (t=4.23)
  en v1: magnitud, error estándar y t casi idénticos, n=3 y n=6, splits distintos. Contrasta
  con LapPE (+0.035 v1 → +0.005 v2) y sparse+RWSE (−0.046 v1 → +0.022 v2), que murieron o
  invirtieron signo en la réplica. **`--edge_drop 0.2` deja de ser "parte de la best config del
  sparse en v1" y pasa a ser la best config del sparse en FB15k-237.** (En WN18RR sigue siendo
  0.000 — la dependencia de dataset del 2026-08-05 b se mantiene, no la de split.)
- **(B) El mecanismo se ve en el gap val→test, y en v2 CAMBIA DE SIGNO.** Sin edge dropout el
  sparse tiene gap +0.020 (valid > test, la firma de sobreajuste estructural de siempre); con
  edge dropout el gap es **−0.030**, o sea el modelo transfiere al grafo disjunto **mejor** que
  in-distribution. Con expander + edge dropout llega a −0.043. Perturbar la estructura en train
  no solo cierra la brecha de transferencia: la invierte. Misma dirección que el barrido de p
  del 2026-08-05 (gap monótono decreciente 0.253 → 0.036), ahora cruzando el cero.
- **(C) La hipótesis "el expander ERA un regularizador" (hallazgo A del 2026-08-05 b) SE
  DEBILITA — no se refuta, pero pierde su evidencia principal.** Esa lectura se apoyaba en que
  el beneficio del expander aparece y desaparece exactamente donde aparece y desaparece el
  beneficio de regularizar. En v2 el edge dropout **sí** vale mucho (+0.060, t=4.30) ⇒ hay
  sobreajuste estructural que amortiguar ⇒ (A) predice que el expander sin regularizar debería
  comprar algo del orden de +0.06. **Midió +0.018 y no es significativo (t=1.22).** La razón
  expander/edge-dropout cae de 0.78 en v1 a 0.30 en v2. Direccionalmente (A) sobrevive (donde
  regularizar vale 0 —WN18RR— el expander vale 0), pero la **equivalencia** cuantitativa que
  sostenía el argumento es de v1. Formulación honesta: *el expander es un regularizador peor
  que el edge dropout en los dos splits de FB237, y en v2 apenas lo es.*
- **(D) Hay un residuo del expander SOBRE el edge dropout, chico y notablemente replicable:
  +0.0118 (v2) vs +0.011 (v1).** Dos splits, n=3, mismo signo y prácticamente la misma
  magnitud — más consistente que casi cualquier ablation del proyecto. No es explicable por
  (A) (el edge dropout ya está puesto en ambos brazos). Pero cuesta **1.85× por época** (9.1
  vs 16.8 it/s en v2, coherente con el +71 % de aristas: 6×2608 = 15 648 expander sobre 22 086
  reales) y sigue sin superar a NBFNet de forma establecida ⇒ **no rehabilita la línea**
  (lista negra #8), la matiza: el efecto es real y pequeño, no cero.
- **(E) El mejor brazo ROZA a NBFNet pero LA COMPARACIÓN NO ESTÁ ESTABLECIDA, en las dos
  direcciones.** `sparse_exp + edge_drop 0.2` mide **0.5275 ± 0.0032** contra el **0.526**
  registrado de NBFNet en v2 — sería el primer modelo de atención en alcanzar a NBFNet en
  FB15k-237 **sin heredar su V** (`sparse_nbfv` lo hace por construcción, ver 2026-06-23). Pero:
  (i) **el 0.526 de NBFNet es n=1 a 20 épocas**, exactamente el protocolo que esta bitácora
  declara insuficiente — no hay contraste posible contra un número sin barra;
  (ii) **el 0.5275 es un PISO**: los best-valid de sus 3 semillas caen en las épocas **39, 45 y
  48 de 50**, dos de tres todavía mejorando al final. El protocolo del 2026-08-05 pide
  explícitamente verificar que el best-valid no caiga en las últimas épocas; acá cae.
  ⇒ **No se puede afirmar ni "empata" ni "no empata".** Es la medición que falta, y es barata.
  Nótese que el MR sí es llamativamente bueno (**52.5**, contra 49.3 de NBFNet y 79.9 del
  sparse sin expander): el expander mejora fuerte la cola del ranking.
- **(F) Corrección fila-por-fila, del mismo tipo que la del full attention.** El sparse v2 sin
  edge dropout mide **0.4556 ± 0.0225** a 50 ép contra el **0.450** registrado a 20 ép: Δ
  **+0.006**. O sea **en v2 el sparse tampoco estaba subentrenado a 20 épocas**; el +0.093 de
  20→50 ép era **específico de v1**. Tercera vez que una advertencia formulada "por familia"
  resulta ser de una fila concreta (antes: el full no estaba subentrenado, la inestabilidad era
  del sparse y no de la atención). **Regla que se confirma: en este proyecto ninguna propiedad
  medida en un brazo se hereda a otro brazo, ni a otro split, sin re-medirla.**

**Decisión**

- **El edge dropout queda establecido como efecto real y robusto a split en FB15k-237** (dos
  splits, n≥3, t>4 en ambos). Es el único resultado positivo del proyecto con esa credencial.
- **La hipótesis (A) del 2026-08-05 b se marca como debilitada**, no refutada. La entrada de esa
  fecha NO se reescribe; esta la matiza. Lo que cae es la equivalencia expander≡regularización;
  lo que queda en pie es que el expander no da atajos estructurales genuinos.
- **La lista negra #8 NO se toca**: el expander sigue costando 1.85× para un residuo de +0.012.
- **Prioridad #1 nueva y bloqueante: NBFNet en v2 a 50 ép, n≥3** (~1–2 h). Sin eso el 0.5275 no
  es interpretable. Segunda: **sparse_exp ed=0.2 en v2 a 100 ép, n=3** (~4 h) para saber dónde
  converge el mejor brazo. Ambas pendientes; hay GPUs libres.
  ⚠️ `sbatch_edrop_v2.sh` tiene `--model sparse` fijo — hay que extenderlo para NBFNet.
- Artefactos: `sbatch_edrop_v2.sh`, `sbatch_exp_v2.sh`, `experiments/edrop_v2_sparse_p{02,00}_ep50_s{42,43,44}/`,
  `experiments/exp_v2_deg3_p{02,00}_ep50_s{42,43,44}/`, `logs/edrop_v2_9143{5,6}.log`,
  `logs/exp_v2_91445.log`.

~~**En curso al cierre de esta entrada**: job **91434** — `sparse_exp deg3` en WN18RR
TRANSDUCTIVO~~ ⇒ **TERMINADO, ver entrada 2026-08-08 (b)**: test 0.5566 vs 0.5536 del sparse
softmax, Δ **+0.0030** ⇒ el nulo esperado, cuarto régimen consecutivo.

---

## 2026-08-07 (b) — FULL ATTENTION re-medido con el protocolo vigente (prioridad #1 de la cola, CERRADA): el 0.375 histórico era CORRECTO, el full NO estaba subentrenado, el edge dropout es NULO en el full, y la inestabilidad NO es de la familia de atención sino del sparse

**Contexto**: es la prioridad #1 abierta el 2026-08-05. El full attention es el modelo de la
**pregunta central de GOALS.md** y su único número registrado (0.375) era n=1 a 20 épocas, o
sea el régimen que esa misma sesión demostró sesgado CONTRA los modelos de atención. Nunca se
había corrido regularizado ni entrenado lo suficiente. Config: best config del full (L6,
dim 64, 8 heads, drop 0.0, lr 1e-3, batch 16, wd 1e-4), 50 épocas, semillas 42/43/44, dos
brazos de `--edge_drop`. **Idéntico flag por flag a `sbatch_edrop_final.sh`** (el script que
produjo el 0.397 del sparse y el 0.458 de NBFNet) salvo `--model rfat`. Jobs 91381-91384,
91387-91388; una semilla por GPU en paralelo.

### Resultado (FB15k-237 ind v1, full-filtered, 50 ép, n=3, full-filtered)

| brazo | **test_mrr** | **sd** | rango | H@1 | H@3 | H@10 | MR | valid | gap val→test |
|-------|-------------:|-------:|------|----:|----:|-----:|---:|------:|-------------:|
| full `--edge_drop 0.0` | **0.3764** | **0.0021** | 0.3742–0.3785 | 0.315 | 0.410 | 0.489 | 192.9 | 0.4324 | 0.056 |
| full `--edge_drop 0.2` | 0.3694 | **0.0301** | 0.3351–0.3914 | 0.309 | 0.402 | 0.467 | 162.5 | 0.4035 | 0.034 |
| *sparse ed=0.2 (ref, n=6)* | *0.397* | *0.019* | — | — | — | — | — | — | — |
| *sparse ed=0.0 (ref, n=6)* | *0.340* | *0.027* | — | — | — | — | — | — | — |
| *NBFNet (ref, n=6)* | *0.458* | *0.004* | — | *0.371* | *0.520* | *0.605* | *117* | — | *0.033* |

Contrastes (Welch, todo a 50 épocas):
- **NBFNet vs full ed=0.0: +0.0816 ± 0.0020 (t=39.9)** ← el contraste mejor establecido del proyecto.
- **sparse ed=0.2 vs full ed=0.0: +0.0206 ± 0.0079 (t=2.62)** ← el orden histórico se INVIERTE.
- **edge_drop 0.2 vs 0.0 en el full: −0.0070 ± 0.0174 (t=−0.40) ⇒ NULO.**

**Análisis (causa mecánica)**

- **(A) El full attention NO estaba subentrenado a 20 épocas — el histórico era correcto.**
  0.375 (n=1, 20 ép) → **0.3764 ± 0.0021** (n=3, 50 ép): Δ **+0.001**. La advertencia de la
  cabecera de este archivo ("las filas de atención están subestimadas por entrenar solo 20
  épocas") **vale para el sparse (+0.093) y es FALSA para el full**. El subentrenamiento no
  era una propiedad de los modelos de atención: era del sparse. Corolario metodológico: la
  advertencia de validez hay que aplicarla fila por fila, no por familia.
- **(B) El edge dropout es NULO en el full (−0.007 ± 0.017), contra +0.057 ± 0.013 en el
  sparse.** No es un regularizador genérico de atención: funciona donde las aristas **definen
  el soporte** de la atención (sparse) y no donde la atención es all-pairs y las aristas
  entran solo por el bias relacional y la corrección de valor (full). Esto es consistente con
  el diagnóstico acumulado: el que se sobreajusta a la topología del train graph es el sparse
  (gap val→test 0.108) y no el full (0.056).
  ⚠️ **Error de lectura registrado**: con solo la semilla 42 del brazo ed=0.2 disponible
  (0.3351) se reportó en curso que "el edge dropout DAÑA al full (−0.041)" y se construyó una
  explicación mecánica sobre eso. **Las semillas 43 y 44 dieron 0.3818 y 0.3914** y el efecto
  colapsó a −0.007 ± 0.017. Era n=1 — exactamente lo que el protocolo del 2026-08-05 prohíbe.
  Se deja anotado porque es el mismo modo de fallo que originó ese protocolo.
- **(C) CORRECCIÓN al hallazgo (C) del 2026-08-05: la inestabilidad NO es de la familia de
  atención, es del SPARSE.** Esa entrada concluyó que "el ruido es una propiedad de la familia
  de ATENCIÓN, no del harness" y lo elevó a firma mecánica de la agregación aprendida ("no
  solo transfiere peor en media: transfiere de forma inestable"). Pero el full attention
  **también** es agregación aprendida y mide **σ = 0.0021** sin regularizar — comparable a
  NBFNet (0.004) y **13× más estable que el sparse sin regularizar (0.027)**, a igual
  protocolo, datos y número de épocas. La firma es del **modelo sparse en particular**, no de
  la atención ni de la agregación aprendida. Reformulación correcta: *la atención por
  adyacencia transfiere de forma inestable; la atención densa no*.
  Matiz nuevo: **el edge dropout le infla la varianza al full 14× (0.0021 → 0.0301) sin mover
  la media.** Es esperable de meter estocasticidad en el train, y explica el outlier de la
  semilla 42, cuyo best-valid quedó clavado en la **época 9 de 50** contra 44-49 en las otras
  cinco corridas.
- **(D) El orden histórico se invierte, y ahora se sabe por qué.** CLAUDE.md dejaba abierto
  si "full (0.375) > sparse (0.338)" era artefacto de subentrenamiento. Lo era, **pero del
  lado del sparse**: con ambos a 50 épocas y su mejor regularización, **sparse+edrop 0.397 ±
  0.019 > full 0.3764 ± 0.0021** (t=2.62). El full no se movió; el sparse subió 0.059.
- **(E) La pregunta central de GOALS.md queda respondida CON barra de error.** El full
  attention —el techo de expresividad de cualquier atención sparse— mide **0.3764 ± 0.0021**
  contra **0.458 ± 0.004** de NBFNet: brecha **+0.0816 ± 0.0020 (t=39.9)**. El "NO" que
  GOALS.md ya registraba era correcto; lo que faltaba era la medición que lo sostuviera.
  Nótese que la brecha del full (0.082) es **mayor** que la del mejor sparse (0.061): el techo
  de expresividad rinde peor que su propia restricción, que sigue siendo el resultado más
  contraintuitivo del proyecto.

**Decisión**

- **Prioridad #1 de la cola CERRADA.** Queda como #1 el head-to-head de los 4 modelos en
  v1/v2/WN18RR con el protocolo nuevo, y después los ablations de `--attn`.
- **`--edge_drop` NO se adopta para el full** (nulo). Sigue siendo parte de la best config del
  **sparse** únicamente.
- Config del full confirmada: `--num_layer 6 --hidden_dim 64 --drop 0.0 --learning_rate 1e-3
  --max_epochs 50`, sin edge dropout.
- **Dos afirmaciones de la bitácora se corrigen** (ver A y C): el subentrenamiento a 20 épocas
  y la inestabilidad son del **sparse**, no de la familia de atención. Ninguna conclusión
  cualitativa del proyecto cambia; sí cambia a qué se le atribuye el mecanismo.
- Artefactos: `sbatch_full_v1.sh`, `experiments/full_v1_p{00,02}_ep50_s{42,43,44}/`,
  `logs/full_v1_9138{1,2,3,4,7,8}.log`. Benchmark de tiempos en `logs/_bench/`.

**Nota de infraestructura (medido hoy)**: el hardware real es **A100-SXM4-40GB**, no la
"H100 NVL 95GB" que decía CLAUDE.md. Full attention en ind v1: 88 s/época y **11.8 GB** de
pico en 1 GPU; en **WN18RR v1 son 313 s/época y 33.7 GB de 40** (84 % de la tarjeta) ⇒ ahí el
margen es de 6 GB y cualquier aumento de `hidden_dim`, `--use_rpb` o split más grande va a
OOM. DDP a 2 GPUs escala 2.18× en FB237 y 1.92× en WN18RR (superlineal en FB porque el batch
por GPU baja a 8 y alivia la presión del O(N²)). El QOS `external` topa en **4 jobs
simultáneos** y 8 GPUs por usuario.

---

## 2026-08-07 — REGISTRO de 3 corridas TRANSDUCTIVAS huérfanas (lanzadas 2026-07-22, terminadas y nunca anotadas): `degree` es NULO también en transductivo, en los dos datasets + primer baseline sparse de WN18RR transductivo

**Contexto**: auditoría de checkpoints en disco a raíz de una pregunta sobre qué hay guardado.
Las 3 corridas de la entrada del 2026-07-22 ("RESULTADOS PENDIENTES") **terminaron sus 20
épocas y corrieron test**, pero el resultado nunca se registró: quedaron sólo en los logs.
Se recuperan aquí. Los `logs/nbfnet_trans_fb237_manual.log` y `logs/nbfnet_trans_test.log`
que menciona la entrada del 2026-07-20 **ya no están en `logs/`**; del NBFNet transductivo
sobrevive únicamente el checkpoint parcial.

### Resultado (full-filtered, seed 42, L4/dim32, lr 1e-3, drop 0.0, 20 ep, n=1)

**WN18RR TRANSDUCTIVO** — no existía baseline previo en la bitácora; esta es la primera
medición del sparse en este régimen. Par softmax-vs-degree a **mismo batch (32)**, misma
config, 2714 steps/época en ambos ⇒ comparación limpia.

| variante | valid_mrr | **test_mrr** | H@1 | H@3 | H@10 | MR | best ep |
|----------|----------:|-------------:|----:|----:|-----:|---:|--------:|
| sparse **softmax** | 0.5481 | **0.5536** | 0.508 | 0.576 | 0.644 | 1616 | 15 |
| sparse **degree**  | 0.5468 | **0.5533** | 0.507 | 0.577 | 0.642 | 1610 | 12 |

**Δ test = −0.0002**; las 4 métricas de ranking coinciden en el tercer decimal.

**FB15k-237 TRANSDUCTIVO** — `--attn degree`, 2 GPU DDP, batch **global 32**.

| variante | valid_mrr | **test_mrr** | H@1 | H@3 | H@10 | MR |
|----------|----------:|-------------:|----:|----:|-----:|---:|
| sparse **degree** (batch 32) | 0.4023 | **0.3957** | 0.297 | 0.436 | 0.592 | 126.1 |
| *sparse softmax (ref, batch 8)* | *0.402* | *0.3965* | *0.300* | *0.437* | *0.591* | *128.4* |
| *sparse_exp deg3 (ref, batch 8)* | *0.4012* | *0.3973* | *0.301* | *0.436* | *0.589* | *126.8* |

**CORRECCIÓN de la nota de comparabilidad del 2026-07-22**: esa entrada advierte que la
corrida `degree` usa batch global 32 contra un baseline de batch global 16 ⇒ "régimen 2×".
Los steps de los checkpoints dicen que el desajuste es **4×, no 2×**: `degree` hace 8504
steps/época (170080 / 20 ep) y el baseline `_4_32` hace 34015 (680300 / 20 ep), o sea el
baseline corrió a **batch global 8** — consistente con la corrección ya registrada el
2026-08-03 sobre esas mismas filas. El producto steps × batch coincide en ambos (≈272 K
triples/época), que es el control que cierra la cuenta.

**Análisis (causa mecánica)**

- **`degree` no hace nada en transductivo, y esta vez el dato es limpio en WN18RR.** El par
  de WN18RR es apples-to-apples de verdad (mismo batch, misma config, mismos steps) y da
  **−0.0002**. Es la misma firma de empate a 3-4 decimales con la que se cerró el expander
  transductivo el 2026-08-03 (Δ +0.0008, 6 métricas coincidentes): no es "ayuda poco", es el
  mismo modelo. En FB237 el Δ (−0.0008) apunta igual, pero **ese contraste está contaminado
  por el 4× de batch** y no debe citarse solo.
- **Coherente con el régimen inductivo, donde `degree` sí parecía la mejor variante**
  (+0.028 sobre softmax en ind v1). Esa ventaja era n=1 a 20 épocas con σ≈0.02–0.05 ⇒ dentro
  del ruido según la advertencia de validez del 2026-08-05, que ya la listaba explícitamente
  entre los deltas no distinguibles de cero. Que en transductivo —donde el problema de
  transferencia inductiva no aplica— el efecto sea exactamente 0.000 **es consistente con que
  el +0.028 inductivo fuera ruido**, no con que el scaler de grado aporte algo.
- **Validez**: n=1, 20 épocas, sin barra de error. Cae bajo la advertencia general de la
  cabecera. Lo que sostiene la lectura no es la magnitud del Δ sino el **patrón de empate
  métrica por métrica** en un par por lo demás idéntico.

**Estado del NBFNet transductivo (lo que falta para cerrar la comparación)**

- **FB15k-237**: `experiments/nbfnet_trans_fb237/` existe pero es la corrida parcial del
  2026-07-20 — **epoch 1 de 20**, best valid_mrr 0.376, **sin test**. Reanudable
  (`--resume_from ./experiments/nbfnet_trans_fb237/last.ckpt`; `sbatch_nbfnet_trans.sh` ya
  está encadenado y corre `trainer.test` al llegar a la época 20).
- **WN18RR**: **no existe**, nunca se lanzó.
- ⇒ **no hay baseline NBFNet propio en NINGÚN dataset transductivo**. Las comparaciones
  transductivas de la bitácora se apoyan en el ~0.415 de literatura para FB15k-237, y en
  WN18RR no hay contra qué comparar el 0.5536 recién registrado.

**Decisión**

- Los 3 resultados quedan registrados; la entrada del 2026-07-22 se marca como resuelta.
- **`degree` cerrado también en transductivo**: no se gasta más GPU en esa variante. Se suma
  al bloque de ablations de `--attn` que la cola de re-medición del 2026-08-05 pone en último
  lugar ("varios probablemente desaparecen dentro del ruido") — éste ya desapareció.
- Pendiente de GPU para cerrar transductivo: reanudar NBFNet FB15k-237 (~2.5 h/época × 18) y
  lanzar NBFNet WN18RR desde cero (~14 min/época, barato). **No cambia la prioridad #1**, que
  sigue siendo el full attention inductivo re-medido con el protocolo nuevo.
- Artefactos: `experiments/sparse_wn_trans/`, `experiments/sparse_degree_wn_trans/`,
  `experiments/sparse_degree_trans_fb237/`, `logs/sparse_wn_trans_819322.log`,
  `logs/sparse_degree_wn_trans.log`, `logs/degree_trans_fb237_819321.log`.

---

## 2026-08-05 (b) — TIPADO de las aristas del expander, 3 variantes (`--exp_typing`), REPLICADO en FB15k-237 v1 + WN18RR v1: el expander era un REGULARIZADOR, no un atajo; y cuanto más "válida" es la relación falsa, más daña

**Contexto**: para la presentación de tesis se necesitan las 3 variantes de tipado de las
aristas del expander (existían en un proyecto anterior, se perdieron). El expander conecta
pares que no existían en el grafo; en KGC toda arista debería llevar una relación válida.
Variantes: `single` (una relación genérica nueva R_exp compartida — la que ya estaba
implementada y la que dio el 0.3973 transductivo), `ultra` (la arista **toma prestada** una
relación aprendida transferible vía ULTRA) y `path` (la relación se **compone** de los caminos
reales entre los dos nodos).

**Implementación (nueva)**: `build_relation_graph()` es port de `ULTRA/ultra/tasks.py:144`
(grafo de relaciones con las 4 interacciones fundamentales hh/tt/ht/th vía productos de
matrices de incidencia); `RelationGNN` corre message passing sobre él con boundary en el
nodo-relación `r_q` ⇒ representaciones **relativas a la query**, y la arista expander toma
prestada una mezcla softmax aprendida de ellas (la mezcla es **por query, no por arista**:
una arista aleatoria no tiene contexto propio del que derivar una relación distinta).
`compute_expander_paths()` reconstruye el camino más corto real (scipy, ≤ `--exp_path_len`)
y la capa compone: bias = Σ b[r_k], valor = Π g[r_k] (DistMult); sin camino ⇒ fallback R_exp.
Flags `--exp_typing {single,ultra,path} --exp_path_len`. Conteo de params como control:
`single` y `path` = 384 K idénticos (path no agrega parámetros, solo compone sobre las tablas
existentes), `ultra` = 401 K (+17 K del RelationGNN y la cabeza de préstamo). **En la
inicialización `path` ≡ `single` exactamente** (tablas en 0 y 1 ⇒ suma y producto neutros):
las 3 variantes parten del mismo punto y solo divergen al entrenar.

**Régimen**: FB15k-237 **inductivo v1**, 50 épocas, n=3 semillas (protocolo vigente). NO
transductivo: allá cuesta ~3:52 h/época (job 91099) ⇒ 4 brazos × 3 semillas × 50 ep sería del
orden de meses. **El 0.3973 transductivo de `single` NO es comparable con esta tabla.**

### Resultado (50 ep, n=3 semillas 42/43/44, full-filtered) — 42 corridas

**FB15k-237 ind v1** (composicional; el sparse está lejos de NBFNet 0.458)

| variante | edge_drop 0.0 | edge_drop 0.2 | MR (ed=0.2) |
|----------|--------------:|--------------:|------------:|
| sparse (sin expander) | 0.306 ± 0.023 | 0.398 ± 0.008 | 176.5 |
| expander `single` (R_exp) | **0.377 ± 0.013** | 0.409 ± 0.002 | 175.6 |
| expander `ultra` (prestada) | 0.309 ± 0.028 | **0.347 ± 0.009** | 163.3 |
| expander `path` (composicional) | 0.349 ± 0.033 | **0.412 ± 0.006** | 170.5 |

**WN18RR ind v1** (local/jerárquico; el sparse ya está en su techo, NBFNet 0.740)

| variante | edge_drop 0.0 | edge_drop 0.2 | MR (ed=0.2) |
|----------|--------------:|--------------:|------------:|
| sparse (sin expander) | 0.734 ± 0.007 | 0.734 ± 0.005 | 35.6 |
| expander `single` (R_exp) | 0.732 ± 0.007 | 0.728 ± 0.009 | 36.4 |
| expander `ultra` (prestada) | 0.720 ± 0.010 | **0.704 ± 0.019** | **79.4** |

Contrastes vs sin expander (Welch):

| | FB237 ed=0.0 | FB237 ed=0.2 | WN18RR ed=0.0 | WN18RR ed=0.2 |
|---|---:|---:|---:|---:|
| `single` | **+0.071** (t=4.73) | +0.011 (t=2.43) | −0.002 (t=−0.39) | −0.006 (t=−1.04) |
| `ultra` | +0.003 (t=0.15) | **−0.051** (t=−7.49) | −0.014 (t=−2.06) | **−0.030** (t=−2.68) |
| `path` | +0.043 (t=1.84) | +0.014 (t=2.60) | — | — |

**Efecto del edge dropout SOLO (sin expander): FB237 +0.091 ± 0.014 (t=6.59);
WN18RR −0.000 ± 0.005 (t=0.00).**

*(Corrección: `path` a ed=0.0 se reportó en curso como 0.368 ± 0.011 con n=2; con la 3ª
semilla baja a 0.349 ± 0.033. El n=2 era optimista.)*

### Cobertura de caminos (medida sobre FB15k-237 v1, exp_degree=3, ≤3 saltos)

| grafo | N | aristas expander | con camino ≤3 | fallback R_exp |
|-------|--:|-----------------:|--------------:|---------------:|
| train | 1594 | 9560 | 3736 (**39.1 %**) | 5824 (60.9 %) |
| **test (ind)** | 1093 | 6554 | 582 (**8.9 %**) | **5972 (91.1 %)** |

Largo medio del camino tipado: 2.65 saltos.

**Análisis (causa mecánica)**

- **(A) El expander funcionaba como REGULARIZADOR, no como atajo estructural — y la
  REPLICACIÓN CRUZADA lo demuestra.** En FB237, sin regularizar `single` gana +0.071 y
  *parece* que ayuda; con edge dropout la ganancia se derrumba a +0.011. El contraste que lo
  cierra: quitar el expander y poner edge dropout vale **+0.091**, más que agregar el expander
  sin edge dropout (+0.071) — hacen el mismo trabajo, y el edge dropout lo hace mejor y ~1.67×
  más barato (aristas aleatorias = accesos no coalescidos, ver 2026-07-20).
  **La prueba decisiva está en WN18RR**: ahí el sparse ya está en su techo (0.734 vs NBFNet
  0.740), el edge dropout vale **exactamente 0.000 (t=0.00)** ⇒ no hay sobreajuste estructural
  que amortiguar… **y el expander también deja de ayudar (−0.002 / −0.006)**. El beneficio del
  expander aparece y desaparece EXACTAMENTE donde aparece y desaparece el beneficio de
  regularizar. Si diera atajos estructurales genuinos debería ayudar *más* en WN18RR, que es
  el grafo con diámetro grande y estructura jerárquica donde el largo alcance importaría: da
  cero. Esto explica además el nulo transductivo (0.3973 vs 0.3965) sin contradecirlo.
- **(B) El tipado ULTRA es el ÚNICO efecto con signo consistente en los DOS datasets, y es
  DAÑINO. El código lo había predicho.** FB237 −0.051 (t=−7.49) y WN18RR −0.030 (t=−2.68),
  ambos con edge dropout; y en los dos el daño **crece** al regularizar (FB237 +0.003→−0.051;
  WN18RR −0.014→−0.030), o sea no es ruido que la regularización limpie sino señal falsa que
  compite con la verdadera. El comentario de diseño de `sparse_exp` (2026-07-20) decía que las
  aristas expander no deben llevar una relación KG real porque "sería inyectar hechos falsos".
  ULTRA hace exactamente eso: la relación prestada es una mezcla de relaciones **reales**, así
  que la arista aleatoria **se presenta como un hecho relacional entre un par que no tiene
  ninguna relación**, y encima condicionada a la query — una mentira más convincente.
  **Firma adicional en WN18RR: el MR se dispara a 79.4 contra 35.6 del baseline (más del
  doble)**, mientras MRR/H@1 caen poco. O sea el daño no es ruido difuso: los hechos falsos
  empujan candidatos equivocados muy arriba en el ranking. En un grafo jerárquico con pocas
  relaciones muy estructuradas, inventar una relación plausible es más destructivo que en uno
  composicional y denso.
- **(C) `path` es casi indistinguible de `single` POR CONSTRUCCIÓN, y el dato de cobertura es
  el hallazgo más nítido para la tesis.** Con edge dropout: `path` 0.412 ± 0.006 vs `single`
  0.409 ± 0.002 — estadísticamente iguales. La razón está en la cobertura: en el grafo de test
  inductivo solo el **8.9 %** de las aristas expander tiene camino ≤3 saltos, o sea **el
  91.1 % cae al fallback R_exp** ⇒ *en test la variante 3 ES la variante 1 para nueve de cada
  diez aristas*. El empate es entonces casi tautológico y **no debe leerse como "componer
  relaciones no sirve"**. Lo que dice el número es más fuerte: **los pares que se pueden tipar
  composicionalmente son justo los que el message passing ya alcanza, y los que aportarían
  conectividad nueva —fuera del horizonte— son exactamente los que NO se pueden tipar.**
  Tipado y aleatoriedad del expander están en tensión estructural, no empírica. (Por eso
  `path` no se corrió en WN18RR: con ~2 triples por entidad la cobertura sería aún menor que
  el 8.9 % y la variante colapsaría a `single` casi por completo.)
- **(D) Matiz sobre la lista negra #2** ("NO expander en inductivo — metían ruido"): aquí el
  ruido **es** el beneficio (regularización), lo cual refina el ítem en vez de contradecirlo.
  Con la regularización correcta ya en el modelo, el expander aporta +0.011 (FB237, t=2.43,
  marginal) y **−0.006 en WN18RR** a cambio de 1.67× de cómputo. No lo rehabilita como
  dirección: lo reduce a un regularizador caro y dependiente del dataset.
- **(E) Nota de implementación**: `path` corre a 3.33 it/s vs 16.38 del resto (5× más lento),
  por el gather `rel_value[:, paths, :]` de forma (H, Ex, K, hd) + `where` + `prod`, por capa
  y por batch. Es overhead de implementación, no del método; el término de bias se podría
  reescribir como matmul contra una matriz de conteos relación×arista precomputada.

**Decisión**

- Las 3 variantes quedan implementadas y medidas; 42 corridas, todas cerradas. Para la
  presentación el arco con evidencia es: **etiqueta genérica (mejor) → composicional real
  (empata, y en test es la genérica el 91 % del tiempo) → relación real prestada (daña, en los
  dos datasets)**. La conclusión no es "el tipado no importa" sino que **importa en la
  dirección contraria a la intuición**: mientras más se parece la arista falsa a un hecho
  verdadero, más daña.
- **Qué replica y qué no** (importante, este proyecto ya se quemó dos veces con efectos de un
  solo split — LapPE y sparse+RWSE): **replica** el daño de ULTRA (signo consistente en ambos
  datasets, y creciente con regularización). **NO replica** el beneficio del expander: +0.071
  en FB237 sin regularizar, 0.000 en WN18RR — y su desaparición coincide exactamente con la
  desaparición del beneficio de regularizar, que es justo lo que predice la hipótesis (A).
- **`path` no se corrió en WN18RR** (decisión del usuario por costo: 5× más lento). Dado el
  argumento de cobertura de (C), el resultado esperado es colapso a `single`; queda sin medir.
- Artefactos: `sbatch_exp_typing.sh`, `experiments/typing_*`,
  `logs/exp_typing_9117{9,80,81,82}.log` (FB237) y `logs/exp_typing_9118{3,4,5}.log` (WN18RR),
  `src/model.py::{build_relation_graph,RelationGNN,compute_expander_paths}`.

---

## 2026-08-05 — EDGE DROPOUT (`--edge_drop`) + primer estudio de SEMILLAS del proyecto: la brecha contra NBFNet SOBREVIVE pero mide la MITAD de lo registrado (0.061, no 0.121); el protocolo de medición de toda la bitácora (n=1, 20 épocas) es más ruidoso que los efectos que reporta

**Contexto**: la conclusión registrada en CLAUDE.md —"el gap del sparse contra NBFNet es
**estructural**, no un artefacto de entrenamiento"— se sacó de un sweep que varió únicamente
**dim, L y lr**. Pero el modo de fallo diagnosticado en cuatro entradas previas es
*sobre-ajuste a la topología del train graph* (firma valid↑/test↓, gap 0.10–0.16 vs 0.033 de
NBFNet), y **la regularización que ataca ese modo nunca se probó**: `--drop` es dropout de
features y de pesos de atención (`model.py:319,333,716`), y `graph_mask` solo quita la arista
de la query (anti-fuga, no augmentación). Nada perturbaba la estructura del grafo en train.
Encima la best config de los 3 modelos de atención es `drop 0.0`: la regularización se había
tuneado **hacia abajo** mientras el diagnóstico decía sobreajuste.

**Implementación**: `apply_edge_dropout` en `src/model.py` + flag `--edge_drop` en `train.py`.
Elimina al azar una fracción p de aristas en **cada forward de train**; en eval el grafo va
completo. Se aplica DESPUÉS de `graph_mask` (que indexa sobre el edge_index completo) y ANTES
de concatenar self-loops, así ningún nodo queda con vecindario vacío y el segment-softmax no
ve conjunto vacío. Cableado en los 5 modelos; default 0.0 ⇒ **comportamiento previo intacto**.
Smoke test CPU: proporción correcta, no-op exacto con p=0 y en eval, forward/backward finito
en los 5 modelos, train estocástico / eval determinista, sin NaN con p=0.99.

### Resultado 1 — sweep de p (20 ep, seed 42, FB15k-237 ind v1, best config sparse)

| edge_drop | valid_mrr | test_mrr | **gap val→test** | H@1 | H@10 | MR |
|-----------|----------:|---------:|-----------------:|----:|-----:|---:|
| 0.0 *(bitácora, H100)* | 0.446 | 0.338 | 0.108 | — | 0.451 | — |
| **0.0 (control local)** | 0.455 | **0.202** | **0.253** | 0.166 | 0.212 | 217 |
| 0.1 | 0.453 | 0.390 | 0.063 | 0.324 | 0.493 | 167 |
| 0.2 | 0.459 | **0.403** | 0.056 | 0.339 | 0.500 | 181 |
| 0.3 | 0.444 | 0.400 | 0.043 | 0.346 | 0.495 | 174 |
| 0.5 | 0.422 | 0.386 | 0.036 | 0.315 | 0.495 | 191 |
| *NBFNet (ref, n=1)* | *0.492* | *0.459* | *0.033* | *0.371* | *0.605* | *117* |

El **control p=0.0 no reprodujo el baseline** (0.202 vs 0.338 registrado) pese a que `valid`
sí reproduce (0.455 vs 0.446). Verificado por `git diff d76f49e..HEAD -- src/model.py` que el
path `softmax` es **idéntico** al que produjo el 0.338: el branch `else` reproduce el código
previo línea por línea y el orden de creación de parámetros no cambió (misma init con la misma
semilla). **El código no explica el salto** ⇒ se lanzó el estudio de semillas.

### Resultado 2 — estudio de SEMILLAS (20 ep) — el hallazgo central

| config | n | media | **sd** | rango |
|--------|--:|------:|-------:|------:|
| p=0.0  | 4 | 0.247 | **0.054** | 0.202 – 0.317 |
| p=0.1  | 3 | 0.333 | **0.098** | 0.220 – 0.390 |
| p=0.2  | 4 | 0.399 | **0.007** | 0.391 – 0.405 |

**La semilla 42 corrida DOS veces con p=0.0 dio 0.202 y 0.317** — mismo código, misma semilla,
mismo cluster. La varianza no es de semilla sino de **no-determinismo de CUDA** (`index_add_`,
que CLAUDE.md declara "aceptable para baseline"; **no lo es**). Ninguna de las 4 corridas de
p=0.0 alcanzó el 0.338 de la bitácora: la media es 0.247 y 0.338 queda **fuera del rango
observado**, o sea el número registrado fue una extracción afortunada.

### Resultado 3 — TODO a 50 épocas, n=6 semillas (42-47), apples-to-apples

| config | n | media | **sd** | rango |
|--------|--:|------:|-------:|------:|
| **NBFNet (50 ep)** | 6 | **0.458** | **0.004** | 0.452 – 0.462 |
| sparse p=0.0 (50 ep) | 6 | 0.340 | 0.027 | 0.294 – 0.367 |
| **sparse p=0.2 (50 ep)** | 6 | **0.397** | 0.019 | 0.367 – 0.422 |

Contrastes (Welch, n=6 por brazo, **todo a 50 épocas**):
**NBFNet vs sparse p=0.2 = +0.061 ± 0.008 (t=7.49)**; NBFNet vs sparse p=0.0
+0.118 ± 0.011 (t=10.64); **p=0.2 vs p=0.0 = +0.057 ± 0.013 (t=4.23)**.

Referencia intermedia (n=3, 20 ep, descartada por subentrenamiento): NBFNet 0.437 ± 0.027;
sparse p=0.0 0.247 ± 0.054; p=0.2 0.399 ± 0.007; p=0.3 0.397 ± 0.016.

**Análisis (causa mecánica)**

- **(A) 20 épocas SUBENTRENA a los modelos de atención, y eso pesa tanto como el edge
  dropout.** El mismo sparse sin regularizar pasa de 0.247 (20 ep, n=4) a **0.340** (50 ep,
  n=6): **+0.093** sin tocar la arquitectura. Todas las corridas de atención de la bitácora
  son a 20 épocas. Esto solo ya invalida la premisa "no cierra con entrenamiento razonable".
  NBFNet también estaba subentrenado (0.437 → 0.458) pero mucho menos: **el subentrenamiento
  castiga desproporcionadamente al modelo de atención**, lo que infla artificialmente todos
  los gaps registrados.
- **(B) El edge dropout es un efecto real y sobrevive a las barras**: **+0.057 ± 0.013
  (t=4.23, n=6)** sobre p=0.0 a igual número de épocas. Y el **mecanismo predicho se
  confirma**: el gap valid→test
  cae monótonamente con p (0.253 → 0.063 → 0.056 → 0.043 → 0.036), llegando al de NBFNet
  (0.033). Perturbar la estructura en train impide anclar la agregación aprendida a la
  topología del train graph — que era exactamente la causa diagnosticada por eliminación en
  las entradas del 2026-07-22 y 2026-07-24. **La causa estaba bien identificada; lo que
  faltaba era la intervención correcta, que no era arquitectónica sino de regularización.**
- **(C) El ruido es una propiedad de la familia de ATENCIÓN, no del harness.** A 50 épocas
  NBFNet tiene **σ = 0.004** (n=6) contra σ = 0.019–0.027 del sparse: **5-7× más estable**.
  El mismo no-determinismo de `index_add_` afecta a los dos, pero solo el sparse lo amplifica
  a ±0.05. Eso es diagnóstico en sí mismo: la agregación aprendida no solo transfiere peor en
  media, **transfiere de forma inestable** — dos corridas idénticas aterrizan en soluciones
  con calidad muy distinta sobre el grafo disjunto. La agregación FIJA de NBFNet no tiene esa
  patología. Es una firma adicional de la misma causa (agregación aprendida) y **no estaba
  documentada**.
- **(D) La brecha titular SOBREVIVE y queda establecida con significancia fuerte:
  +0.061 ± 0.008 (t=7.49, n=6, todo a 50 épocas).** El 0.459 registrado de NBFNet **no era
  una semilla afortunada**: a 50 épocas su media es 0.458 ± 0.004, prácticamente idéntica.
  Lo que cambia es la MAGNITUD: la brecha real contra el mejor sparse es **0.061, no los
  0.121 registrados** — la mitad del gap que la bitácora atribuía a la arquitectura era en
  realidad subentrenamiento (+0.10 de 20→50 ep) y falta de regularización (+0.057 de edge
  dropout). La conclusión cualitativa del proyecto se mantiene; su cuantificación no.
- **(E) CORRECCIÓN de una lectura intermedia de esta misma sesión.** Con NBFNet a 20 épocas
  contra el sparse a 50, el contraste daba +0.036 ± 0.019 (t=1.87) y se leyó como "el gap no
  está establecido". **Eso era un artefacto del desajuste de épocas**: NBFNet también estaba
  subentrenado a 20 ep (0.437 ± 0.027 → 0.458 ± 0.004 a 50). Corregido con las corridas
  91175-91178. Se deja registrado porque la lección es del tipo que causó el problema
  original: comparar brazos con protocolos distintos produce conclusiones invertidas.
- **(F) Alcance de la advertencia al resto de la bitácora.** Con σ ≈ 0.02–0.05 y n=1, los
  ablations finos no son distinguibles de cero: sigmoid +0.018, degree +0.028, rel +0.020,
  RWSE −0.009/+0.022/−0.003, LapPE −0.003/+0.035/−0.005/+0.013. Varias entradas construyen
  interpretación mecánica sobre esos deltas (p.ej. "degree produce el valid más alto pero el
  peor gap", "rel ensanchó levemente la brecha 0.113 > 0.108"): **esas lecturas no están
  soportadas**. Los efectos grandes (anchor −0.060, source_rw −0.062/−0.128, full vs NBFNet)
  probablemente sobreviven, pero ninguno tiene barra de error. Nada de esto se corrige
  reinterpretando: hay que re-medir.

**Decisión**

- **La conclusión central del proyecto se mantiene**: ninguna variante de atención supera a
  NBFNet en FB15k-237 ind v1, y ahora con significancia fuerte (t=7.49, n=6) en vez de con
  una corrida por brazo. **La lista negra de CLAUDE.md NO se toca.** Lo que sí hay que
  corregir en CLAUDE.md es la **magnitud** del gap y la frase "no cierra con tuning
  razonable": sí cierra parcialmente —la mitad— con entrenamiento suficiente y la
  regularización correcta.
- **Protocolo nuevo obligatorio de aquí en adelante**: n≥3 semillas (n≥6 para contrastes
  finos) + media ± sd, y **≥50 épocas** para modelos de atención, verificando que el
  best-valid no caiga en las últimas épocas. Reportar contrastes con error estándar, no
  deltas puntuales. Los modelos de atención necesitan más n que NBFNet (σ 5-7× mayor).
- **`--edge_drop 0.2` pasa a ser parte de la best config del sparse** (+0.057 ± 0.013,
  t=4.23). Config recomendada: `--hidden_dim 64 --drop 0.0 --edge_drop 0.2
  --learning_rate 1e-3 --max_epochs 50`.
- **Cola de re-medición, por prioridad**: (1) **full attention** 50 ep n≥3 con y sin
  edge_drop — es el modelo de la pregunta central de GOALS y nunca se probó regularizado ni
  entrenado lo suficiente; (2) head-to-head de los 4 modelos en v1/v2/WN18RR con el protocolo
  nuevo; (3) recién después, los ablations de `--attn` si siguen valiendo la pena — varios
  probablemente desaparecen dentro del ruido.
- **Nota metodológica**: considerar fijar el no-determinismo (`torch.use_deterministic_algorithms`
  con un scatter determinista) o, más barato, aceptar que el harness requiere n≥3 siempre.
  La decisión registrada en CLAUDE.md de tolerar `index_add_` no determinista "aceptable para
  baseline" es la causa raíz de que un baseline valga 0.202 o 0.317 según la corrida.
- Artefactos: `sbatch_edge_drop_v1.sh`, `sbatch_edge_drop_seeds.sh`, `sbatch_edrop_final.sh`,
  `experiments/edrop_*`, `experiments/final_*`, `logs/edrop_*`.

---

## 2026-08-03 — RESULTADO de `sparse_exp` en FB15k-237 TRANSDUCTIVO: los expander no aportan NADA (Δ +0.0008) — cierra la línea también en transductivo

**Contexto**: terminó la corrida encadenada de `--model sparse_exp --exp_degree 3` lanzada el
2026-07-20 y continuada en el cluster patagon (A100). Job final 91099, log
`logs/sparse_exp_trans_fb237_91099.log`, ckpt `experiments/sparse_exp_trans_fb237/`. Completó
las 20 épocas; test sobre el mejor ckpt (epoch 18, valid_mrr 0.40123). Config L4, dim 32,
batch 8, lr 1e-3, drop 0.0, seed 42. Pico 16.9 GB, ~3:52 h/época en A100.

**Resultado (FB15k-237 transductivo, full-filtered)**

| config | valid_mrr | **test_mrr** | H@1 | H@3 | H@10 | MR |
|--------|----------:|-------------:|----:|----:|-----:|---:|
| **sparse_exp deg3** (L4, d32) | 0.4012 | **0.3973** | 0.301 | 0.436 | 0.589 | 126.8 |
| sparse simple (L4, d32) | 0.402 | 0.3965 | 0.300 | 0.437 | 0.591 | 128.4 |
| sparse simple (L6, d64) | 0.407 | 0.4028 | 0.306 | 0.443 | 0.595 | 116.4 |

**Análisis (causa mecánica)**

- **Δ test = +0.0008 contra el sparse simple de config idéntica: las 6 métricas coinciden una
  por una** (H@1 0.301 vs 0.300, H@3 0.436 vs 0.437, H@10 0.589 vs 0.591). No es "ayuda poco":
  es el mismo modelo dentro del ruido, pagando ~1.67× por época.
- **La comparación SÍ es apples-to-apples**: ambas corridas tienen exactamente **38399
  steps/época**, o sea mismo batch (8) y mismo dataset. (Corrección: la tabla del 2026-07-20
  dice "batch 16" para esas filas transductivas; los steps/época dicen que fue 8.)
- **Por qué no puede funcionar**: en KGC toda arista **es evidencia**; una arista expander
  aleatoria fabrica caminos sin soporte relacional. Distinto del caso donde los expander
  funcionan (grafos moleculares, tareas a nivel de grafo), donde la arista extra solo *mueve*
  información hacia un pooling global y no *afirma* nada. Además **homogeneizan la distancia
  al head**, que es exactamente lo que hace funcionar al labeling trick — el mismo fallo ya
  medido en CPA (cos 0.992, el ancla se lava tras 3 capas) y en source_rw.
- La lista negra #2 prohibía expander en INDUCTIVO; esto los cierra también en transductivo,
  que era el régimen donde Exphormer-Max iba bien y donde quedaba la duda abierta.

**Decisión**: línea expander cerrada en ambos regímenes ⇒ **lista negra #8** en CLAUDE.md.
No gastar más GPU en variantes (degree, capas selectivas). Artefactos:
`sbatch_sparse_exp_trans_patagon.sh`, `experiments/sparse_exp_trans_fb237/`,
`logs/sparse_exp_trans_fb237_91099.log`.

---

## 2026-07-24 — RESULTADO del ablation `--attn rel`: el overfit del sparse NO vive en `q·k` — REFUTADA la hipótesis de logits dependientes de estado

**Contexto**: ejecución del ablation implementado abajo. Pregunta: ¿el gap de transferencia del sparse
(firma valid↑/test↓) vive en el término `⟨q_dst, k_src⟩`, la única parte del logit que lee estados
ocultos de nodo? `--attn rel` lo elimina y deja el logit 100% relacional
(`b[head,rel] + ⟨u[head,r_q], w[head,rel]⟩/√d`), transferible por construcción como NBFNet.
FB15k-237 ind v1, best config sparse (dim 64, L6, drop 0.0, lr 1e-3, batch 16, 20 ep, seed 42),
611 K params. Log `logs/sparse_rel_v1.log`, ckpt `experiments/sparse_rel_v1/` (best epoch 13).
Corrido como step `srun --overlap` sobre la GPU del job 820761 (SIGSTOP'd durante ~10 min y reanudado
después; el grafo ind v1 es diminuto: 328 steps/época, 1.3 GB, ~15 s/época ⇒ 20 ep + test en ~10 min,
no los ~40 min estimados).

**Resultado (FB15k-237 ind v1, full-filtered) — todas las variantes sparse, ordenadas por gap valid→test**

| sparse variant  | valid_mrr | **test_mrr** | **gap val→test** | H@1 | H@3 | H@10 | MR |
|-----------------|----------:|-------------:|-----------------:|----:|----:|-----:|---:|
| softmax (base)  | 0.446     | 0.338        | 0.108            | —   | —   | 0.451 | — |
| sigmoid (A)     | 0.415     | 0.356        | **0.059**        | 0.312 | 0.380 | 0.427 | 200 |
| degree (A')     | 0.462     | **0.366**    | 0.096            | 0.300 | 0.410 | 0.473 | 190 |
| anchor (C)      | 0.442     | 0.278        | 0.164            | 0.188 | 0.334 | 0.434 | 181 |
| **rel (D)**     | **0.471** | 0.358        | 0.113            | 0.288 | 0.398 | 0.468 | 181 |
| *NBFNet (ref)*  | *0.492*   | *0.459*      | ***0.033***      | *0.371* | *0.520* | *0.605* | *117* |

**Análisis (causa mecánica)**
- **Sale la rama 2 de la predicción falsable: `rel` se queda en la banda ~0.34–0.37** (0.358), no
  recupera la zona NBFNet (~0.45). **La hipótesis "el overfit vive en los logits dependientes de estado"
  queda REFUTADA.** Es +0.020 sobre softmax (0.338), del mismo orden que sigmoid (+0.018) y por debajo
  de degree (+0.028): tres intervenciones mecánicamente muy distintas que rinden lo mismo ⇒ ninguna
  toca la causa real.
- **El dato decisivo es el gap, y va en contra de la hipótesis.** `rel` produce el **valid_mrr más alto
  de todas las variantes de atención (0.471**, el más cercano a NBFNet 0.492**)** pero su test es 0.358
  ⇒ **gap 0.113, MAYOR que el del softmax base (0.108)**. Quitar TODA dependencia de estado del logit
  **no redujo la brecha de transferencia: la ensanchó levemente.** Y el contraste de fondo: NBFNet
  transfiere con gap **0.033**, un tercio del de cualquier variante de atención. La brecha es una
  propiedad de la familia, no del término `q·k`.
- **Alcance honesto de la conclusión**: `rel` deja los *pesos de atención* libres de estado, pero el
  modelo sigue leyendo estados en el **valor** (`to_v(h)`) y en el readout. Lo refutado es, con
  precisión, que la dependencia de estado **del logit** sea el cuello de botella — no que el modelo
  entero sea transferible. Aun así la lectura es fuerte: el canal que se sospechaba (y que en el
  proyecto viejo la cirugía de pesos señalaba, `K.weight`/`proj_q`) resultó no ser el dominante.
- Lectura conjunta A + C + D: **normalización (A), anclaje a agregación fija (C) y dependencia de
  estado en el logit (D) están los tres refutados como causa dominante.** Lo que queda en pie es la
  **agregación aprendida per se**: que los pesos de mezcla sean *aprendidos y query-dependientes* —
  con lo que sea que se calculen — es lo que no transfiere al grafo disjunto, frente a la agregación
  FIJA (suma + PNA) de NBFNet.

**Decisión**
- Ablation cerrado con diagnóstico completo. La línea "sparse attention como reemplazo del message
  passing en inductivo" queda cerrada con causa identificada por eliminación: no es la normalización,
  no es el ancla, no son los logits de estado ⇒ es la agregación aprendida. Candidato a lista negra.
- El `qc` transductivo (`run_qc_trans.sh`) sigue en pie como pregunta **separada**: su valor es medir
  cuánto del 0.456 viejo era la fuga de labels, en un régimen (transductivo) donde el problema de
  transferencia inductiva no aplica. No hereda la refutación de arriba.
- Artefactos: `run_rel_v1.sh`, `experiments/sparse_rel_v1/`, `logs/sparse_rel_v1.log`.

---

## 2026-07-24 — IMPLEMENTADOS: `--attn rel` (ablation de transferencia) y `--attn qc` (paquete QC-Exphormer) + `--filtered_ce` — `rel` YA CORRIDO (ver entrada de arriba); `qc` pendiente

**Contexto 1 — el ablation `rel`**: tras refutar A (normalización) y C (anclar a mean-pool), el
sospechoso restante del overfit del sparse es el **término `⟨q_dst, k_src⟩` del logit**: es la única
parte que depende de los **estados ocultos de nodo**, cuya distribución cambia en el grafo de test
disjunto. El canal relacional `b[head, rel]`, en cambio, se comparte train/test por construcción.
`--attn rel` **elimina `q·k`** y deja el logit puramente relacional:
`a_e = b[head,rel] + ⟨u[head,r_q], w[head,rel]⟩/√d` (compatibilidad query×relación aprendida). Sigue
siendo atención aprendida y compite entre vecinos, pero con la transferibilidad de NBFNet. La capa en
este modo **ni siquiera crea** `to_q`/`to_k`. Predicción falsable: si `rel` recupera la zona de NBFNet
(~0.45) el overfit estaba localizado en `q·k` (hallazgo publicable: *la atención en KGC inductivo
falla por sus logits dependientes de estado, no por ser atención*); si se queda en ~0.34, la
agregación aprendida per se queda refutada y la línea sparse se cierra con diagnóstico completo.

**Contexto 2 — el paquete `qc`**: del proyecto viejo (Exphormer_Max) se recuperó la arquitectura del
run FB15k-237 transductivo que marcaba **0.456**. Análisis de ese código (ver abajo): el número **NO
es comparable al SOTA** porque su loss de train filtraba con `all_triples_filter` = train+**val+test**
(`losses.py::kgc_full_graph_ce` líneas 39-50, alimentado desde `trainer.py:429`) ⇒ las respuestas del
test nunca eran penalizadas como negativos: **fuga de labels**. La selección de modelo sí era legítima
(por val). Pero el *paquete arquitectónico* es legítimo y coincide punto por punto con los diagnósticos
de este harness: (i) **suma-exp sin normalizar** `exp(clip(·,±5))` sin `/Z`; (ii) **Q anclado a x⁰**
(no lee `h` ⇒ mata medio canal dependiente de estado); (iii) **logit trilineal** `(q⊙k⊙e[rel])·1` con
relación **vectorial** en vez de bias escalar; (iv) **query conditioning** `c_q` sumado a q/k/e (el
sparse actual NO condiciona la atención a `r_q` — solo entra por x⁰); (v) **residual Bellman-Ford**
`x += x⁰` por capa. `--attn qc` porta los cinco al harness limpio.

**Contexto 3 — `--filtered_ce`**: versión SIN fuga del truco del proyecto viejo. Excluye del
denominador del CE las otras respuestas conocidas **solo de train** de la query (h,r) — multi-answer
label handling estándar. Usa el `filter_mask` que `train_collate_fn` ya construye desde
`data.train_filters` (solo triplets de train). Aplica a todos los modelos ⇒ **para comparar hay que
re-correr también el baseline NBFNet con el flag**.

**Implementación (HECHA, sin correr)**: `src/model.py::SparseRelationalAttentionLayer` (branches `rel`
y `qc` en el logit y en la agregación; parámetros nuevos `rel_att_q/rel_att_k` para `rel`,
`rel_e/q_rel_emb/proj_{q,k,e}` para `qc`), threading de `x0`/`r_index` por `SparseGraphTransformer` y
`SparseExpanderGraphTransformer` (+ residual BF cuando `attn=='qc'`), `--attn {…,rel,qc}` y
`--filtered_ce` en `train.py`. Default `softmax` sin `--filtered_ce` ⇒ **comportamiento previo
intacto**. Smoke test CPU: los 6 modos × 2 modelos hacen forward/backward con scores finitos; `rel` no
tiene `to_q`/`to_k` y sus tablas relacionales reciben gradiente; `rel` y `qc` cambian el score al
cambiar `r_q` (query-conditioned de verdad); los 6 modos difieren entre sí con la misma semilla; el
enmascarado de `filtered_ce` nunca toca el gold y baja el CE.

**Experimentos planificados**: `run_rel_v1.sh` (ind v1, diagnóstico, ~40 min), `run_qc_v1.sh` (ind v1),
`run_qc_trans.sh` (FB15k-237 transductivo, L5/dim64/lr 2e-4 alineado al config viejo — es donde el
paquete debería brillar y donde el gap con NBFNet ya era chico: sparse 0.403 vs ~0.415). **Resultados:
PENDIENTES.** Si `qc` limpio llega a 0.44-0.45 transductivo, es un resultado real de atención pura
sobre NBFNet; si cae a ~0.41, el grueso del 0.456 viejo era la fuga.
## 2026-07-22 — Lanzamientos TRANSDUCTIVOS de `degree` (+ primer uso multi-GPU DDP del harness) — ~~RESULTADOS PENDIENTES~~ **RESUELTO: ver entrada 2026-08-07** (las 3 terminaron; `degree` nulo en ambos datasets)

**Contexto**: cerrada la opción `degree` en inductivo (fue la mejor variante de atención en FB15k-237 ind v1:
test 0.366 vs softmax 0.338, ver entrada de abajo), se lleva `--attn degree` al régimen **transductivo** para
ver si el scaler de grado ayuda ahí. Se lanzaron 3 corridas (todas dim32/L4, lr 1e-3, drop 0.0, seed 42, 20 ep).

**Corridas lanzadas** (~~resultado MRR PENDIENTE~~ — las 3 terminaron 20 épocas + test; números
en la entrada del **2026-08-07**: WN18RR softmax 0.5536 / degree 0.5533 / FB237 degree 0.3957):

| # | job | modelo | dataset | GPUs | batch (global) | ckpt / log |
|---|-----|--------|---------|-----:|---------------:|------------|
| 1 | `819321` | sparse **degree** | FB15k-237 trans | 2 (DDP) | **32** (16/GPU) | `experiments/sparse_degree_trans_fb237/` · `logs/degree_trans_fb237_819321.log` |
| 2 | `819322` | sparse **softmax** | WN18RR trans | 1 | 32 | `experiments/sparse_wn_trans/` · `logs/sparse_wn_trans_819322.log` |
| 3 | (overlap 812436) | sparse **degree** | WN18RR trans | 1 | 32 | `experiments/sparse_degree_wn_trans/` · `logs/sparse_degree_wn_trans.log` |

Sbatch: `sbatch_degree_trans.sh` (#1, encadenable), `sbatch_sparse_wn_trans.sh` (#2, encadenable),
`run_degree_wn_trans.sh` (#3, step `srun --overlap`).

**Nota de comparabilidad (importante para leer los números luego)**:
- El baseline sparse softmax transductivo FB15k-237 **dim32/L4 = test 0.3965 con batch GLOBAL 16**. La corrida
  #1 usa **batch global 32** (a pedido) ⇒ es un **régimen de batch 2×, NO apples-to-apples en batch** vs 0.3965.
  ⚠️ **CORREGIDO (2026-08-07)**: el baseline corrió a batch global **8**, no 16 (34015 steps/época contra 8504
  de #1) ⇒ el desajuste real es **4×**. Mismo tipo de corrección que la del 2026-08-03 sobre esas filas.
  Antes de concluir "degree ayuda/no ayuda en transductivo" hay que aislar el efecto batch (correr degree a
  batch 16 o softmax a batch 32). Anotado en el header del sbatch.
- #2 y #3 (WN18RR trans) son el par softmax vs degree a **mismo batch 32** ⇒ comparación limpia degree-vs-softmax
  en WN18RR. No hay baseline softmax previo de WN18RR **transductivo** en la bitácora (los WN18RR anteriores eran
  inductivos v1) ⇒ #2 crea ese baseline. WN18RR es chico: ~14 min/época, 20 ep en ~4.7h.

**Hallazgo operacional — DDP funciona en este harness (primer uso de `--devices>1`)**: todas las corridas
previas eran single-GPU. Se verificó que `train.py --devices N` levanta DDP (PL 1.9 + SLURM, backend NCCL, 1
tarea/GPU vía `srun` con `--ntasks-per-node=N`). **Correctitud del eval bajo DDP confirmada por código**: las
métricas en `src/metric.py` (MR/MRR/Hits) usan `add_state(..., dist_reduce_fx='sum')` sobre `rank_sum` y `total`
⇒ agregan bien entre ranks. Para preservar el batch global se usa `batch_size` POR GPU = global/N. Se probó
2 GPU (~0:42 h/época FB15k-237 trans) y 3 GPU antes de fijar la config; con 3 GPU el batch global 16 no es
divisible en enteros (16/3) — limitación a recordar si se quiere batch 16 exacto en multi-GPU (usar 2 o 4 GPU).

**Detalle de reuso de GPU (patrón de siempre)**: nodo `compute-gpu-3-1` con 8×H100; en el pico quedó 8/8
ocupado. La corrida #3 se lanzó reusando la GPU del job **812436** (proceso sparse SIGSTOP'd, PID 2773478,
retiene la asignación) como step `srun --jobid=812436 --overlap`, sin tocar el proceso detenido — así esa GPU
pasó de ociosa a útil sin perder la asignación ni esperar cola.

**Decisión**: dejar las 3 corriendo; registrar test_mrr (y valid_mrr) de cada una al terminar. Para #1, al
comparar contra 0.3965 tener presente la diferencia de batch. Pendiente decidir si se arma watchdog+resume
(los sbatch #1/#2 ya son encadenables vía `last.ckpt`; #3 depende del wall del 812436, ~13h, suficiente para
WN18RR).

---

## 2026-07-22 — Atención sin normalizar (`--attn sigmoid`/`degree`) — opción A: REFUTADA como causa dominante

**Contexto (hipótesis mecánica)**: diagnóstico de por qué el sparse pierde contra NBFNet pese a ser
estructuralmente el más cercano: (1) el **segment-softmax normaliza a Σα=1** por nodo destino ⇒ la
agregación es un promedio ponderado convexo que **borra el conteo de caminos de evidencia** (10 caminos
de soporte puntúan igual que 1 con la misma composición media) y es ciega al grado; NBFNet agrega por
SUMA y conserva ambos. (2) PNA entrega múltiples estadísticos, el softmax uno. (3) La agregación
aprendida sobre-ajusta el train graph (firma valid↑/test↓ ya documentada). La opción A ataca (1), la
causa candidata dominante: reemplazar el softmax por **gates sigmoides por arista SIN normalizar**
⇒ suma ponderada aprendida que conserva conteo de caminos y sensibilidad al grado, manteniendo la
selectividad por query (el gate puede apagar aristas irrelevantes, cosa que NBFNet no puede). Nota de
techo: sparse+expander interpola entre sparse y full, ambos < NBFNet ⇒ más conectividad no saca del
intervalo; cambiar la agregación sí cambia la familia de funciones.

**Implementación (HECHA, sin correr aún)**: flag `--attn {softmax,sigmoid,degree}` en `train.py`,
aplicable a `--model sparse` y `sparse_exp` (misma capa). `softmax` = comportamiento actual (default,
sin cambio); `sigmoid` = opción A (α=σ(logit), sin scatter-amax ni denom ⇒ además algo más barato);
`degree` = fallback intermedio softmax × log(1+grado_in) del destino (reinyecta el conteo como scaler
estilo PNA) por si la suma cruda desestabiliza el entrenamiento. En `src/model.py::
SparseRelationalAttentionLayer` (branch en forward), threading por `SparseGraphTransformer` y
`SparseExpanderGraphTransformer`. Smoke test CPU sintético OK: forward/backward de los 3 modos en
ambos modelos, gradiente fluye por `rel_bias`, y sigmoid ≠ softmax con los mismos pesos.

**Experimento planificado (predicción falsable)**: FB15k-237 ind v1, best config sparse
(`--model sparse --attn sigmoid --hidden_dim 64 --drop 0.0 --learning_rate 1e-3 --num_layer 6
--batch_size 16 --max_epochs 20 --seed 42`), head-to-head vs sparse softmax (0.338) y NBFNet (0.459).
Si el conteo de caminos es el gap dominante ⇒ salto grande (zona 0.42+); si queda ~0.34 ⇒ la causa
(1) no era la dominante, pasar a regularización hacia agregación fija.

**Resultado (FB15k-237 ind v1, full-filtered, corrido 2026-07-22)** — sigmoid y degree lanzados como
steps `srun --overlap` sobre la GPU del job transductivo 812436 (SIGSTOP'd, PID 2773478; asignación
retenida). Logs `logs/sparse_sigmoid_v1.log`, `logs/sparse_degree_v1.log`; ckpts
`experiments/sparse_{sigmoid,degree}_v1/`.

| sparse variant       | valid_mrr | **test_mrr** | H@1 | H@3 | H@10 | MR |
|----------------------|----------:|-------------:|----:|----:|-----:|---:|
| softmax (base)       | 0.446     | 0.338        | —   | —   | 0.451 | — |
| **sigmoid (A)**      | 0.415     | **0.356**    | 0.312 | 0.380 | 0.427 | 200 |
| **degree (A')**      | **0.462** | **0.366**    | 0.300 | 0.410 | 0.473 | 190 |
| *NBFNet (ref)*       | *0.492*   | *0.459*      | *0.371* | *0.520* | *0.605* | *117* |

**Análisis (causa mecánica)**
- **La opción A queda REFUTADA como hipótesis dominante.** Ni la suma cruda (sigmoid) ni el scaler de
  grado (degree) sacan el test de la zona ~0.34-0.37: sigmoid +0.018 y degree +0.028 sobre softmax
  (0.338), lejísimos del 0.42+ que habría confirmado que el conteo de caminos era el gap. **La causa (1)
  —softmax borra el conteo— NO era la dominante.**
- **Señal más nítida = firma de overfit (causa 3).** `degree` produce el **valid_mrr más alto de las tres
  variantes de atención (0.462 > softmax 0.446)** pero el test se queda en 0.366 ⇒ **gap valid→test 0.096,
  el peor de los tres.** Reinyectar el conteo como scaler de grado le da más con qué ajustar el train
  graph, pero NO transfiere al grafo inductivo disjunto. Es exactamente el mal estructural del sparse ya
  documentado (valid↑/test↓), ahora amplificado. El cuello de botella no es la normalización sino la
  **transferencia/sobre-ajuste de la agregación aprendida**.
- sigmoid además **baja el valid** (0.446→0.415): la suma sin normalizar desestabiliza el ajuste
  in-distribution sin comprar test.

**Decisión**
- Opción A (sigmoid/degree) cerrada: no cierra el gap con NBFNet; confirma que el gap del sparse es de
  **transferencia (causa 3)**, no de normalización (causa 1). Siguiente paso: **opción C — regularizar la
  atención hacia la agregación fija** (interpolar α_final = λ·α_aprendida + (1−λ)·uniforme con λ aprendido
  por capa), que ataca directamente la firma valid↑/test↓. Implementada y lanzada a continuación (ver
  entrada de abajo). El flag `--attn {softmax,sigmoid,degree,anchor}` queda en el código (default softmax,
  sin cambio de comportamiento previo).

---

## 2026-07-22 — Opción C: regularizar la atención hacia agregación fija (`--attn anchor`) — REFUTADA (y ancla mal elegida)

**Contexto (hipótesis)**: cerrada la opción A (el gap del sparse no es de normalización sino de
**transferencia/overfit, causa 3**), atacar directamente la firma valid↑/test↓ interpolando la atención
aprendida con una agregación fija: `α_final = λ·α_softmax + (1−λ)·uniforme`, con **λ aprendido por cabeza**
(sigmoide). Idea: λ→0 recupera la media fija (regulariza, transfiere mejor), λ→1 la atención aprendida; el
modelo elige el punto. Implementada en `src/model.py::SparseRelationalAttentionLayer` (branch `anchor` en
forward, `self.anchor_logit = nn.Parameter(zeros(num_heads))` por capa), flag `--attn anchor` en `train.py`.

**Experimento**: FB15k-237 ind v1, best config sparse (`--model sparse --attn anchor --hidden_dim 64
--drop 0.0 --learning_rate 1e-3 --num_layer 6 --batch_size 16 --max_epochs 20 --seed 42`). Log
`logs/sparse_anchor_v1.log`, ckpts `experiments/sparse_anchor_v1/` (best epoch 14).

**Resultado (FB15k-237 ind v1, full-filtered)**

| sparse variant   | valid_mrr | **test_mrr** | H@1  | H@3  | H@10 | MR  |
|------------------|----------:|-------------:|-----:|-----:|-----:|----:|
| softmax (base)   | 0.446     | 0.338        | —    | —    | 0.451 | — |
| sigmoid (A)      | 0.415     | 0.356        | 0.312 | 0.380 | 0.427 | 200 |
| degree (A')      | **0.462** | 0.366        | 0.300 | 0.410 | 0.473 | 190 |
| **anchor (C)**   | 0.442     | **0.278**    | 0.188 | 0.334 | 0.434 | 181 |
| *NBFNet (ref)*   | *0.492*   | *0.459*      | *0.371* | *0.520* | *0.605* | *117* |

**Análisis (causa mecánica)**
- **Opción C REFUTADA: es el PEOR test_mrr de todas las variantes sparse (0.278 < softmax base 0.338),
  y produce el MAYOR gap valid→test hasta ahora (0.442→0.278 = 0.164).** Exactamente lo contrario de lo
  que la regularización buscaba: mezclar la media fija no redujo la firma valid↑/test↓, la **amplificó**.
- **λ aprendido NO se fue a 0.** Leído del checkpoint (best epoch 14), los 48 λ=σ(anchor_logit) se
  estabilizaron en **~0.53–0.69 (media ≈0.60)** en todas las cabezas y capas (capa 0 la más alta,
  ~0.60–0.69; capas 1-5 ~0.53–0.64). El modelo **no prefirió la agregación fija**; mantuvo ~60% atención
  aprendida y mezclar el ~40% de media fija **degradó el test** en vez de estabilizarlo.
- **El ancla elegida es la equivocada (crítica de diseño).** El branch mezcla `α_softmax` con `1/grado_in`
  (mean-pooling), y **ambos suman 1 por nodo destino** ⇒ la mezcla convexa **sigue siendo un promedio
  normalizado**, en la MISMA familia que la causa (1) ya refutada (el softmax que borra el conteo de
  caminos). El ancla `1/grado_in` es *mean-pool*, **NO** la agregación de NBFNet (que **suma** con
  estadísticos PNA y es sensible al grado). O sea: C interpola entre dos agregaciones que **ambas**
  destruyen el conteo de caminos, y la fija es encima un baseline peor. Nunca llegó a testear "regularizar
  hacia NBFNet"; testeó "regularizar hacia mean-pool", que degrada.

**Decisión**
- Opción C cerrada como implementada: no cierra el gap y empeora la transferencia. **Dos conclusiones**:
  (i) confirma de nuevo que el cuello de botella es de transferencia (causa 3), no de normalización;
  (ii) anclar hacia un promedio convexo normalizado (mean-pool) es el ancla incorrecta — para atacar la
  causa (1) haría falta una agregación que **sume y conserve conteo/grado** (estilo suma/PNA), no otro
  promedio convexo. **No re-probar mean-pool como "agregación fija".** El flag `--attn anchor` queda en el
  código. Artefactos: `run_anchor_v1.sh`, `experiments/sparse_anchor_v1/`, `logs/sparse_anchor_v1.log`.

---

## 2026-07-20 — sparse_exp lanzado en FB15k-237 TRANSDUCTIVO + hallazgo de costo del expander

**Contexto operacional**: se lanzó `--model sparse_exp` en FB15k-237 transductivo reusando la GPU del
job **798757** (sbatch `sparse_trans_fb237`, 24h, dueño de la asignación vía un proceso sparse SIGSTOP'd
= PID 2609016 en estado `T`). Primero se detuvo el NBFNet transductivo que corría como step `srun
--overlap` **una vez terminada su época 1** (checkpoint guardado: `epoch=1-step=34016.ckpt` + `last.ckpt`,
**valid_mrr 0.376**, reanudable con `--resume_from ./experiments/nbfnet_trans_fb237/last.ckpt`), y su
watchdog. sparse_exp se lanzó como nuevo step `srun --overlap` detached. **Se probó exp_degree=4 y se
cambió a exp_degree=3** (a pedido, deg4 demasiado lento). Config: L4, dim 32, batch 8, lr 1e-3, drop 0.0,
20 ep, seed 42. Log `logs/sparse_exp_trans_fb237.log`, ckpts `experiments/sparse_exp_trans_fb237/`.
**Resultado MRR: PENDIENTE** (época 0 en curso; a ~2:55h/época solo caben ~4 ep en el wall restante).

**Hallazgo de rendimiento (por qué el expander casi DUPLICA el tiempo/época pese a agregar solo +15,6%
de aristas)**: comparación it/s directa log-contra-log, misma config L4/dim32/batch8 en FB15k-237 trans:

| run | aristas | it/s real | época |
|-----|--------:|----------:|------:|
| sparse simple (log 797546) | 558.735 | **6,1** | ~1:36h |
| sparse_exp deg3 | 645.765 (+15,6%) | **3,65** | ~2:55h |

Ratio de tiempo **1,67x**. Si el costo fuera plano por arista, +15,6% daría 5,3 it/s, no 3,65. Despejando
`1.67 = 1 + (87030/558735)·(c_exp/c_real)` ⇒ **cada arista expander cuesta ~4,3x una arista real del KG**.

**Causa mecánica**: NO es el cómputo (idéntico) ni el conteo de aristas. Es el **patrón de acceso a
memoria**. La capa (`SparseRelationalAttentionLayer.forward`) está dominada por gather/scatter indexados
por `src`/`dst` (`q[:,dst]`, `k[:,src]`, `v[:,src]`, `scatter_reduce_(amax)`, `index_add_(1,dst,·)` del
segment-softmax), kernels **memory-bandwidth-bound** cuya velocidad depende de la localidad:
- **Aristas reales del KG**: grafo *scale-free* (clusterizado en hubs) → filas `q/k/v` de los hubs quedan
  en cache L2 y se reusan, accesos coalescidos. Baratas.
- **Aristas expander**: permutación aleatoria d-regular sobre los N=14.505 nodos → `src`/`dst` uniformes
  y aleatorios → cache misses, accesos no coalescidos, sin reuso. Cada gather/scatter cae en fila distinta
  e impredecible (3-8x más caro en GPU; medido ~4,3x).

La lentitud **es la aleatoriedad del expander** — justo lo que le da valor (atajos fuera del horizonte);
no se puede "clusterizar" sin destruir el punto.

⚠️ **CORREGIDO 2026-08-08 (l): esta explicación mecánica NO se sostiene.** Ordenar el edge_index
por `dst` —que cambia el patrón de acceso de los cinco gathers/scatters dominantes— midió
**1.00×** en 5 casos, incluido WN18RR transductivo donde los tensores de nodo (503 MB) desbordan
la L2 de la A100 (40 MB) por 12×. Y en FB15k-237 los tensores **caben en L2**, así que no puede
haber cache miss y el sobrecosto igual aparece. **El 1.67-1.75× es real y reproducido; la causa
queda ABIERTA.** No re-proponer optimizaciones por inspección: perfilar primero. Palancas: bajar degree (lineal sobre la porción cara),
o (requiere tocar `src/model.py`) meter el expander solo en algunas capas / cachear índices ya en GPU.

**Decisión**: dejar corriendo deg3 (~4 ep en el wall). El costo del expander es esperado y entendido
(memoria, no cómputo). Registrar test_mrr cuando avance para comparar vs sparse (0.4028 dim64/L6, 0.3965
dim32/L4) y NBFNet transductivo.

---

## 2026-07-20 — PLAN: Sparse attention + expander graphs (estilo Exphormer) — `--model sparse_exp`

**Contexto**: añadir un modelo más a la comparación transductiva — el `SparseGraphTransformer` actual
pero aumentado con **aristas expander** (grafo aleatorio d-regular) al estilo Exphormer, para dar
atajos estructurales fuera del horizonte de propagación. Decisión de diseño consultada con el código
de `Exphormer/` (`graphgps/transform/expander_edges.py`, `graphgps/encoder/exp_edge_fixer.py`,
`graphgps/layer/Exphormer.py`).

**Cómo lo hace Exphormer (evidencia)**: las aristas expander se generan como grafo aleatorio d-regular
`(sender, receiver)` **sin relación real**. Pero en la atención **sí** reciben un `edge_attr`: un ÚNICO
embedding aprendido compartido por todas (`self.exp_edge_attr = nn.Embedding(1, dim_edge)`), y un
edge_type dedicado (reales=0, expander=1, virtuales/globales=2). O sea: no van peladas ni con relación
KG real, sino con un **tipo de arista aprendido propio**.

**Decisión de diseño (port a la atención relacional del harness)**: las aristas expander llevan **una
relación sintética reservada `R_exp`** (una fila nueva en la tabla de embeddings de relación,
`num_relation += 1`), usada solo por ellas — no una relación KG real (sería inyectar hechos falsos,
las expander son aleatorias) ni nada (Exphormer sí las tipa). `R_exp` entra en los dos canales
relacionales que ya usa el sparse: el bias escalar `b[head, R_exp]` (rol del edge-type de Exphormer,
estilo Graphormer) y la corrección de valor `g[R_exp]` (composición DistMult). Simétricas → ambas
direcciones con `R_exp`. **Inductive-safe**: `R_exp` se comparte train/test como toda relación y las
aristas son estructurales (sin identidad de entidad).

**Advertencias registradas**:
- **Lista negra #2** ("NO expander en inductivo — metían ruido, Exphormer-Max") es específica de
  INDUCTIVO; en transductivo Exphormer-Max iba bien y los experimentos sparse actuales son
  transductivos ⇒ este modelo NO viola la lista negra *en transductivo*. Probarlo inductivo sí sería
  re-pisar terreno refutado.
- **Tensión con GOALS**: las expander son atajos estructurales aleatorios (conectividad fuera del
  horizonte), no evidencia composicional/relacional (caminos). Hay razón teórica para dudar que ayuden;
  es una prueba empírica de si acortar el diámetro efectivo mejora la propagación en transductivo.

**Implementación (HECHA)**: `src/model.py::SparseExpanderGraphTransformer` + helper
`generate_expander_edges` (port del permutation algorithm de Exphormer, simétrico, sin self-loops,
cacheado por num_nodes). Flags nuevos en `train.py`: `--model sparse_exp` y `--exp_degree` (default 4).
Las tablas relacionales de la capa se dimensionan a `num_relation+2` (self-loop en `num_relation`,
expander en `num_relation+1`). Smoke test CPU OK: forward/backward corre y la fila `R_exp` de `rel_bias`
recibe gradiente (se usa). Comando (best config sparse):
`python train.py --data_path ./data/fb15k-237 --model sparse_exp --exp_degree 4 --num_layer 4
--hidden_dim 32 --num_heads 8 --batch_size 8 --learning_rate 1e-3 --drop 0.0 --seed 42 --max_epochs 20
--checkpoint_save_path ./experiments/sparse_exp_trans_fb237`.
**ESTADO**: lanzado el 2026-07-20 (deg3 tras descartar deg4 por lento); ver entrada de arriba
"sparse_exp lanzado en FB15k-237 TRANSDUCTIVO + hallazgo de costo del expander". Resultado MRR pendiente.

---

## 2026-07-20 — Sparse attention (labeling trick, sin nbfv) en FB15k-237 TRANSDUCTIVO: capacidad grande vs chica

**Contexto**: primer experimento del modelo `SparseGraphTransformer` (`--model sparse`: atención
sparse por adyacencia, labeling trick `x⁰_v = emb(r_q)` si `v==head` else `0`, SIN V de NBFNet) en
FB15k-237 **transductivo** (no inductivo). Dos configs para medir sensibilidad a capacidad vs costo:
grande (dim 64, L6) y chica (dim 32, L4). Seed 42, 20 ep, batch 16, full-filtered. Logs
`logs/sparse_trans_fb237_795838.log` (grande) y `logs/sparse_trans_fb237_797546.log` (chica);
ckpts `experiments/sparse_trans_fb237/` y `experiments/sparse_trans_fb237_4_32/`.

**Resultado (FB15k-237 transductivo, full-filtered)**

| config          | params | valid_mrr | test_mrr | H@1 | H@3 | H@10 | MR | mem GPU | t/época | t test |
|-----------------|-------:|----------:|---------:|----:|----:|-----:|---:|--------:|--------:|-------:|
| **dim 64, L6**  | 440 K | 0.407 | **0.4028** | 0.306 | 0.443 | 0.595 | 116.4 | 39.8 GB | ~3:53 h | 9:12 |
| dim 32, L4      | 126 K | 0.402 | 0.3965 | 0.300 | 0.437 | 0.591 | 128.4 | 14.8 GB | ~1:36 h | 3:24 |

(best-valid: grande epoch 12, chica epoch 19.)

**Análisis (causa mecánica)**
- **La capacidad grande casi no compra métrica pero cuesta ~2.5× en todo.** Δ test_mrr solo +0.0063
  (0.4028 vs 0.3965, +1.6% rel.), y H@10 prácticamente empata (0.595 vs 0.591). Pero el grande cuesta
  **2.4× por época** (3:53 h vs 1:36 h), **2.7× en test** (9:12 vs 3:24) y **2.7× memoria** (39.8 vs
  14.8 GB). El único gap real es MR (116 vs 128). ⇒ **dim 32 / L4 da ~98% del rendimiento a ~40% del
  costo**: mejor relación coste/beneficio en transductivo. El modelo satura rápido en capacidad, la
  señal no está limitada por parámetros.
- **Sigue por debajo del message passing**: NBFNet transductivo en FB15k-237 ronda ~0.415 MRR en
  literatura, el sparse (0.403) queda −0.012 debajo. Consistente con el hallazgo inductivo (la atención
  no supera al MP). Baseline propio (mismo eval full-filtered) **EN CURSO** — ver abajo.

**Decisión**
- Config chica (dim 32, L4) preferida para iterar en transductivo por costo. Confirmación (débil, falta
  baseline propio) de que el sparse tampoco supera al MP en régimen transductivo. Baseline NBFNet
  transductivo lanzado para cerrar la comparación (abajo).

### NBFNet transductivo FB15k-237 — baseline EN CURSO (lanzado 2026-07-20)

**Contexto**: cerrar la comparación con el sparse transductivo con un NBFNet propio, mismo eval
full-filtered. Config alineada a `NBFNet/config/knowledge_graph/fb15k237.yaml` + best config validada
del harness: `--model nbfnet --aggregate pna`, L6, dim 32, distmult, short_cut, layer_norm, dependent,
lr 5e-3, drop 0.1, batch 16, 20 ep, seed 42. Log `logs/nbfnet_trans_fb237_manual.log`, ckpts
`experiments/nbfnet_trans_fb237/`, sbatch encadenable de resume `sbatch_nbfnet_trans.sh`.

**Detalle operacional (reuso de GPU sin perder la asignación)**: no había GPUs libres (cola ~2 días).
Se aprovechó el job 798757 (sparse `_3_32`, no necesario) **suspendiéndolo con SIGSTOP** (proceso vivo
→ SLURM mantiene la asignación; retiene ~12.5 GB VRAM, irrelevante) y lanzando NBFNet como job step
superpuesto (`srun --jobid=798757 --overlap`) en la misma GPU. NBFNet es **compute-bound** (~2.5 h/época
en H100 a batch 16; subir batch no acelera) ⇒ en las ~18 h restantes del job de 24 h solo caben **~6-7
épocas**. Se añadió `--eval_only` a `train.py` y un watchdog (`watchdog_nbf_trans.sh`) que ~1.9 h antes
del wall mata el entrenamiento y corre el test sobre el best-valid ckpt → resultado en
`logs/nbfnet_trans_test.log`. `last.ckpt` permite reanudar a 20 ep en futuros jobs
(`sbatch sbatch_nbfnet_trans.sh`; al llegar a época 20 corre `trainer.test` automático) para el número
final comparable.

**Progreso parcial**: valid_mrr 0.365 tras época 0 (NBFNet converge rápido). **PENDIENTE registrar** el
test_mrr del NBFNet (parcial ~6-7 ep, y final a 20 ep cuando haya GPU) para completar la comparación vs
sparse transductivo (dim64/L6 test 0.4028; dim32/L4 test 0.3965).

---

## 2026-07-08 — source_rw (labeling trick condicionado a la fuente) en full attention, FB15k-237 v1+v2

**Contexto**: probar un labeling trick NATIVO del transformer, condicionado a la query, como
alternativa a los PE node-only ya refutados (RWSE local, LapPE global). Feature por nodo
v = proj([P^1[head,v], …, P^8[head,v]]) sumada a x^0, donde P=D^-1 A (adyacencia simetrizada,
relación ignorada) y la fila head de P^k son las probabilidades de landing de un random walk de k
pasos que arranca en el head de la query. La diferencia clave vs RWSE/LapPE: **es query-conditioned**
(depende del source de la query, no sólo de la estructura del nodo) → en principio es la "evidencia
composicional/relacional fuera del horizonte" que GOALS.md deja abierta, no una coordenada de nodo.
Inductivo-safe (sólo estructura + head, sin identidad de entidad). Implementado en `src/model.py`
(`compute_source_rw`/`source_rw_features`, flags `--use_source_rw --source_rw_dim`). Best config del
full (dim 64, drop 0.0, lr 1e-3, L6, 20 ep, seed 42), apples-to-apples vs full sin PE. Script
`run_source_rw.sh`, logs `logs/source_rw_full_v{1,2}.log`, ckpts `experiments/source_rw_full_v{1,2}/`.

**Resultado (FB15k-237 ind v1+v2, full-filtered)**

| split | full baseline (no PE) | **full + source_rw** | Δ test | con-source_rw: H@1 / H@3 / H@10 / MR | NBFNet |
|-------|----------------------:|---------------------:|-------:|-------------------------------------:|-------:|
| v1 valid | 0.429 | **0.234** |        |                                      | 0.492 |
| v1 test  | 0.375 | **0.313** | **−0.062** | 0.266 / 0.332 / 0.395 / 384          | 0.459 |
| v2 valid | 0.461 | **0.279** |        |                                      | 0.483 |
| v2 test  | 0.491 | **0.363** | **−0.128** | 0.292 / 0.402 / 0.479 / 198          | 0.526 |

**Análisis (causa mecánica)**
- **source_rw es el encoding MÁS dañino probado, y el patrón es distinto al de RWSE/LapPE.** RWSE/LapPE
  degradaban poco y sólo en test (firma de overfit: valid se mantiene o sube, test baja). Aquí
  **valid Y test se derrumban juntos en ambos splits**: valid v1 0.429→0.234 (−0.195), v2 0.461→0.279
  (−0.182). No es sobre-ajuste estructural: es **degradación neta, in-distribution incluida**.
- El canal query-conditioned (proj de las landing probs desde el head) no aporta evidencia nueva; **le
  compite/ahoga al x^0 = emb(r_q) la señal útil** y perjudica hasta el ajuste del train graph. train_loss
  se mantuvo bajo (v1 0.147) ⇒ el modelo ajusta algo, pero el feature extra empeora el ranking incluso
  en validación. Que sea condicionado a la fuente (y no node-only) no lo salva: sigue siendo re-codificar
  estructura dentro del horizonte del MP, ahora con más ruido por depender del head.
- **Muy por debajo de NBFNet en ambos** (v1 0.313 vs 0.459; v2 0.363 vs 0.526). Ni de lejos.

**Decisión**
- source_rw se une a RWSE (local) y LapPE (global) como encoding NO confiable para la atención — de
  hecho el peor. Confirma otra vez que la señal útil no viene de re-codificar la estructura del grafo
  como feature por nodo, ni siquiera condicionada a la query. Candidato a lista negra #7 de CLAUDE.md
  (ampliar de "PE por nodo" a "labeling/encoding estructural derivado de random walks, incl.
  condicionado a la fuente"). Artefactos: `run_source_rw.sh`, `experiments/source_rw_full_v{1,2}/`,
  `logs/source_rw_full_v{1,2}.log`.

---

## 2026-07-08 — LapPE en FB15k-237 ind v2 (réplica del v1): el bump del full NO es robusto

**Contexto**: replicar LapPE en FB15k-237 ind v2 con los mismos 3 modelos y la misma best config
(dim 64, drop 0.0, lr 1e-3, L6, 20 ep, seed 42, `--use_lappe --lappe_dim 16`) para testear la
pregunta que dejó abierta v1: **¿es robusto el +0.035 que LapPE le dio al full attention?** Script
`run_lappe_v2.sh`, logs `logs/lappe_*_v2.log`, ckpts `experiments/lappe_*_v2/`.

**Resultado (FB15k-237 ind v2, full-filtered)**

| modelo             | valid sin→con | **test sin→con** | Δ test v2 | *Δ test v1* | con-LapPE: H@1 / H@10 / MR |
|--------------------|--------------:|-----------------:|----------:|------------:|---------------------------:|
| Full attention     | 0.461 → 0.469 | 0.491 → **0.496** | +0.005 | *+0.035* | 0.397 / 0.681 / 98 |
| Sparse adyacencia  | 0.471 → 0.474 | 0.450 → **0.445** | −0.005 | *−0.003* | 0.358 / 0.592 / 112 |
| Sparse_nbfv (V=NBF)| 0.488 → 0.484 | 0.527 → **0.524** | −0.003 | *+0.013* | 0.424 / 0.687 / 57 |
| *NBFNet (ref)*     | *0.483*       | *0.526*          | —      | —      | *0.416 / 0.727 / 49* |

**Análisis (causa mecánica)**
- **El bump del full NO se replica.** v1 daba +0.035 (0.375→0.410); v2 solo +0.005 (0.491→0.496),
  dentro del ruido. La "primera evidencia a favor de PE global" de v1 era un **artefacto de split**,
  no un mecanismo estructural. Mismo fracaso de robustez que ya se vio con sparse+RWSE (que invertía
  signo entre splits): un efecto que no sobrevive al cambio de split no es una mejora real.
- **Sparse**: negativo-plano en ambos (−0.003 v1, −0.005 v2). La coordenada global no le sirve
  porque solo atiende vecinos; consistente con v1.
- **Sparse_nbfv**: +0.013 (v1) → −0.003 (v2), inconsistente y dentro de ruido; su señal es el V de
  NBFNet, no LapPE.
- **Ninguno supera a NBFNet en v2** (0.526): full+LapPE 0.496 (−0.030), sparse_nbfv+LapPE 0.524
  (−0.002, y por debajo de su propio baseline 0.527). El techo sigue siendo el message passing.

**Decisión**
- **LapPE se une a RWSE como PE no confiable para atención.** Lo que parecía la única pista viva tras
  v1 (PE global ayuda al full) NO resiste la réplica en v2. Ni PE local (RWSE) ni PE global (LapPE)
  dan una mejora robusta que supere a NBFNet. Lista negra #7 de CLAUDE.md actualizada: el matiz
  "PE global ayuda" se retira; queda como efecto dependiente de split. Artefactos: `run_lappe_v2.sh`,
  `experiments/lappe_*_v2/`, `logs/lappe_*_v2.log`.

---

## 2026-07-08 — LapPE (Laplacian positional encoding) en los 3 modelos de atención, FB15k-237 ind v1

**Contexto**: probar un PE ESTRUCTURAL GLOBAL (LapPE: los k=16 autovectores no triviales del
Laplaciano normalizado simétrico L = I − D^-1/2 A D^-1/2, autovalores más chicos), sumado a x^0,
como contraste con RWSE (que es local). LapPE depende solo de la estructura (no de identidad de
nodo) → inductivo-safe, NO viola lista negra #3. Ambigüedad de signo de los autovectores tratada
con sign-flip aleatorio en train (canónico). Implementado en `src/model.py` (`compute_lappe`/
`lappe_features`, flags `--use_lappe --lappe_dim`). Best config de cada modelo (dim 64, drop 0.0,
lr 1e-3, L6, 20 ep, seed 42), apples-to-apples vs baselines sin PE. Script `run_lappe_v1.sh`,
logs `logs/lappe_*_v1.log`, ckpts `experiments/lappe_*_v1/`.

**Resultado (FB15k-237 ind v1, full-filtered)**

| modelo             | valid sin→con | **test sin→con** | Δ test | con-LapPE: H@1 / H@3 / H@10 / MR |
|--------------------|--------------:|-----------------:|-------:|---------------------------------:|
| Full attention     | 0.429 → 0.424 | 0.375 → **0.410** | **+0.035** | 0.351 / 0.441 / 0.524 / 227 |
| Sparse adyacencia  | 0.446 → 0.453 | 0.338 → **0.335** | −0.003 | 0.285 / 0.363 / 0.410 / 197 |
| Sparse_nbfv (V=NBF)| 0.472 → 0.477 | 0.421 → **0.434** | +0.013 | 0.346 / 0.500 / 0.571 / 115 |
| *NBFNet (ref)*     | *0.492*       | *0.459*          | —      | *0.371 / 0.520 / 0.605 / 117* |

**Análisis (causa mecánica)**
- **LapPE es el PRIMER encoding estructural que ayuda — y solo al full attention** (+0.035, el
  mejor resultado de full attention hasta ahora: 0.375 base, 0.366 con RWSE, **0.410 con LapPE**).
  Contraste directo con RWSE, que dañaba el full (−0.009).
- **Por qué ayuda al full y no a los sparse**: LapPE es GLOBAL (coordenadas espectrales de todo el
  grafo), RWSE es LOCAL (diag(P^k), dentro del horizonte de k saltos). El full attention es all-pairs
  pero no tenía NINGUNA noción de posición/distancia global; LapPE le da un sistema de coordenadas
  para distinguir nodos lejanos. Eso es señal de FUERA del horizonte de propagación → primera
  evidencia empírica a favor de la única dirección que GOALS.md deja abierta.
- **Sparse**: plano con firma de overfit. valid sube (0.446→0.453) pero test queda igual
  (0.338→0.335): como solo atiende a vecinos, la coordenada global no le sirve para propagar mejor,
  solo le da más con qué sobre-ajustar el train graph. Mismo patrón estructural de siempre.
- **Sparse_nbfv**: +0.013 chico. Su columna vertebral es el V de NBFNet (MR 115 ≈ NBFNet 117);
  LapPE aporta un poco pero no rompe el techo.
- **Ninguno supera a NBFNet (0.459).** Full+LapPE (0.410) sigue −0.049 debajo. El signo positivo es
  real e interpretable (info global espectral), pero no cierra el gap con message passing.

**Decisión**
- LapPE NO es otro RWSE: matiza la conclusión "todo PE es redundante". PE LOCAL (RWSE) sí es
  redundante con el MP; PE GLOBAL (LapPE) aporta señal real de fuera del horizonte al full attention,
  aunque insuficiente para ganarle a NBFNet en v1. Lista negra #7 de CLAUDE.md reformulada (ya no
  prohíbe todo PE; distingue local redundante vs global útil-pero-insuficiente).
- Pista abierta a seguir: (i) ¿es robusto el +0.035 del full en v2 / WN18RR?; (ii) ¿escala el
  efecto global con más lappe_dim o capacidad del full? Artefactos: `run_lappe_v1.sh`,
  `experiments/lappe_*_v1/`, `logs/lappe_*_v1.log`.

---

## 2026-06-28 — RWSE (random-walk structural encoding) en los 3 modelos de atención, FB15k-237 ind v1

**Contexto**: probar si añadir un encoding ESTRUCTURAL por nodo (RWSE: diag(P^k) de la
adyacencia relation-agnostic, k=1..16) a x^0 mejora los modelos de atención. RWSE es
puramente estructural → inductivo-safe (depende de la estructura local, no de identidad de
nodo), NO viola lista negra #3. Implementado en `src/model.py` (`compute_rwse`/`rwse_features`,
flags `--use_rwse --rwse_dim`). Best config de cada modelo (dim 64, drop 0.0, lr 1e-3, L6,
20 ep, seed 42). Apples-to-apples vs los baselines sin RWSE. Scripts `run_rwse_v1.sh`, logs
`logs/rwse_*_v1.log`, ckpts `experiments/rwse_*_v1/`.

**Resultado (FB15k-237 ind v1, full-filtered)**

| modelo            | valid sin→con | **test sin→con** | Δ test | con-RWSE: H@1 / H@3 / H@10 / MR |
|-------------------|--------------:|-----------------:|-------:|--------------------------------:|
| Full attention    | 0.429 → 0.434 | 0.375 → **0.366** | −0.009 | 0.310 / 0.407 / 0.459 / 227 |
| Sparse adyacencia | 0.446 → 0.438 | 0.338 → **0.292** | −0.046 | 0.227 / 0.337 / 0.376 / 211 |
| Sparse_nbfv (V=NBF)| 0.472 → 0.474 | 0.421 → **0.424** | +0.003 | 0.332 / 0.480 / 0.593 / 103 |
| *NBFNet (ref)*    | *0.492*       | *0.459*          | —      | *0.371 / 0.520 / 0.605 / 117* |

**Análisis (causa mecánica)**
- **RWSE no ayuda a ninguno; el techo sigue siendo NBFNet (0.459).**
- **Full**: neutral-negativo. valid sube un pelo (0.429→0.434) pero test baja (0.375→0.366).
  El encoding estructural no aporta evidencia nueva; lo poco que añade no transfiere.
- **Sparse adyacencia**: RWSE lo **empeora claramente** (test −0.046). Agrava su firma de
  overfit estructural: valid se mantiene ~0.44 mientras test cae fuerte (0.338→0.292) → más
  features estructurales para sobre-ajustar el train graph, peor transferencia al grafo
  inductivo disjunto. Mismo patrón que ya tenía, amplificado.
- **Sparse_nbfv**: plano (+0.003, dentro de ruido). Su señal viene del V de NBFNet; sumar
  RWSE encima no mueve nada.

**Decisión**
- RWSE descartado como mejora para atención en FB15k-237 v1: es estructura DENTRO del horizonte
  de propagación que el message passing ya captura → redundante (full/sparse_nbfv) o dañina
  (sparse, abre más margen de overfit). Consistente con GOALS.md: la señal útil debe venir de
  FUERA del horizonte de propagación, no de re-codificar estructura local. Añadido a lista
  negra de CLAUDE.md. Artefactos: `run_rwse_v1.sh`, `experiments/rwse_*_v1/`, `logs/rwse_*_v1.log`.

### Réplica en FB15k-237 ind v2 (misma best config, mismos 3 modelos)

**Resultado (FB15k-237 ind v2, full-filtered)**

| modelo            | valid sin→con | **test sin→con** | Δ test | con-RWSE: H@1 / H@10 / MR |
|-------------------|--------------:|-----------------:|-------:|--------------------------:|
| Full attention    | 0.461 → 0.455 | 0.491 → **0.477** | −0.014 | 0.382 / 0.652 / 154 |
| Sparse adyacencia | 0.471 → 0.465 | 0.450 → **0.472** | +0.022 | 0.382 / 0.644 / 109 |
| Sparse_nbfv (V=NBF)| 0.488 → 0.486 | 0.527 → **0.529** | +0.002 | 0.427 / 0.703 / 60 |
| *NBFNet (ref)*    | *0.483*       | *0.526*          | —      | *0.416 / 0.727 / 49* |

**Lectura cruzada v1 vs v2 (Δ test_mrr por RWSE)**

| modelo      | Δ v1   | Δ v2   |
|-------------|-------:|-------:|
| Full        | −0.009 | −0.014 |
| Sparse      | −0.046 | +0.022 |
| Sparse_nbfv | +0.003 | +0.002 |

**Análisis (causa mecánica)**
- **RWSE es inconsistente y nunca supera a NBFNet en ningún split.**
- **Full**: neutral-negativo en ambos (−0.009, −0.014). El encoding no aporta evidencia nueva.
- **Sparse**: el signo se **invierte** entre splits (daña −0.046 en v1, ayuda +0.022 en v2).
  Efecto NO robusto / dependiente del split, no un mecanismo real: en v2 la estructura local
  de RWSE casualmente transfiere, en v1 sobre-ajusta. Señal poco confiable.
- **Sparse_nbfv**: plano en ambos (+0.003, +0.002, ruido); su señal es el V de NBFNet. En v2
  el mejor (sparse_nbfv+RWSE 0.529) solo EMPATA a NBFNet (0.526), heredando su V — no lo supera.

**Decisión**
- Confirma v1 en v2: RWSE no es una mejora confiable para atención y no rompe el techo del
  message passing. Línea cerrada. Artefactos: `run_rwse_v2.sh`, `experiments/rwse_*_v2/`,
  `logs/rwse_*_v2.log`.

### Réplica en WN18RR ind v1 (corrido 2026-06-29, registrado 2026-07-08)

**Contexto**: los runs terminaron el 29-jun pero quedaron sin registrar (aparecían como
"pendiente/corriendo"). Misma best config (dim 64, drop 0.0, lr 1e-3, L6, 20 ep, seed 42).
Script `run_rwse_wn_v1.sh`, logs `logs/rwse_*_wn_v1.log`, ckpts `experiments/rwse_*_wn_v1/`.

**Resultado (WN18RR ind v1, full-filtered)**

| modelo            | valid (con) | **test sin→con** | Δ test | con-RWSE: H@1 / H@10 / MR |
|-------------------|------------:|-----------------:|-------:|--------------------------:|
| Full attention    | 0.514       | 0.673 → **0.670** | −0.003 | 0.638 / 0.734 / 120 |
| Sparse adyacencia | 0.535       | 0.738 → **0.677** | −0.061 | 0.628 / 0.769 / 56 |
| Sparse_nbfv (V=NBF)| 0.557      | 0.740 → **0.728** | −0.012 | 0.684 / 0.798 / 44 |
| *NBFNet (ref)*    | *0.578*     | *0.740*          | —      | *0.689 / 0.822 / 30* |

**Análisis (causa mecánica)**
- **RWSE daña los 3 en WN18RR también; ninguno supera a NBFNet (0.740).**
- **Sparse: el más dañado otra vez (−0.061, 0.738→0.677)** — su firma de overfit estructural
  (más features estructurales → sobre-ajusta el train graph, transfiere peor). Mismo patrón que
  FB15k-237 v1 (−0.046). El sparse pasa de EMPATAR a NBFNet sin PE (0.738) a quedar claramente
  debajo con RWSE.
- **Full (−0.003) y sparse_nbfv (−0.012)**: neutral-negativos, el encoding local no aporta.

**Decisión**
- Cierra RWSE en los 3 datasets (FB15k-237 v1+v2, WN18RR v1). Conclusión uniforme: RWSE nunca
  supera a NBFNet y perjudica al sparse en todos los splits salvo v2 (donde el signo casualmente
  se invierte). Confirma lista negra #7. Artefactos: `run_rwse_wn_v1.sh`, `experiments/rwse_*_wn_v1/`,
  `logs/rwse_*_wn_v1.log`.

---

## 2026-06-23 — Los 4 modelos en FB15k-237 ind v2 (best config de cada uno)

**Contexto**: a pedido del usuario, replicar el head-to-head de v1 ahora en FB15k-237 ind
v2. Pregunta explícita: ¿corre el full attention en v2? **Sí** — grafo chico para O(N²):
train graph N≈2608 nodos, test graph (`_ind`) N≈1660. Cada modelo con su mejor config:
NBFNet dim 32 / drop 0.1 / lr 5e-3 / pna; full/sparse/sparse_nbfv dim 64 / drop 0.0 / lr
1e-3. Todos L6, batch 16, 20 ep, seed 42. Script `run_v2_all.sh`, logs `experiments/v2_*.log`,
ckpts `experiments/v2_*/`.

**Resultado (FB15k-237 ind v2, full-filtered)**

| test     | NBFNet | full attn | sparse adj | sparse_nbfv (V=NBF) |
|----------|-------:|----------:|-----------:|--------------------:|
| valid_mrr| 0.483  | 0.461     | 0.471      | **0.488**           |
| test_mrr | 0.526  | 0.491     | 0.450      | **0.527**           |
| Hits@1   | 0.416  | 0.389     | 0.369      | **0.431**           |
| Hits@3   | **0.595** | 0.544  | 0.494      | 0.586               |
| Hits@10  | **0.727** | 0.686  | 0.586      | 0.690               |
| MR       | **49.3** | 76.6    | 112.6      | 54.5                |

Uso de memoria GPU (H100): full attn **31 GB**, sparse_nbfv 7.1 GB, sparse 3.6 GB, NBFNet
2.0 GB. El full attn cabe de sobra; solo es el más lento (~3.5 it/s, 3.5 min/época).

**Análisis (causa mecánica)**
- **Se repite el patrón de v1**: ni full attn ni sparse-adyacencia superan a NBFNet.
  Orden test: sparse_nbfv (0.527) ≈ NBFNet (0.526) > full (0.491) > sparse adj (0.450).
- **Sparse-adyacencia sigue siendo el peor** (0.450) y reproduce la firma de sobre-ajuste
  estructural: valid (0.471) > su propio test (0.450) → ajusta mejor el train graph y
  transfiere peor al grafo inductivo disjunto. Mismo eco que v1 y que los expander.
- **sparse_nbfv empata/roza a NBFNet** (0.527 vs 0.526, mejor Hits@1 0.431): como en
  WN18RR, su V es el de NBFNet y la reponderación por adyacencia no degrada aquí. Sigue sin
  **superar** de forma significativa al MP (Δ test_mrr +0.001, dentro de ruido).
- NBFNet sube a 0.526 (de 0.459 en v1): v2 es un split más grande/fácil.

**Decisión**
- Confirma GOALS.md también en v2: ninguna atención supera de forma clara a NBFNet. El único
  candidato que empata (sparse_nbfv) hereda el V del message passing por construcción, no
  aporta señal nueva. Artefactos: `experiments/v2_{nbfnet,rfat,sparse,sparse_nbfv}/`,
  `experiments/v2_*.log`, `run_v2_all.sh`.

---

## 2026-06-22 — Revisión: confirmación de logs + recuperación del historial CPA/A1

**Contexto**: lectura de CLAUDE.md y los logs/documentos del proyecto. Sin correr nada
nuevo; verificación cruzada de números y rescate de resultados previos relevantes.

**Resultado**
- Los `.log` de `experiments/` (gt/nbf/sweep_*/sparse_*) **coinciden exactamente** con la
  tabla head-to-head de CLAUDE.md. Implementación de NBFNet validada (test_mrr 0.459, en
  rango de literatura ~0.42–0.46).
- Se recuperó del `SESSION_LOG.md` (proyecto previo, legacy) que la idea **CPA
  (Compositional Pivot Attention)** y su sucesora **A1 (candidate-set attention)** YA
  fueron probadas y **refutadas** (R12–R15). Ver entrada histórica abajo.

**Análisis**
- La pregunta "¿alimentar Q/K/V de la atención desde NBFNet?" se mapea a tres casos:
  (V = NBFNet → KnowFormer, lista negra #6, decorativa); (Q/K = NBFNet → línea Structural
  Query-PE, load-bearing↑ pero MRR plano); (valores composicionales de NBFNet fuera del
  horizonte → **eso ES CPA, ya refutado**).

**Decisión**
- Crear `GOALS.md` y este `SESSION_NOTES.md`.
- Pendiente (decisión usuario): añadir CPA/A1 a la lista negra de CLAUDE.md y/o guardar
  memoria con la causa mecánica de CPA (corridas NBFNet ancladas colapsan a cos 0.992).

### Experimento nuevo — sparse attention con V desde stream NBFNet (lista negra #6, a pedido)

**Contexto**: a pedido del usuario, para completar la tabla. `SparseNBFValueTransformer`
(`src/model.py`, `--model sparse_nbfv`): atención sparse por adyacencia donde Q,K salen del
stream de atención (labeling trick) pero **V = representaciones de nodo de un NBFNet corrido
aparte** (`NBFNet.encode()`, h^L_{u->v}). Ambos streams entrenan end-to-end. Recrea a
propósito el patrón refutado de KnowFormer (V-stream alimentando la atención). Config B
EXACTA del mejor sparse adyacencia (dim 64, drop 0.0, L6, lr 1e-3, 20 ep, seed 42) para
comparación apples-to-apples vs la fila 0.338.

**Resultado (FB15k-237 ind v1, best ckpt epoch 7)**

| test     | NBFNet | full attn | sparse adj | **sparse_nbfv (V=NBF)** |
|----------|-------:|----------:|-----------:|------------------------:|
| valid_mrr| 0.492  | 0.429     | 0.446      | **0.472**               |
| test_mrr | **0.459** | 0.375  | 0.338      | **0.421**               |
| Hits@1   | 0.371  | —         | —          | 0.339                   |
| Hits@3   | 0.520  | —         | —          | 0.476                   |
| Hits@10  | 0.605  | 0.471     | 0.451      | 0.554                   |
| MR       | 117    | —         | —          | 118                     |

**Análisis (causa mecánica)**
- Es el **mejor de todas las variantes de atención** (0.421 > full 0.375 > sparse 0.338):
  hereda casi toda la señal de NBFNet a través de V. MR ~118 ≈ NBFNet (117) confirma que la
  columna vertebral es el V del message passing.
- Pero **sigue −0.038 MRR bajo NBFNet puro (0.459)**: la atención encima del V de NBFNet es
  **net-ligeramente-destructiva**. Reponderar por adyacencia un V que ya resuelve la query
  no añade nada y cuesta un poco. Confirma lista negra #6 / diagnóstico KnowFormer (atención
  sobre V-stream de MP = redundante). No supera al MP, lo degrada.

**Decisión**
- Línea "V desde NBFNet en la atención" cerrada: redundante por construcción, como se
  predijo. Queda en la tabla como evidencia. Artefactos: `SparseNBFValueTransformer` +
  `--model sparse_nbfv`, `experiments/sparse_nbfv_B/`, `logs/sparse_nbfv_B.log`.

---

### Experimento nuevo — los 4 modelos en WN18RR ind v1 (best config de cada uno)

**Contexto**: a pedido del usuario, replicar el head-to-head en WN18RR ind v1. Cada modelo
con su mejor config: NBFNet dim 32 / drop 0.1 / lr 5e-3 / pna (= config NBFNet repo
wn18rr); full/sparse/sparse_nbfv dim 64 / drop 0.0 / lr 1e-3. Todos L6, batch 16, 20 ep,
seed 42. Script `run_wn18rr_v1.sh`, logs `logs/wn_*_v1.log`.

**Resultado (WN18RR ind v1, full-filtered)**

| test     | NBFNet | full attn | sparse adj | sparse_nbfv (V=NBF) |
|----------|-------:|----------:|-----------:|--------------------:|
| valid_mrr| 0.578  | 0.521     | 0.567      | 0.578               |
| test_mrr | **0.740** | 0.673  | 0.738      | **0.740**           |
| Hits@1   | 0.689  | 0.638     | 0.686      | 0.691               |
| Hits@3   | 0.774  | 0.691     | 0.774      | 0.766               |
| Hits@10  | 0.822  | 0.739     | 0.819      | 0.832               |
| MR       | 29.6   | 97.4      | 26.7       | 29.5                |

**Análisis**
- NBFNet 0.740 ≈ literatura (paper NBFNet ~0.741 wn18rr v1) → baseline validado.
- **Cuadro INVERTIDO respecto a FB15k-237**: aquí la **full attention es la PEOR** (0.673,
  −0.067) — el mecanismo global daña en WN18RR (grafo local/jerárquico; eco del log previo
  "WN18RR selecciona en contra de cualquier mecanismo global").
- **Sparse adyacencia (0.738) ≈ NBFNet** (−0.002): restringir a vecinos cierra el gap, al
  revés que en FB15k-237 (donde sparse era el peor). En WN18RR lo local es lo que importa.
- **sparse_nbfv (0.740) EMPATA NBFNet** (test_mrr idéntico, H@10 0.832 > 0.822): su V es
  NBFNet y la reponderación por adyacencia no degrada en este régimen.

**Síntesis cruzada (FB15k-237 + WN18RR)**: ninguna variante de atención **supera** a NBFNet
en ningún dataset. FB15k-237 (composicional): todas pierden, sparse el peor. WN18RR
(local): full pierde, sparse empata. El techo es el message passing en ambos regímenes.

**Decisión**
- Confirma GOALS.md: la dirección "atención supera a NBFNet" sigue sin evidencia a favor en
  ningún régimen. Artefactos: `experiments/wn_{nbf,full,sparse,sparse_nbfv}_v1/`,
  `logs/wn_*_v1.log`, `run_wn18rr_v1.sh`.

---

## 2026-06-21 — Reinicio desde cero: RFAT (full) vs NBFNet vs Sparse en FB15k-237 ind v1

**Contexto**: arranque del proyecto nuevo (RFAT, `src/model.py` escrito desde cero, NO
KnowFormer). Sanity check de `GOALS.md`: ¿atención full supera a NBFNet?

**Resultado (FB15k-237 ind v1, full-filtered, seed 42, 6 capas, dim 32, lr 5e-3, 20 ep)**

| test     | NBFNet (pna) | RFAT full (base) | RFAT full (best) | Sparse adyacencia |
|----------|-------------:|-----------------:|-----------------:|------------------:|
| valid_mrr| 0.492        | —                | 0.429            | 0.446             |
| test_mrr | **0.459**    | 0.318            | 0.375            | 0.338             |
| Hits@1   | **0.371**    | 0.271            | —                | —                 |
| Hits@3   | **0.520**    | 0.341            | —                | —                 |
| Hits@10  | **0.605**    | 0.405            | 0.471            | 0.451             |
| MR       | **117**      | 273              | —                | —                 |

Sweep RFAT (descarta subentrenamiento): base(d32,dp.1,lr5e-3)=0.318 → A(d64,L6)=0.353 →
C(d64,L3)=0.363 → **B(d64,L6,lr1e-3)=0.375** (mejor). Sparse sweep: best test_mrr 0.338.

**Orden en test: NBFNet (0.459) > full attention (0.375) > sparse adyacencia (0.338).**

**Análisis (causa mecánica)**
- El gap es **estructural, no de entrenamiento**: el tuning solo movió RFAT 0.318→0.375;
  sigue −0.084 MRR bajo NBFNet (−22% rel.), no cierra con hiperparámetros razonables. Lo
  que más ayudó fue estabilidad de optimización (lr 1e-3), no capacidad.
- El **sparse generaliza PEOR que el denso** pese a parecerse más a NBFNet: valid_mrr más
  alto (0.446 > 0.429) pero test_mrr más bajo (0.338 < 0.375) → ajusta mejor el train
  graph y **transfiere peor** al grafo inductivo disjunto. Eco del fracaso de los expander
  en inductivo: agregación blanda aprendida sobre-ajusta la estructura del train.
- Confirma `transformer_vs_nbfnet.tex`: la atención reimplementa peor el message passing.

**Decisión**
- Dirección "graph transformer (full o sparse) como reemplazo de NBFNet" **descartada**.
- Único margen abierto: inyectar evidencia fuera del horizonte de propagación, no
  recombinar/reponderar lo que el MP ya entrega.
- Config RFAT recomendada si se reusa: `--hidden_dim 64 --drop 0.0 --learning_rate 1e-3`.

---

## HISTÓRICO (proyecto previo, KnowFormer — de `SESSION_LOG.md`, contexto, NO repetir)

Estos resultados son del proyecto anterior (ya descartado por construir sobre KnowFormer),
pero **refutan ideas que podrían re-proponerse**. Conservados como advertencia.

### R12 (2026-06-14) — CPA (Compositional Pivot Attention): NEGATIVO LIMPIO
- **Idea**: Edge Transformer restringido a fila h × k pivotes. 1ª corrida V-RMPNN anclada
  en h → x_{h,v}; top-k pivotes u; 2ª corrida anclada en cada u → x_{u,t}; atención
  out(t)=Σ_u softmax(β(x_{h,u}))·g(x_{h,u}⊙x_{u,t}).
- **Resultado**: TODAS las celdas < baseline (0.4626). cpa_k8=0.4313 (−0.031). Control
  invertido: pivotes **aleatorios** (0.4538) > selección dirigida (0.4313).
- **Causa mecánica decisiva**: las corridas NBFNet ancladas en pivotes distintos son casi
  idénticas entre sí (**cos = 0.992**); el ancla one-hot se lava tras 3 capas de MP. Solo
  el **7.4%** de la energía composicional varía entre pivotes → CPA = un canal promediado,
  no k evidencias distintas. La premisa "NBFNet es la fila h del Edge Transformer" es solo
  débilmente cierta.
- Fix `--cpa_center` (R12c): valid↑ sobre baseline (señal real in-distribution) pero NO
  transfiere a test y dirigido≈aleatorio sigue fallando. **Kill-criterion cumplido.**

### R14 (2026-06-17) — A1-v0 candidate-set attention (sin bias par): NEGATIVO
- Reranker listwise sobre top-K. El reordenamiento **daña −0.08**; empeora 99 queries,
  mejora 2. El readout pointwise ya extrae la mejor señal de las features de nodo.

### R15 (2026-06-17) — A1 con bias por par estructural: señal real pero techo ~baseline
- Bias por par (adyacencia + vecinos comunes) **pasa gate de capacidad** (pair > shuffle):
  señal no-redundante REAL. Pero gate MRR falla: cuello de botella = pick accuracy 44%
  (no calibración). Techo calculado ~baseline aun con reranking perfecto-no-dañino.

**Tema recurrente R9/R12/R14/R15**: toda señal dada a la atención en KGC es redundante con
lo que el readout/MP ya extrae, o no transfiere al grafo inductivo. Mismo hallazgo que el
reinicio 2026-06-21.
