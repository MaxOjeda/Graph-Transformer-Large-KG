# OBJETIVOS_Y_PLAN_WWW.md — Graph Transformer podado para KGC transductivo (paper ACM WWW)

Fecha: 2026-09-10. Deadline: primera semana de octubre de 2026 (~3.5 semanas).
Complementa `RESULTADOS_PAPER_WWW.md` (tablas) y `SESSION_NOTES.md` (bitácora). Este documento
fija **qué queremos**, **por qué el plan es éste** y **qué no vamos a poder decir**.

---

## 1. Objetivo y restricciones (decisión del usuario, 2026-09-10)

| | |
|---|---|
| **Régimen** | transductivo, únicamente |
| **Datasets** | WN18RR, FB15k-237, YAGO3-10, ogbl-wikikg2 — **los cuatro con el mismo modelo** |
| **Contra quién** | los números **PUBLICADOS** de NBFNet (NeurIPS 2021) y A\*Net (NeurIPS 2023), no nuestras reproducciones |
| **Meta** | por encima de ambos en los cuatro datasets; un punto de MRR es el mínimo, ojalá más |
| **Prohibido** | embeddings de entidad, en cualquier forma (también como PE). Rompen la tesis inductiva |
| **Requisito de forma** | que sea defendible como **Graph Transformer**, no como "A\*Net con atención" ni como una GAT relacional |
| **Escalabilidad** | tiene que correr en ogbl-wikikg2 (2.5 M entidades), donde NBFNet da OOM |

### 1.1 Dos cambios de protocolo que este documento fija

1. **Ya no estamos atados al presupuesto de entrenamiento de A\*Net.** Hasta hoy corríamos 20 épocas
   porque "es lo que usan ellos" y comparábamos contra *nuestra* reproducción a igual presupuesto. Al
   comparar contra los números publicados, el presupuesto de épocas, batch y búsqueda pasa a ser un
   hiperparámetro nuestro, elegido por `valid_mrr` y **declarado**. Nuestro mejor checkpoint cae en la
   época 18 de 20 con el valid todavía subiendo (entrada 2026-08-25); NBFNet satura en la época 4.
   Entrenar hasta converger es legítimo y hoy no lo hacemos.
2. **La columna de referencia es la publicada.** Nuestras seis reproducciones con los repos originales
   caen **todas por debajo** de lo publicado (−0.0003 a −0.0099, ver §3.1): ellos entrenan en 4 GPUs
   (batch efectivo 256) y probablemente seleccionan entre corridas. Reportaremos las dos columnas
   (publicado y reproducido), pero el objetivo se mide contra la publicada. Consecuencia: arrancamos con
   0.005–0.010 de desventaja que no viene del método.

### 1.2 Metas numéricas (MRR full-filtered, test)

| dataset | NBFNet pub. | A\*Net pub. | **meta mínima** | GT podado (n=3) | **GT + bloque global (n=1)** | falta a la meta |
|---|---:|---:|---:|---:|---:|---:|
| FB15k-237 | 0.415 | 0.411 | **≥ 0.425** | 0.4018 ± 0.0013 | **0.4122** | +0.013 |
| WN18RR | 0.551 | 0.549 | **≥ 0.560** | 0.5266 ± 0.0049 | 0.5321 (dentro del ruido) | +0.028 |
| YAGO3-10 | 0.563 | 0.556 | **≥ 0.570** | 0.5427 ± 0.0084 (β=200 %) | **0.5780** ✅ | **superada** |
| ogbl-wikikg2 | OOM | 0.6851 (valid) / 0.6767 (test) | **≥ 0.69** | sin número | en curso (job 93414) | — |

Estado al 2026-09-14: **YAGO3-10 supera la meta** (+0.015 sobre NBFNet publicado, y empata a
presupuesto de entrenamiento igualado); FB15k-237 supera a A\*Net publicado por +0.0012 y queda
−0.0028 bajo NBFNet; WN18RR no se mueve; wikikg2 corriendo. **Todo n=1** ⇒ nada es reportable aún.

Las metas incluyen el punto de margen sobre el mejor publicado. Son ambiciosas: ningún método puramente
estructural ha superado 0.43 en FB15k-237 en cuatro años (§4.3 explica por qué).

---

## 2. Estado actual (todo con poda, `sparse_state`, BCE-32, batch 8, lr 1e-3, 20 épocas, n=3)

| dataset | GT podado | A\*Net **medido** | NBFNet **medido** | Δ vs A\*Net med. | MR GT / A\*Net / NBFNet | mem GT |
|---|---:|---:|---:|---:|---:|---:|
| FB15k-237 (β=100 %) | 0.4018 ± 0.0013 | 0.4084 ± 0.0014 | 0.4147 ± 0.0010 | −0.007 | 260 / 498 / 116 | 0.95 GB (ckpt) |
| WN18RR (β=100 %) | 0.5266 ± 0.0049 | 0.5431 ± 0.0013 | 0.5466 ± 0.0015 | −0.017 | 6 575 / 5 854 / 653 | 4.9 GB |
| YAGO3-10 (β=200 %) | 0.5427 ± 0.0084 | 0.5461 ± 0.0015 (β=100 %) | 0.563 (pub.) | −0.003 | ~1 300 / 2 489 / — | 29 GB |
| YAGO3-10 (β=100 %) | 0.5280 ± 0.0148 | 0.5461 ± 0.0015 | — | −0.018 | 2 359 / 2 489 / — | 22.8 GB |
| ogbl-wikikg2 | corre (11.9 GB, batch 8, `--grad_ckpt`) | 0.6851 (pub.) | OOM | — | — | 11.9 GB |

Semillas nuevas desde la última entrada de la bitácora (jobs 93114/93115/93224–93226/93139/93140):
WN18RR 0.5321 / 0.5231 / 0.5247; FB15k-237 0.4012 / 0.4032 / 0.4009; YAGO β=200 % 0.5515 / 0.5349 / 0.5417.
⚠️ La "ventaja" de YAGO a β=200 % que reportaba H13 **desaparece con la tercera semilla**: queda empatada con
A\*Net a β=200 % (0.5390 ± 0.0109) y por debajo de su β=100 %.

**Lectura honesta**: el modelo es A\*Net con atención por arista. Empata dentro de 1–2 puntos, con MR mejor en
FB15k-237 y YAGO y peor en WN18RR, y escala a wikikg2. No hay ningún resultado por encima de lo publicado.

---

## 3. Justificación del plan (detallada)

### 3.1 Sesgo de reproducción: contra publicados arrancamos por debajo

| brazo | nuestro (n=3) | publicado | Δ |
|---|---:|---:|---:|
| NBFNet FB15k-237 | 0.4147 ± 0.0010 | 0.415 | −0.0003 |
| A\*Net FB15k-237 | 0.4084 ± 0.0014 | 0.411 | −0.0026 |
| NBFNet WN18RR | 0.5466 ± 0.0015 | 0.551 | −0.0044 |
| A\*Net WN18RR | 0.5431 ± 0.0013 | 0.549 | −0.0059 |
| A\*Net YAGO3-10 (β=100 %) | 0.5461 ± 0.0015 | 0.556 | −0.0099 |
| A\*Net YAGO3-10 (β=200 %) | 0.5390 ± 0.0109 | — | — |

Seis de seis por debajo, y el sesgo crece con el tamaño del dataset. Causa probable: 4 GPUs (batch 256) y
selección de corrida. Implicación para la meta: lo que midamos como "+0.01 sobre A\*Net medido" son ~+0.00
contra el publicado. **Hay que ganar por 0.015–0.02 sobre el baseline propio para estar un punto sobre el
publicado.**

### 3.2 Por qué el modelo actual no puede ganar por clase de funciones

- La atención con **soporte = adyacencia** es, según la taxonomía del survey de Graph Transformers
  (Yuan et al., ACM CSUR 2026, §3.3.2), una GNN con máscara de atención: *"al truncar la atención a los
  nodos no conectados, la atención se ve forzada al vecindario local, lo que puede reducir un GT a una GNN"*.
  Un revisor tiene razón al preguntar por qué se llama Graph Transformer.
- Formalmente: cualquier función del **multiconjunto de mensajes entrantes** `{f(h_src, r)}` — suma, PNA o
  atención — es un C-MPNN, acotado por el test `rawl2` (Huang, Romero, Ceylan, Barceló, NeurIPS 2023, Teorema
  5.1). NBFNet, A\*Net y nuestro GT podado están en la **misma clase**. La corrección teórica de la entrada
  2026-09-07 lo confirma: ni siquiera la atención *entre* mensajes entrantes sale de la clase.
- Tres meses de mediciones son coherentes con eso: ninguna variante de agregación (softmax, sigmoid, degree,
  anchor, rel, lowrank, edge dropout, expander) le ganó a NBFNet fuera del ruido en ningún régimen.

⇒ Lo que puede ganar tiene que **salir de la clase**: (a) un readout global condicionado a la query (C-MPNN +
READ captura `erFO3_cnt`, estrictamente más que `rawl2`, Teorema 5.3 del mismo paper), (b) coordenadas
estructurales globales (PPR, espectro), (c) identidad de entidad — **descartada**. Las dos primeras son
exactamente los ingredientes que el survey usa para definir un GT: canal de atención más allá de la
adyacencia y positional encoding.

### 3.3 Hallazgo central: NBFNet, A\*Net y el GT fallan en LAS MISMAS queries

Medido el 2026-09-10 sobre FB15k-237 (40 932 queries de test = 20 466 tripletas × 2 direcciones),
alineando por **nombre de entidad** los volcados de ranks de NBFNet (torchdrug, semilla 1024), A\*Net
(torchdrug, semilla 1024) y nuestro GT podado (semilla 42, job 93224). Las 40 932 queries alinearon.
Una query "falla" si la respuesta queda fuera del top-10 (o sea, el complemento de Hits@10).

**3.3.1 Métricas de las tres corridas alineadas**

| | MRR | Hits@1 | Hits@10 | MR | falla (rank > 10) |
|---|---:|---:|---:|---:|---:|
| NBFNet | 0.4070 | 0.3113 | 0.5940 | 114.3 | 40.6 % |
| A\*Net | 0.4100 | 0.3215 | 0.5833 | 465.5 | 41.7 % |
| GT podado | 0.4012 | 0.3105 | 0.5781 | 289.8 | 42.2 % |

(Publicados: NBFNet MRR 0.415 / H@10 0.599; A\*Net 0.411 / 0.586.) El "40 % que nadie captura" existe:
**4 de cada 10 queries tienen la respuesta fuera del top-10 en los tres modelos.**

**3.3.2 Solapamiento de los conjuntos de fallo**

| | valor |
|---|---:|
| fallan **los tres** a la vez | **34.1 %** del test |
| falla al menos uno | 48.7 % |
| Jaccard(fallos NBFNet, fallos A\*Net) | 0.792 |
| Jaccard(fallos NBFNet, fallos GT) | 0.786 |
| Jaccard(fallos A\*Net, fallos GT) | 0.794 |
| falla GT y acierta NBFNet | 5.8 % |
| falla NBFNet y acierta GT | 4.2 % |
| correlación de Pearson del log-rank, NBFNet–A\*Net | 0.855 |
| correlación de Pearson del log-rank, NBFNet–GT | **0.871** |
| correlación de Pearson del log-rank, A\*Net–GT | 0.852 |

**Lectura**: de los ~41 % de fallos de cada modelo, **34 puntos son compartidos por los tres**. El GT se
parece a NBFNet (Jaccard 0.786, corr 0.871) exactamente tanto como A\*Net se parece a NBFNet (0.792, 0.855),
pese a que A\*Net y NBFNet comparten codebase, kernel y receta, y el GT es otra implementación, otra
agregación (atención por arista, sin normalizar) y otra semilla. **Los tres computan esencialmente la misma
función**, que es la predicción de §3.2 hecha empírica. Ésta es una figura del paper.

**3.3.3 Cotas oráculo (mejor rank por query)**

| combinación | MRR oráculo | H@10 oráculo |
|---|---:|---:|
| NBFNet + A\*Net | 0.4602 | 0.6364 |
| NBFNet + GT | 0.4575 | 0.6358 |
| A\*Net + GT | 0.4590 | 0.6287 |
| los tres | 0.4867 | 0.6586 |

Un oráculo de dos modelos casi idénticos vale +0.05: es lo que compra el **ruido entre corridas**, no una
señal aprendible. Sirve de calibración: cualquier "ensemble" o "reranking" que reporte menos de eso está
explotando varianza, no información. Y aun el oráculo de los tres deja el 34 % de fallos con H@10 en 0.66.

**3.3.4 Dónde está la respuesta cuando los tres fallan**

| | p25 | p50 | p75 | p90 | en top-20 | en top-50 | en top-100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NBFNet | 35 | 98 | 316 | 847 | 11.8 % | 34.5 % | 50.4 % |
| A\*Net | 40 | 137 | 634 | 2 613 | 10.5 % | 30.2 % | 44.1 % |
| GT | 40 | 122 | 472 | 1 515 | 10.6 % | 30.6 % | 45.7 % |

La respuesta mediana está en la posición ~100 y la mitad está fuera del top-100. **No es "casi acierta"**:
un reranker sobre el top-K no la alcanza (esto ya se midió el 2026-08-22 b; ahora se ve por qué).

**3.3.5 Control de ruido: Jaccard entre semillas del MISMO modelo (FB15k-237, 3 semillas por lado)**

La objeción obvia a 3.3.2 es que dos corridas *cualesquiera* comparten fallos porque las queries
difíciles son difíciles para todos. El control es medir cuánto comparten dos semillas del mismo modelo:

| par | Jaccard medio de fallos |
|---|---:|
| GT vs GT (semillas 42/43/44) | **0.854** |
| A\*Net vs A\*Net (semillas 1024/1025/1026) | **0.814** |
| GT vs A\*Net (9 pares) | 0.793 |
| GT vs NBFNet | 0.788 |
| A\*Net vs NBFNet (mismo codebase y kernel) | 0.791 |

| | valor |
|---|---:|
| fallan las **7** corridas (3 GT + 3 A\*Net + NBFNet) | **30.4 %** del test |
| oráculo de las 3 semillas del GT | 0.4569 |
| oráculo de las 3 semillas de A\*Net | 0.4785 |
| oráculo de las 7 corridas | 0.5282 |

**Lectura**: cambiar de arquitectura (GT ↔ A\*Net, Jaccard 0.79) mueve el conjunto de fallos apenas más
que cambiar de semilla (0.81–0.85), y exactamente lo mismo que A\*Net ↔ NBFNet, que comparten codebase.
El oráculo de tres semillas del mismo modelo (+0.05) es del mismo tamaño que el oráculo entre modelos:
la "diversidad" entre arquitecturas es ruido de entrenamiento. Script: `analyze_seeds_fb237.py`.

**3.3.6 Replica en YAGO3-10 (10 000 queries, 100 % alineadas por nombre; A\*Net semilla 1024 β=100 %; GT
tres semillas a β=200 % y tres a β=100 %)**

| | MRR | H@10 |
|---|---:|---:|
| A\*Net s1024 | 0.5458 | 0.695 |
| GT β=200 % s42 / s43 / s44 | 0.5515 / 0.5349 / 0.5417 | 0.695 / 0.693 / 0.695 |
| GT β=100 % s42 / s43 / s44 | 0.5301 / 0.5123 / 0.5417 | 0.691 / 0.649 / 0.697 |

| par | Jaccard medio de fallos |
|---|---:|
| GT β=200 % vs GT β=200 % (semillas) | **0.863** |
| GT β=100 % vs GT β=100 % (semillas; la s43 es un outlier, 0.512) | 0.780 |
| GT β=200 % vs GT β=100 % (el doble de presupuesto de búsqueda) | 0.821 |
| **GT vs A\*Net** (6 pares) | **0.798** |

| | valor |
|---|---:|
| fallan GT (s42) y A\*Net | 27.4 % del test |
| fallan las 7 corridas | 24.3 % |
| falla alguna | 41.3 % |
| oráculo GT + A\*Net | 0.5995 |
| **oráculo de las 3 semillas del GT** | **0.6033** |

En YAGO el oráculo entre modelos es **menor** que el oráculo entre semillas del mismo modelo: no hay
nada que A\*Net capture y el GT no, ni al revés. Duplicar el presupuesto de búsqueda (β) cambia más el
conjunto de fallos (0.821) que cambiar de arquitectura a presupuesto igual… y aun así menos que una
semilla mala. Script: `analyze_overlap_yago.py`.

**3.3.7 Replica en WN18RR (6 268 queries, 100 % alineadas por nombre; 3 semillas por modelo, best
checkpoint por valid; volcados del job 93301)**

| | MRR (s1 / s2 / s3) | H@10 |
|---|---|---:|
| NBFNet | 0.5458 / 0.5457 / 0.5483 | 0.661–0.663 |
| A\*Net | 0.5430 / 0.5445 / 0.5420 | 0.650–0.655 |
| GT podado (β=100 %) | 0.5321 / 0.5231 / 0.5247 | 0.626–0.633 |

| par | Jaccard medio de fallos |
|---|---:|
| GT vs GT (semillas) | **0.883** |
| A\*Net vs A\*Net (semillas) | **0.874** |
| NBFNet vs NBFNet (semillas) | **0.866** |
| A\*Net vs NBFNet | 0.844 |
| GT vs A\*Net | 0.838 |
| GT vs NBFNet | 0.817 |

| | valor |
|---|---:|
| fallan los tres (una semilla por modelo) | **30.6 %** del test |
| falla alguno | 39.8 % |
| oráculo entre modelos | 0.5875 |
| oráculo de las 3 semillas del GT | 0.5676 |

Misma estructura que en FB15k-237 y YAGO: los tres modelos comparten el 31 % del test como fallo, y
cambiar de arquitectura mueve el conjunto de fallos (0.82–0.84) apenas más que cambiar de semilla
(0.87–0.88). Aquí el oráculo entre modelos sí supera al de semillas (0.588 contra 0.568): el GT se
separa un poco más de los otros dos, pero **hacia abajo** (es 0.02 peor en MRR y 0.03 en H@10), o sea la
diferencia son fallos propios del GT, no aciertos que los otros no tengan. Scripts:
`analyze_overlap_wn18rr.py`, `analyze_wn18rr_distance.py`.

**3.3.8 🔴 RE-MEDIDO CON LOS MODELOS NUEVOS (2026-09-14): el bloque global NO cambia en qué
queries se falla.** Rehecho el análisis de 3.3.2-3.3.7 con los brazos del bloque global, que son
0.005 a 0.035 mejores en MRR. Script: `analyze_overlap_v2.py --ds {fb237,wn18rr,yago}`.

| Jaccard medio de fallos | FB15k-237 | YAGO3-10 | WN18RR |
|---|---:|---:|---:|
| **GT viejo vs externos** | 0.792 | 0.816 | 0.829 |
| **GT+global vs externos** | **0.789** | **0.813** | **0.827** |
| externos entre sí (NBFNet vs A\*Net) | 0.799 | — | 0.846 |
| GT viejo entre semillas | 0.854 | 0.863 | 0.883 |
| GT viejo vs GT+global | 0.835 | 0.822 | 0.874 |

**El bloque global no aleja el conjunto de fallos de los baselines: lo mueve 0.002-0.003, o sea
nada.** Y lo mueve MENOS que cambiar de semilla (0.835 contra 0.854 en FB15k-237). Más aún, el
mismo modelo a 20 y a 30 épocas tiene Jaccard 0.959 entre sí, y los tres checkpoints de YAGO
entre 0.891 y 0.924.

⇒ **Corrección al encuadre de §3.2 y §3.6.** La hipótesis era que el canal global saca al modelo
de la clase `rawl2` y que eso se vería como un conjunto de fallos distinto. **No se ve.** Lo que
el bloque hace, medido, es **ajustar mejor la misma función**: mejora el ranking sin cambiar
qué queries son irresolubles. Las dos consecuencias:
1. El techo de §3.4.6 **sigue en pie sin cambios**: el 30-34 % de fallos compartidos es
   incompletitud del KG y el bloque global no lo toca.
2. El argumento del paper **no puede ser "salimos de la clase C-MPNN"**. Lo que sí se sostiene es
   un reclamo empírico más modesto y verificable: *un canal de atención global sobre el conjunto
   podado mejora el ranking de forma consistente en tres datasets, sin cambiar el perfil de
   fallo*. Si se quiere el reclamo de clase hay que demostrarlo con una tarea sintética donde
   `rawl2` sea provablemente insuficiente, no con MRR.

**Perfil estratificado contra el baseline EXTERNO (no contra el nuestro), que es lo que importa:**

| cardinalidad de (h,r) | FB15k-237: GT+global 30ep − NBFNet | YAGO: GT+global 20ep − A\*Net | WN18RR: GT+global − NBFNet |
|---|---:|---:|---:|
| 0 | **−0.0103** (t=−3.9) | +0.0157 | **−0.0226** (t=−6.0) |
| 1-3 | +0.0105 | **+0.0515** (t=5.7) | −0.0102 |
| 4-10 | **+0.0164** (t=5.8) | +0.0243 | −0.0108 |
| 11-30 | +0.0102 | +0.0227 | −0.0016 |
| 31-100 | **+0.0167** (t=5.2) | +0.0266 | −0.0129 |
| 101+ | **−0.0066** (t=−2.4) | **+0.0483** (t=9.2) | — |

⚠️ **Esto corrige lo que dije el 2026-09-12.** Contra NUESTRO baseline la ganancia era máxima en
101+; contra **NBFNet** en FB15k-237 ahí **perdemos**. La explicación es simple: nuestro baseline
es débil en ese estrato y NBFNet es fuerte. Contra el baseline externo el patrón real en
FB15k-237 es **ganamos en el medio (1-100) y perdemos en los dos extremos**. En YAGO ganamos en
los seis estratos. En WN18RR perdemos en casi todos, coherente con que ahí el brazo no supera al
baseline propio.

### 3.4 Qué caracteriza a los fallos compartidos: alta cardinalidad desde un hub

Comparación de los 13 974 fallos comunes contra los 21 003 aciertos comunes (rank ≤ 10 en los tres):

| | fallan los tres | aciertan los tres |
|---|---:|---:|
| grado mediano de la **respuesta** | 70 | 258 |
| grado mediano de la **fuente** | 182 | 86 |
| **respuestas verdaderas de (h, r) ya en train, mediana** | **37** | **4** |
| respuestas con grado ≤ 14 (cola larga) | 4.6 % | 1.3 % |
| queries con cardinalidad ≥ 31 en train | **52.0 %** | 19.4 % |
| queries con cardinalidad ≥ 11 en train | 64.8 % | — |

⚠️ Esto **corrige la lectura de H4** (2026-08-22 b), que ponía el fallo en la cola larga de la respuesta. El
decil de grado más bajo sí tiene el MRR más bajo (0.20), pero son solo el 4.6 % de los fallos. **El fallo
típico es una query de alta cardinalidad lanzada desde un hub**: (h, r) ya tiene decenas o miles de
respuestas conocidas y la de test es una más de un conjunto grande.

**3.4.1 Fallo y MRR por cardinalidad de (h, r) en train** (cuántas respuestas verdaderas ya conoce el modelo,
que el filtrado quita del ranking)

✅ **Re-medido el 2026-09-14 con el modelo NUEVO** (columna `GT+global` = `fb_global_s42_e30`, el
mejor brazo). La conclusión **no cambia**: fallos compartidos 33.8 % contra 34.1 % con el GT viejo,
misma caracterización (mediana 37 respuestas en train, 51.9 % con cardinalidad ≥ 31), y el MRR sigue
cayendo de 0.68 a 0.18 con la cardinalidad **para los tres modelos**.

| # respuestas en train | n | % del test | fallan los tres | MRR NBFNet | MRR A\*Net | **MRR GT+global** | H@10 NBFNet | grado h (med.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 881 | 16.8 % | 15.4 % | 0.684 | 0.685 | 0.674 | 0.813 | 54 |
| 1–3 | 7 136 | 17.4 % | 21.9 % | 0.488 | 0.502 | 0.499 | 0.725 | 62 |
| 4–10 | 7 161 | 17.5 % | 31.4 % | 0.402 | 0.414 | **0.418** | 0.625 | 74 |
| 11–30 | 5 504 | 13.4 % | 32.2 % | 0.390 | 0.400 | 0.400 | 0.606 | 100 |
| 31–100 | 5 461 | 13.3 % | 41.3 % | 0.339 | 0.342 | **0.355** | 0.512 | 190 |
| **101+** | **8 789** | **21.5 %** | **55.9 %** | **0.181** | 0.165 | 0.175 | **0.335** | **1 204** |

Oráculo de los tres con el modelo nuevo: MRR 0.4913 y H@10 0.6625 (antes 0.4867 / 0.6586).

El MRR cae monótonamente de 0.68 a 0.18 con la cardinalidad, **para los tres modelos por igual**, y la
cardinalidad viene de la mano del grado de la fuente (mediana 54 → 1 204). El 21.5 % del test son queries
con más de 100 respuestas conocidas: ahí Hits@10 es 0.33 y ningún modelo se separa.

**3.4.2 Cruce cardinalidad × grado de la fuente (MRR de NBFNet, n entre paréntesis)**

| cardinalidad \ grado de h | 0–27 | 28–99 | 100–249 | 250+ |
|---|---:|---:|---:|---:|
| 0–3 | 0.600 (2 516) | 0.572 (8 382) | 0.600 (2 632) | 0.632 (487) |
| 4–30 | 0.264 (628) | 0.379 (7 009) | 0.447 (3 974) | 0.402 (1 054) |
| 31+ | — (0) | 0.325 (498) | 0.330 (3 311) | **0.210 (10 441)** |

La celda de **cardinalidad ≥ 31 y fuente ≥ 250** contiene **10 441 queries (25.5 % del test) con MRR 0.21**.
Es la celda que explica el techo de FB15k-237 y donde los tres modelos son indistinguibles. Nótese que a
cardinalidad baja el grado de la fuente **no** importa (0.60 en las cuatro columnas): la "inundación desde
hubs" de H5 era en realidad cardinalidad.

**3.4.3 Ejemplo real (FB15k-237, fallo de los tres)**

Query: **(Estados Unidos `/m/09c7w0`, `nationality⁻¹`, ?)** — "¿quién tiene nacionalidad estadounidense?"

| | |
|---|---:|
| grado de la fuente (Estados Unidos) | 15 228 aristas |
| respuestas verdaderas **ya en train** para (EE.UU., nationality⁻¹) | **2 509 personas** |
| respuestas **en test** para la misma (h, r) | **301 personas** |
| respuesta de esta query: persona `/m/09ftwr` | grado 30 |
| rank NBFNet / A\*Net / GT | 19 / 113 / 31 |

Qué pasa mecánicamente. La arista de la query se quita del grafo en train y en test, así que la evidencia
que el modelo tiene para *esta* persona es el conjunto de caminos EE.UU. ⇝ persona **sin** la arista
`nationality`: por ejemplo `EE.UU. →contains→ ciudad →place_of_birth⁻¹→ persona`, o `EE.UU. →film_country⁻¹→
película →actor→ persona`. Esos caminos existen para las 301 respuestas de test, pero existen
**exactamente igual** para miles de personas del grafo cuya arista `nationality` con EE.UU. simplemente
**no está en el KG** (o cuya nacionalidad es otra pero nacieron o trabajaron allí). El modelo tiene que
rankear a `/m/09ftwr` sobre ~12 000 candidatos que no se filtran, y entre ellos hay cientos con evidencia de
caminos idéntica. En el filtrado se quitan las 2 509 conocidas, pero las que faltan en el KG cuentan como
**negativos falsos**. Ninguna agregación de caminos —fija o aprendida— puede distinguir entre dos personas
nacidas en Nueva York de las cuales el KG dice que una es estadounidense y de la otra no dice nada.

Otros fallos de los tres del mismo tipo (todos hubs de tipo/categoría):

| fuente | relación (inversa) | grado h | resp. en train | resp. en test | rank NBFNet / A\*Net / GT |
|---|---|---:|---:|---:|---|
| `/m/02h40lc` (inglés) | `film/language⁻¹` — "¿qué películas están en inglés?" | 5 350 | 1 477 | 186 | 14 / 12 / 7 725 |
| `/m/05zppz` (género masculino) | `person/gender⁻¹` | 5 856 | 2 914 | 333 | 40 / 92 / 313 |
| `/m/02hrh1q` (actor) | `person/profession⁻¹` | 4 544 | 2 264 | 261 | 215 / 9 704 / 341 |
| `/m/04ztj` (matrimonio) | `marriage/type_of_union⁻¹` | 5 984 | 2 586 | 299 | 328 / 532 / 1 102 |
| `/m/08mbj5d` (categoría web) | `webpage/category⁻¹` | 7 224 | 3 612 | 402 | 1 711 / 10 128 / 1 537 |

"¿Qué películas están en inglés?" tiene 1 477 respuestas conocidas y 186 en test; las películas en inglés
cuya arista de idioma falta en FB15k-237 son, con seguridad, muchas más. Estas queries son **preguntas de
tipo**, no de hecho, y el protocolo de ranking filtrado las trata como si tuvieran una respuesta única.

**3.4.4 Replica en YAGO3-10: el mismo eje, más un segundo modo de fallo**

Fallo por cardinalidad de (h, r) en train (GT β=200 % s42 y A\*Net s1024, 10 000 queries):

| # respuestas en train | n | % del test | fallan ambos | MRR A\*Net | MRR GT | H@10 A\*Net | grado h (med.) | grado t (med.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 946 | 9.5 % | 29.3 % | 0.541 | 0.534 | 0.671 | 18 | 218 |
| 1–3 | 1 039 | 10.4 % | 29.4 % | 0.458 | 0.490 | 0.661 | 14 | 46 |
| 4–10 | 3 069 | 30.7 % | 18.2 % | **0.702** | **0.702** | 0.799 | 28 | 272 |
| 11–30 | 1 368 | 13.7 % | 28.1 % | 0.599 | 0.595 | 0.692 | 56 | 56 |
| 31–100 | 1 159 | 11.6 % | 25.5 % | 0.515 | 0.503 | 0.710 | 222 | 32 |
| **101+** | **2 419** | **24.2 %** | **37.9 %** | **0.373** | **0.393** | 0.582 | **1 000** | 30 |

| | fallan ambos | aciertan ambos |
|---|---:|---:|
| n | 2 740 | 6 640 |
| grado mediano de la respuesta | 22 | 84 |
| grado mediano de la fuente | 62 | 42 |
| respuestas de (h, r) en train, mediana | 17 | 9 |
| respuestas con grado ≤ 14 | **23.8 %** | 4.0 % |
| cardinalidad ≥ 31 | **44.3 %** | 31.2 % |
| rank p50 / p90 del fallo | A\*Net 188 / 42 727 · GT 184 / 4 951 | — |

El eje de cardinalidad replica: la cuarta parte del test son queries con más de 100 respuestas conocidas
lanzadas desde hubs de grado ~1 000, y ahí los dos modelos rinden 0.37–0.39 e igual. Pero en YAGO aparece
**además** el modo de fallo de la periferia (que en FB15k-237 era marginal): las queries con 0–3
respuestas conocidas también rinden mal (0.46–0.54) y salen de fuentes de grado 14–18, y el 23.8 % de
los fallos comunes tienen respuesta de grado ≤ 14. YAGO es 8.5× más grande y más disperso (grado medio
17.5 contra 37.4), así que la escasez de evidencia pesa más. Los dos modos son los dos extremos de la
misma variable: cuánta evidencia relacional hay por candidato.

Ejemplos reales de YAGO (fallan ambos):

| fuente | relación (inversa) | grado h | resp. en train | resp. en test | rank A\*Net / GT |
|---|---|---:|---:|---:|---|
| `male` | `hasGender⁻¹` — "¿quién es hombre?" | 122 088 | **61 044** | 275 | 42 727 / 3 994 |
| `United_States` | `isLocatedIn⁻¹` — "¿qué está en EE.UU.?" (resp. `Oahu`) | 24 832 | 10 173 | 52 | 1 278 / 1 001 |
| `female` | `hasGender⁻¹` | 10 238 | 5 119 | 22 | 2 345 / 744 |
| `wordnet_guitar` | `hasMusicalRole⁻¹` (resp. `David_Lee_Roth`) | 3 320 | 1 660 | 8 | 86 / 68 |
| `United_States` | `isCitizenOf⁻¹` (resp. `Robert_Kagan`) | 24 832 | 1 316 | 2 | 1 296 / 647 |
| `United_Kingdom` | `isLocatedIn⁻¹` (resp. `City_of_Wakefield`) | 3 768 | 1 263 | 2 | 41 / 288 |

"¿Quién es hombre?" con 61 044 respuestas conocidas y 275 en test es el caso límite: la query no tiene
respuesta única y todo candidato humano es plausible. El KG tiene `hasGender` para una fracción de las
personas; las demás son negativos falsos en la evaluación.

**3.4.5 WN18RR: el modo de fallo es DISTANCIA y aislamiento, no cardinalidad**

WN18RR tiene 11 relaciones, grado medio ~4 y casi ninguna query de alta cardinalidad, así que el eje de
§3.4.1 no aplica. Lo que separa fallos de aciertos es la distancia h–t en el grafo de train (no dirigida,
GT s42 y A\*Net s1024 y NBFNet s1024 fallando los tres, 1 916 queries):

| distancia h–t en train | fallos comunes | aciertos comunes |
|---|---:|---:|
| ≤ 2 | 3.5 % | **68.3 %** |
| 3 | 9.4 % | 26.2 % |
| 4 | 11.2 % | } 5.5 % |
| 5 | 19.6 % | |
| 6 | 13.2 % | |
| **> 6 o inalcanzable** | **43.0 %** | |

El **87 %** de los fallos comunes está a 4 o más saltos, y el 43 % fuera del horizonte de 6 capas o
desconectado. En FB15k-237 era al revés (99.7 % de los fallos dentro del horizonte, §H4). Dos
consecuencias medidas:

| | valor |
|---|---:|
| queries cuya respuesta **no tiene ninguna arista** en el grafo de train | **212 = 3.4 % del test** |
| MRR de los tres modelos en esas queries | 0.000 / 0.000 / 0.002 |
| **techo de MRR de cualquier método de caminos** en WN18RR (acertando todo lo demás) | **0.966** |
| fallos comunes con la respuesta **sin score** (rank = N, fallback) en GT / A\*Net / NBFNet | 11.7 % / 12.0 % / **0.0 %** |
| rank p50 en fallos comunes, GT / A\*Net / NBFNet | 34 844 / 2 313 / 482 |

La poda deja a la respuesta lejana sin visitar (12 % de los fallos comunes reciben el score de fallback en
el GT y en A\*Net; NBFNet, que propaga por todo el grafo, nunca). Eso es exactamente el MR malo de los
métodos podados en WN18RR (6 575 y 5 854 contra 653) y explica por qué en este dataset nuestro MR es
peor que el de A\*Net y no mejor como en FB15k-237 y YAGO.

Por relación: el 69 % de los fallos comunes son `hypernym` (directa e inversa), que es el 40 % del test y
falla el 55–57 % de las veces; le siguen `has_part` (57 % de fallo) y `synset_domain_topic_of` (61 %).
Son las relaciones jerárquicas de cadena larga.

Fallo por cardinalidad y por grado de la respuesta (referencia; confirma que aquí manda la escasez):

| # respuestas en train | % del test | fallan ambos | MRR A\*Net | MRR GT | grado h / t (med.) |
|---|---:|---:|---:|---:|---|
| **0** | **45.1 %** | 38.4 % | 0.478 | 0.457 | 4 / 14 |
| 1 | 15.5 % | 17.3 % | 0.755 | 0.751 | 8 / 10 |
| 2–3 | 14.4 % | 19.8 % | 0.711 | 0.710 | 14 / 8 |
| 4–10 | 12.9 % | 29.7 % | 0.563 | 0.562 | 24 / 8 |
| 11–30 | 5.1 % | 55.9 % | 0.235 | 0.233 | 44 / 4 |
| 31+ | 7.1 % | 43.4 % | 0.337 | 0.332 | 282 / 4 |

| grado de la respuesta (dirigido, con inversas) | % del test | MRR A\*Net | MRR GT |
|---|---:|---:|---:|
| 0–1 (aislada) | 3.4 % | 0.000 | 0.000 |
| 2–3 | 11.4 % | 0.302 | 0.246 |
| 4–7 | 24.5 % | 0.531 | 0.531 |
| 8–15 | 26.4 % | 0.625 | 0.635 |
| 16+ | 34.3 % | 0.62–0.64 | 0.58–0.61 |

**Consecuencia para el plan en WN18RR**: el bloque global de §3.6 no crea caminos, así que no debería
mover el 43 % de fallos fuera del horizonte; lo que ataca ese modo es (a) no dejar la respuesta sin
score (fallback informado: es donde el MR se pierde), (b) más alcance a igual presupuesto (β o capas), y
(c) es el **único** dataset donde una coordenada estructural global (PPR como PE absoluta) tiene una
razón *medida* para ayudar: 43 % de los fallos están donde el message passing no llega. Lo contrario de
FB15k-237, donde toda señal global era redundante. Y el techo de 0.966 va en el paper: el 3.4 % del test
es imposible para toda la familia, incluidos NBFNet y A\*Net.

**3.4.6 Consecuencia para el paper**

1. El 34 % compartido es en su mayor parte **incompletitud y ambigüedad del KG**, no falta de expresividad.
   Sin información externa (texto, identidad de entidad) no es alcanzable, y las dos están fuera del
   alcance de este paper. Hay que decirlo así, con la tabla 3.4.1, porque explica por qué FB15k-237 lleva
   años en 0.41–0.45 para todos los métodos estructurales.
2. El margen que **sí** existe está en (a) el 14.6 % del test donde los modelos discrepan, (b) el estrato de
   cardinalidad 4–30 (30 % del test, MRR 0.39–0.45, H@10 0.61–0.63), donde la respuesta está cerca y hay
   candidatos que discriminar, y (c) lo que hoy perdemos por poda y por protocolo (§3.5).
3. **Un canal global donde los candidatos se ven entre sí** es el mecanismo dirigido a (b): en una query de
   cardinalidad 20 el problema es ordenar entre candidatos con evidencia de caminos parecida, y con BCE-32
   los candidatos nunca se comparan en el objetivo (la CE de grafo completo, que sí los acopla, valía
   +0.167 en inductivo). Eso es exactamente lo que un readout global aprendido aporta, y es lo que lo hace
   Graph Transformer y no GAT.

### 3.5 Dónde están los puntos: lo que hoy perdemos y es recuperable

⚠️ **Tabla actualizada el 2026-09-14 con lo MEDIDO** (antes eran estimaciones):

| palanca | evidencia | estimado | **medido** |
|---|---|---:|---:|
| **Bloque global (§3.6)** | `fb_global` s42 0.4111 contra 0.4018 del baseline; `yago_global` 0.5632 contra 0.5427, misma época | ~+0.01 | **+0.010 (FB), +0.021 (YAGO), +0.005 (WN, = ruido)** |
| **Épocas hasta converger** | FB: 20 → 30 ép da 0.4111 → 0.4122 y el valid hace plató en la 20. YAGO: 10 → 20 ép da 0.5632 → 0.5780, plató en la 14 | +0.005 a +0.01 | **+0.001 (FB), +0.015 (YAGO)** |
| **Poda (α=10 %) en FB15k-237** | GT sin poda dio 0.4183; con poda 0.4018. Ablación pareada 2026-09-06: poda −0.018, batch/lr −0.017 | +0.015 a +0.02 | sin re-medir con el bloque global |
| **Batch de ellos y lr 5e-3** | memoria con poda 0.95 GB (FB) y 4.9 GB (WN) a batch 8 ⇒ batch 64 entra | +0.00 a +0.02 | **sin medir** (semana 1 saltada por decisión del usuario) |
| **β por dataset** | YAGO 0.5301 → 0.5551 al duplicar L; WN18RR nula (−0.004) | +0.00 a +0.02 | sin re-medir con el bloque global |
| **`--dependent`** | +0.07 en FB15k-237 (0.321 → 0.393) | — | ya en la config |
| **Frontera consciente de la relación** | `fb_relfront` s42 | sin medir | **NULO: +0.0013, pierde 16 001 queries y gana 14 104, MR peor** |
| **Fallback informado por PPR** | `wn_fallback` s42 | sin medir | **NULO: −0.0006, y sube los sin-score de 4.4 % a 14.7 %** |
| **PPR como condición de borde** | `ppr_diag_{wn,fb}` | sin medir | **+0.038 en WN18RR, −0.155 en FB15k-237** (efecto de dataset) |

**Sobre α y β por dataset.** A\*Net los declara como hiperparámetros y los varía **50×** entre datasets
(α = 10 % en FB15k-237/WN18RR/YAGO, 0.2 % en wikikg2; su Tabla 9). Usar α = 100 % en FB15k-237 y WN18RR y
α = 10 % / 0.2 % en YAGO / wikikg2 **es el mismo modelo con otro presupuesto de búsqueda**, exactamente como
ellos. Lo que no es aceptable es esconderlo: se reporta la tabla α × MRR × memoria por dataset, y en los
chicos se muestra también el brazo podado. Esta decisión es del usuario (regla del 2026-09-05).

### 3.6 Por qué el bloque global es "Graph Transformer" y no otra variante de agregación

- **Clase**: C-MPNN + readout global captura `erFO3_cnt`, estrictamente más que `rawl2` (Huang et al.,
  Teorema 5.3). Es el único ingrediente disponible que cambia de clase sin embeddings de entidad.
- **Evidencia empírica externa**: en ese mismo paper el readout global **por suma** empeora en 2 splits de
  FB15k-237 y el readout **específico por relación** (nodos con arista entrante / saliente de tipo q) da
  SOTA en FB15k-237 inductivo y nada en WN18RR. ⇒ el readout tiene que ser **selectivo**. "Readout selectivo
  aprendido sobre un conjunto" es atención global, o sea el bloque Transformer que falta.
- **Diseño**: M ≈ 32 tokens inductores por query (estilo Set Transformer / Perceiver), inicializados desde
  `q_emb`, que atienden sobre los S nodos activos y sobre los que luego atienden los nodos. Costo
  O(S·M·d), lineal en el conjunto podado e independiente de las aristas ⇒ escala igual que la poda (en
  wikikg2 con S = 386 604 son ~0.4 GB por capa, y con `--grad_ckpt` entra; se puede aplicar solo en las
  últimas capas). Salida con **init cero** ⇒ al arrancar es exactamente el modelo actual (convención del
  proyecto: `--dependent`, `RelationalGlobalReadout`). El `RelationalGlobalReadout` que ya existe es el caso
  degenerado M = 2 sin aprendizaje de qué agregar, y sirve de control.
- **Positional encoding**: salto de activación desde el head (gratis: la capa en la que el nodo entró al
  conjunto activo) como PE relativa; bins de PPR como PE absoluta cuando se arregle (§5). Ninguno usa
  identidad de entidad.
- **Predicción falsable, escrita antes de medir**: el bloque global debe mover el estrato de cardinalidad
  4–30 y las queries donde los modelos discrepan; **no** debe mover el estrato 101+ (ahí no hay información
  que recuperar). Si mueve el 101+, es ruido; si no mueve el 4–30, no sirve.

  🔴 **VEREDICTO (2026-09-14): la predicción FALLA en sus dos mitades, y el argumento de clase
  también.** (a) Contra NBFNet en FB15k-237 el bloque gana en 1–100 pero **pierde** en 0 y en 101+
  (§3.3.8); contra A\*Net en YAGO gana en los seis estratos, incluido 101+. O sea el efecto **no
  está localizado** donde el mecanismo propuesto predecía. (b) Más grave para el encuadre: el
  conjunto de queries que fallan **no se mueve** (Jaccard contra los externos 0.789 contra 0.792
  del GT viejo), así que el bloque **no está resolviendo queries que la clase `rawl2` no podía**;
  está ajustando mejor la misma función. ⇒ **El reclamo de §3.6 tal como está escrito no se
  sostiene y hay que reescribirlo**: lo que queda es un resultado empírico (un canal de atención
  global mejora el ranking en 3 datasets, con ganancia máxima en YAGO) sin una explicación
  mecánica establecida. Establecerla pide o bien una tarea sintética donde `rawl2` sea
  provablemente insuficiente, o bien un análisis de a qué atienden los tokens inductores.

---

## 4. Plan con compuertas (2026-09-10 → 2026-10-05)

| semana | qué | compuerta |
|---|---|---|
| **1 (09-10 → 09-16)** | **Piso honesto contra publicados, sin tocar arquitectura.** FB15k-237 y WN18RR: batch 64, lr 5e-3, 40 épocas, n=3, α ∈ {10 %, 100 %}, β ∈ {100 %, 200 %}. Depurar el PPR (con init cero no puede arrancar en valid 0.13: es un bug). Volcar ranks de todos los brazos | Si α = 100 % + convergencia supera lo publicado en FB15k-237 y WN18RR, ya hay dos datasets. Si no, el resto del plan tiene que valer más de lo que hoy falta |
| **2 (09-17 → 09-23)** | **Bloque global con tokens inductores + PE de salto.** Verificar equivalencia bit a bit al init. Medir en FB15k-237 y WN18RR con n=3, **estratificando por cardinalidad de (h, r)** | ≥ +0.01 sobre nuestro propio baseline en FB15k-237, localizado en cardinalidad 4–30. Si no, se reporta como ablación negativa y el paper se sostiene en §3.3–3.5 |
| **2 (paralelo)** | Frontera consciente de la relación (`prioridad_destino + λ·b_rel[r, r_q]`), un flag, un brazo | ≥ +0.005 en FB15k-237 o mejora de MR en WN18RR |
| **2 (paralelo, solo WN18RR)** | Las dos palancas que §3.4.5 justifica: (a) **fallback informado** para los nodos sin visitar (hoy un escalar; el 12 % de los fallos comunes recibe ese score y es donde se pierde el MR) y (b) **PPR como PE absoluta** una vez depurado (§5.7), porque el 43 % de los fallos está fuera del horizonte | MR del GT por debajo del de A\*Net (5 854) y ≥ +0.01 de MRR |
| **3 (09-24 → 09-30)** | YAGO3-10 (n=3, ~15 h/semilla) y wikikg2 (~22 h/semilla a batch 8, protocolo OGB de 500 negativos, `test_node_ratio` 1 %) con la config ganadora | YAGO ≥ 0.57; wikikg2 con número reportable, ojalá ≥ 0.69 |
| **4 (10-01 → 10-05)** | Escritura. Figuras: solapamiento de fallos (§3.3), fallo por cardinalidad (§3.4.1), ablación α/β/bloque global, memoria/tiempo vs escala | — |

### 4.1 Estado de ejecución (2026-09-10)

**Decisión del usuario**: se salta la semana 1 (piso honesto con batch 64 / 40 épocas): si el bloque
global se adopta, ese piso habría que re-medirlo igual con la arquitectura nueva. Los brazos de la
semana 2 corren con la **config exacta de los baselines ya medidos** (batch 8, lr 1e-3, 20 épocas,
β=100 %, `fb_b8lr1` y `wn_b8lr1_b100`, n=3), así que cada contraste es de **una sola variable**.

Implementado en `src/model.py::SparseStateGraphTransformer` (flags en `train.py`, todos con init
cero; `diag_global_equiv.py` verifica forward inicial **bit a bit idéntico** al baseline y gradiente
vivo tras un paso de optimizador):

| flag | qué | brazo |
|---|---|---|
| `--global_tokens 32 --global_from 2 --global_pool topk` | `GlobalInducedBlock`: 32 tokens inductores por query, pool sobre los K expandidos, broadcast a los S activos, desde la capa 2 | `fb_global`, `wn_global` |
| `--hop_pe` | PE de salto: embedding de la capa de activación, sumado al entrar | (dentro de `*_global`) |
| `--rel_frontier 1.0` | `<q_emb, rel_compat[r]>` sumado a la prioridad del destino y al logit de atención | `fb_relfront` |
| `--fallback ppr` | score de no visitados + término por bin de PPR desde el head | `wn_fallback` |
| `--indicator ppr` (ya existía) | al init es un no-op **exacto** (verificado) ⇒ el colapso de 93091/93092 es de entrenamiento, no de forward | `ppr_diag_wn` (3 épocas cortas, onehot vs ppr, misma semilla) |

Cola (`queue_semana2.txt`, sometida por `launch_queue.sh` a medida que la QOS libera slots, 4 a la
vez): smoke test → `wn_global` s42 → `fb_global` s42 → `ppr_diag_wn` → `wn_fallback` s42 →
`fb_relfront` s42 → `ppr_diag_fb` → semillas 43/44 de cada brazo → `wn_ppr` s42/43/44. 17 jobs;
~27 h por job en FB15k-237 y ~11 h en WN18RR ⇒ ~5 días de reloj con 4 slots. Cada job vuelca ranks
(`<brazo>_s<seed>_ranks.pt`) para la evaluación estratificada por cardinalidad y distancia (§3.4),
con `analyze_arm_vs_base.py`.

**Primeras lecturas (2026-09-11, NO son mediciones):**
- Smoke test GPU de todos los flags por la ruta real: rc=0, 1.0 GB, 4.7 it/s.
- `wn_global` s42, primeras validaciones: 0.480 / 0.476 contra 0.463 / 0.478 del baseline en las
  mismas épocas. `fb_global` s42: 0.380 contra 0.359 en la época 0. No diverge; nada más.
- **Diagnóstico del PPR en WN18RR (`ppr_diag_wn`, 3 épocas de 800 pasos, misma semilla): NO
  colapsa.** onehot valid 0.324 / 0.352 / 0.338 (test 0.324); ppr 0.317 / 0.299 / 0.387 (test
  0.362). Más ruidoso pero termina arriba. El colapso de los jobs 93091/93092 (valid 0.41 en WN18RR
  y **0.13** en FB15k-237 tras una época completa) no se reproduce en corto ⇒ o depende de la
  longitud del entrenamiento (deriva del embedding `distance`) o de FB15k-237 (`--dependent`,
  `--remove_one_hop`). `ppr_diag_fb` corre onehot / ppr / ppr sin `--dependent` para separarlo.
  Mientras tanto se encola `wn_ppr` (n=3) porque en WN18RR es el único dataset con razón medida
  para una coordenada global (§3.4.5).

**Primeros resultados completos (2026-09-11, una semilla por brazo; los contrastes son pareados
contra las 3 semillas del baseline, así que el `t` mide la diferencia de ESTAS corridas, no de
los métodos):**

| brazo (s42) | test MRR | baseline (n=3) | Δ pareado | t | H@10 | MR |
|---|---:|---:|---:|---:|---:|---:|
| **`fb_global`** | **0.4111** | 0.4018 ± 0.0013 | **+0.0093** | 11.1 | 0.590 vs 0.582 | **182 vs 260** |
| `wn_global` | 0.5321 | 0.5266 ± 0.0048 | +0.0054 | 3.2 | 0.631 vs 0.630 | 6 517 vs 6 575 |

- **FB15k-237**: el bloque global está por encima del baseline en las 20 validaciones (0.380 →
  0.416 contra 0.359 → 0.405), y el test empata con el **A\*Net publicado (0.411)** con una semilla.
  Falta n=3 y quedan −0.004 contra NBFNet publicado. **La predicción falsable de §3.6 se cumple a
  medias**: la ganancia sí aparece en cardinalidad 4–30 (+0.005/+0.006), pero **también en 101+
  (+0.011, t=5.7)**, que según la predicción no debía moverse. El perfil real es otro: la ganancia
  es mayor en **respuestas de grado bajo** (4–14: +0.024; 15–43: +0.016; 250+: +0.004) y en los
  dos extremos del grado de la fuente (0–27: +0.019; 1000+: +0.017). Lectura: el canal global no
  actúa (solo) acoplando candidatos en queries de cardinalidad media; actúa donde la evidencia
  local es escasa, o sea le da a los nodos con pocos caminos algo con qué compararse. Mecanismo a
  reescribir en §3.6 cuando haya n=3.
- **WN18RR**: +0.005 contra la media del baseline, pero la semilla 42 del baseline dio
  exactamente 0.5321, igual que el brazo ⇒ **dentro del ruido entre semillas (σ 0.005)**. Por
  distancia sí se ve lo predicho: gana a d=3 (+0.014) y d=4 (+0.024), y no toca d≥7 ni los
  inalcanzables (0.000 en ambos), porque no crea caminos. El MR mejora poco (6 517).
- **Diagnóstico del PPR en FB15k-237 (`ppr_diag_fb`, 3 épocas de 800 pasos, misma semilla)**: el
  colapso **se reproduce y NO es `--dependent`**: onehot test 0.297, ppr 0.142, ppr sin
  `--dependent` 0.097. Con el mismo protocolo en WN18RR el ppr **gana** (0.362 vs 0.324). Es un
  efecto de dataset: en FB15k-237 (grado medio 37, hubs de 15 000) miles de nodos recién
  activados comparten el mismo vector de borde y la agregación **sin normalizar** (`sigmoid`) los
  suma en los hubs; en WN18RR (grado ~4) no pasa. Coherente con la observación de A\*Net de que
  PNA no generaliza con grados dinámicos. ⇒ El PPR como condición de borde queda **solo para
  WN18RR** (`wn_ppr`, en cola); para FB/YAGO habría que inyectarlo en el readout o en la prioridad,
  no en el stream de mensajes. No se hace en esta semana.
- Encolado `yago_global` s42 (config `gt_b200` + bloque global) detrás de `fb_global` s44.

**`fb_relfront` s42: NULO (2026-09-12).** Test **0.4031** contra 0.4018 ± 0.0013 del baseline;
pareado +0.0013 (t=1.68) y, lo que decide, **gana 14 104 queries y pierde 16 001** — la media
positiva la arrastran pocas ganancias grandes. **MR empeora** (271 vs 260). Por estrato solo se
mueve el 101+ (+0.005) y el de fuentes de grado ≥1000 (+0.009, t=3.5), y en los dos los conteos
pareados están en contra (3 691/4 695 y 2 146/2 862). ⇒ La frontera consciente de la relación
**no pasa su compuerta** (≥ +0.005 o mejora de MR). No se corren s43/s44. Es la sexta intervención
del proyecto sobre la búsqueda/agregación local que sale nula, y coherente con §3.2: no cambia la
clase de funciones. **El único brazo con señal es el bloque global.**

**`wn_fallback` s42: NULO, y empeora la cobertura (2026-09-12).** Test **0.5260** contra
0.5266 ± 0.0048 del baseline; pareado −0.0006 (t=−0.37) con **1 254 ganadas contra 2 033
perdidas**. El MR mejora poco (6 429 vs 6 575) y a cambio **la fracción de respuestas sin score
sube de 4.4 % a 14.7 %**: el término por bin de PPR no ordena la cola, la desordena — al sumarse
al escalar de fallback, empuja a más respuestas reales por debajo del bloque de no visitados.
⇒ No pasa su compuerta (MR por debajo de los 5 854 de A\*Net y ≥ +0.01 de MRR). No se corren
s43/s44.

⇒ **De los cuatro mecanismos de la semana 2, tres son nulos** (`rel_frontier`, `fallback ppr`, y
el bloque global en WN18RR) **y uno tiene señal** (bloque global en FB15k-237). Los tres nulos son
intervenciones sobre la búsqueda o el score, no sobre la clase de funciones; el que funciona es el
único que sale de `rawl2`. Es exactamente lo que predecía §3.2, ahora con cuatro mediciones más.
Cola recortada a los brazos vivos: `fb_global` s43/s44, `yago_global` s42 y `wn_ppr` s42–s44.

**Decisiones del usuario (2026-09-12):**
- **Prioridad: seguir entrenando `fb_global` s42.** El job 93304 cerró sus 20 épocas con el valid
  aún subiendo, así que se reanudó hasta 30 (job 93330, `fb_global_s42_e30`). Los ranks y el mejor
  checkpoint de las 20 épocas quedan preservados
  (`fb_global_s42_ranks.pt`, `experiments/fb_global_s42_ep19_SAFE.ckpt`).

  **Resultado a 30 épocas: el valid HACE PLATÓ y el brazo supera al A\*Net publicado (n=1).**

  | | 20 épocas | **30 épocas** | baseline (n=3) | *A\*Net pub.* | *NBFNet pub.* |
  |---|---:|---:|---:|---:|---:|
  | test MRR | 0.4111 | **0.4122** | 0.4018 ± 0.0013 | *0.411* | *0.415* |
  | valid (mejor época) | 0.4165 (19/20) | 0.4172 (**27**/30) | 0.4054 | — | — |
  | Hits@10 | 0.5900 | 0.5898 | 0.5815 | *0.586* | *0.599* |
  | MR | 182.4 | **181.2** | 260.1 | *497.8 (medido)* | *114.3 (medido)* |

  Valid por época 20-29: 0.417 / 0.416 / 0.417 / 0.416 / 0.417 ⇒ **plató desde la época 20**; las
  10 épocas extra compran **+0.0011** de test. La convergencia deja de ser una palanca en este
  brazo: el 0.4111 de las 20 épocas ya era esencialmente el número final. Contraste pareado contra
  el baseline: **+0.0104 (t=12.0)**, 15 732 ganadas contra 14 586 perdidas.

  Contra los números **publicados** (§1.1): **+0.0012 sobre A\*Net (0.411)** y **−0.0028 bajo
  NBFNet (0.415)**. Primer número del proyecto por encima de un publicado en transductivo, pero
  **n=1** ⇒ no reportable hasta cerrar semillas (s43/s44 canceladas en la época 2 y 1 por decisión
  del usuario para liberar GPU a YAGO; `last.ckpt` conservado en ambas).

  Perfil por cardinalidad a 30 épocas (mismo patrón que a 20, más marcado en los extremos):
  0 respuestas +0.0111 · 1-3 +0.0091 · 4-10 +0.0064 · 11-30 +0.0054 · 31-100 +0.0139 ·
  **101+ +0.0150 (t=7.5)**. ⚠️ Confirma que la predicción de §3.6 estaba mal: el estrato que más
  se mueve es el de **alta cardinalidad**, que se había predicho inmóvil. El canal global no
  desambigua candidatos intercambiables por acoplamiento local; hace algo que ayuda **justo donde
  la evidencia de caminos es más pobre o más diluida**. Hay que reescribir el mecanismo con n=3 y
  con el análisis de a qué atienden los tokens.
- 🎯 **`yago_global` s42: IGUALA A NBFNet PUBLICADO (2026-09-13, n=1).** Config `gt_b200`
  (β=200 %, 0.40 pasadas) + bloque global + `--grad_ckpt`.

  | | **`yago_global` s42** | baseline `gt_b200` (n=3) | *A\*Net pub.* | *NBFNet pub.* | *ULTRA pub.* |
  |---|---:|---:|---:|---:|---:|
  | test MRR | **0.5632** | 0.5427 ± 0.0084 | *0.556* | *0.563* | *0.557* |
  | Hits@10 | **0.7042** | 0.6946 | *0.707* | *0.708* | *0.71* |
  | MR | **991** | 1 274 | — | — | — |

  Contraste pareado contra las 3 semillas del baseline: **+0.0205 (t=10.2)**, 3 450 ganadas contra
  2 683 perdidas. Contra publicados: **+0.0072 sobre A\*Net, +0.0002 sobre NBFNet** ⇒ primer brazo
  del proyecto **a la altura del mejor número publicado** en un dataset. ⚠️ n=1.

  Perfil por cardinalidad: **el efecto crece con la cardinalidad hasta 31-100 (+0.0435, t=6.9)** y
  sigue alto en 101+ (+0.0243). Por grado de la fuente, máximo en 100-999 (+0.039/+0.041). Mismo
  patrón cualitativo que FB15k-237 ⇒ **el hallazgo replica en dos datasets**: el canal global rinde
  donde la evidencia local está diluida por muchas respuestas o muchos vecinos, no en la
  cardinalidad media como predecía §3.6.

  Coste: **7.3 GB** (contra 29 GB del baseline sin `--grad_ckpt`) y 2.7 it/s ⇒ hay margen para
  subir el batch de 8, que es la desviación de protocolo §6.1.

  El valid cerró **en su máximo en la última época** (0.5718 en la 9 de 10) ⇒ lanzada la
  continuación a 20 épocas-PL (job 93388, `yago_global_s42_e20`), mismo criterio que el usuario
  aplicó a `fb_global`.

  🏆 **La época 11 SUPERA A LOS TRES PUBLICADOS (2026-09-13, job 93396, eval-only, n=1).** En la
  continuación, la época 11 (Lightning `epoch=10`) dio el mejor valid de YAGO del proyecto
  (**0.57554**); evaluada en test:

  | | **`yago_global` ép. 11** | ép. 10 | baseline (n=3) | *A\*Net pub.* | *NBFNet pub.* | *ULTRA pub.* |
  |---|---:|---:|---:|---:|---:|---:|
  | test MRR | **0.5693** | 0.5632 | 0.5427 ± 0.0084 | *0.556* | *0.563* | *0.557* |
  | Hits@10 | **0.7138** | 0.7042 | 0.6946 | *0.707* | *0.708* | *0.71* |
  | MR | **864** | 991 | 1 274 | — | — | — |

  **+0.0063 sobre NBFNet, +0.0133 sobre A\*Net, +0.0123 sobre ULTRA**, y **Hits@10 por encima de
  los tres**. Contra el baseline propio: **+0.0265 pareado (t=13.5)**, 3 729 ganadas contra 2 410.
  ⚠️ **n=1**, y el checkpoint se eligió por `valid_mrr`, que es el criterio correcto, pero sobre
  una trayectoria de 20 épocas ⇒ hay selección; con n=3 hay que reportar la media de los mejores
  por semilla, no el mejor de los tres.

  ✅ **Y la desviación de presupuesto es MENOR de lo que anoté**: 11 épocas-PL de 5 395 pasos ×
  batch 8 = **0.44 pasadas**, contra las 0.40 de A\*Net ⇒ **1.1× su presupuesto**, no 2×. El
  entrenamiento completo a 20 épocas-PL sí sería 0.80 y hay que declararlo si el mejor checkpoint
  cae más tarde.

  El perfil por cardinalidad se mantiene y se hace más parejo: 1-3 +0.026 · 4-10 +0.025 (t=8.2) ·
  11-30 +0.031 · **31-100 +0.038** · 101+ +0.027. Ganancia en todos los estratos.
  Checkpoint preservado: `experiments/yago_global_s42_ep10_v0576_SAFE.ckpt`;
  ranks en `yago_global_s42_ep10_ranks.pt`.

  **Continuación a 20 épocas-PL terminada (job 93388): test 0.5780.** Mejor checkpoint en la
  época 17 (`valid_mrr` 0.58347); valid de la 10 a la 19: 0.576 / 0.579 / 0.581 / 0.582 / 0.581 /
  0.583 / 0.582 / 0.583 / 0.5835 ⇒ **plató desde la época 14**. Curva presupuesto → resultado,
  con el presupuesto expresado en pasadas sobre el train set (A\*Net usa **0.40**):

  | checkpoint | pasadas | × A\*Net | test MRR | vs NBFNet pub. (0.563) |
  |---|---:|---:|---:|---:|
  | época 10 | 0.40 | **1.0×** | 0.5632 | +0.0002 |
  | época 11 | 0.44 | 1.1× | 0.5693 | +0.0063 |
  | **época 17** | **0.68** | **1.7×** | **0.5780** | **+0.0150** |

  ⇒ **A presupuesto EXACTAMENTE igualado (0.40 pasadas) ya empatamos a NBFNet publicado**, y la
  ventaja crece con el entrenamiento hasta +0.015 a 1.7× su presupuesto. Las dos lecturas se
  reportan: la de presupuesto igualado es la defendible sin asteriscos, la de 0.68 pasadas es el
  mejor número y va declarada. Checkpoint: `experiments/yago_global_s42_ep16_v0583_SAFE.ckpt`.
  ⚠️ Sigue siendo **n=1** y el checkpoint se elige por `valid_mrr` sobre 20 épocas.
- **Canceladas `wn_global` s43 y s44.** El brazo daba +0.005 sobre la media del baseline pero la
  semilla 42 del baseline vale exactamente lo mismo que el brazo ⇒ dentro del ruido. Queda
  reanudable el checkpoint de la época 12 de la s43.

Reglas que siguen vigentes: `--loss bce` en todo; n ≥ 3 en todo lo que se reporte; ninguna propiedad medida
en un dataset se hereda a otro; predicción escrita antes de correr; `PERFILAR, NO ESTIMAR` para costos.

**Qué NO se vuelve a intentar** (cerrado con evidencia): expander graphs (tres refutaciones, la última a
presupuesto igualado), PE por nodo dentro de la atención (RWSE, LapPE, source_rw), `--rel_param lowrank`,
reranking sobre el top-K de A\*Net, variantes de la normalización de la agregación local, embeddings de
entidad.

---

## 5. Limitaciones (para el paper y para nosotros)

1. **El techo compartido.** El 34 % de fallos comunes es, en su mayor parte, incompletitud del KG y
   ambigüedad de queries de tipo. Ningún método sin información externa lo va a recuperar; no prometerlo.
   Superar "por bastante" a NBFNet en FB15k-237 y WN18RR no lo ha hecho ningún método estructural.
2. **El análisis de solapamiento está replicado en los tres datasets con control de semillas** (FB15k-237:
   3 GT + 3 A\*Net + 1 NBFNet; YAGO: 6 GT + 1 A\*Net, sin NBFNet porque nunca se corrió; WN18RR: 3 + 3 + 3).
   Lo que falta: NBFNet en YAGO (solo el número publicado), y en YAGO A\*Net es una semilla. Los modos de
   fallo son **distintos por dataset** (cardinalidad en FB15k-237, cardinalidad + periferia en YAGO,
   distancia y aislamiento en WN18RR): la regla del proyecto de no heredar propiedades entre datasets vale
   también para esto, y el paper tiene que presentar los tres perfiles, no uno.
3. **El argumento de expresividad es sobre distinguibilidad, no sobre generalización.** El propio survey
   (§6.1) advierte que "más expresivo" no predice desempeño empírico. El bloque global puede ser más
   expresivo y no ganar nada; por eso el plan lo trata como hipótesis con compuerta, no como resultado.
4. **Señales globales han sido nulas en WN18RR** en todo el proyecto (edge dropout 0.000, expander −0.006,
   β +0.003, readout global de Huang et al. sin efecto). Es probable que la meta de WN18RR haya que
   cumplirla solo con protocolo y α, no con el bloque global.
5. **Costo estructural de la atención por arista**: no podemos usar el kernel fusionado `rspmm` porque
   nuestros pesos de arista requieren gradiente (`AStarNet/reasoning/layer.py:132`). Vale ~3× en tiempo y
   mucho más en memoria (31.6 GB contra 13 de A\*Net en YAGO a batch 8). `--grad_ckpt` lo mitiga; un kernel
   con backward por arista (Yandex, ICML 2026) exige torch ≥ 2.4.1 y queda para después del envío.
6. **wikikg2 usa otra métrica** (ranking contra 500 negativos fijos de OGB, split temporal) y hoy no tiene
   número reportable: la receta completa de A\*Net para ese dataset (PPR, 1 M negativos, edge dropout,
   `break_tie`) fue peor que la simple, con un bug sospechado en el dropout sin reescalar. Es el dataset
   con más riesgo de quedar como "corre, sin ganar".
7. **PPR está roto en nuestra implementación** (valid 0.13 en FB15k-237 y 0.41 en WN18RR con init cero, que
   debería ser un no-op al arranque). Hasta arreglarlo, no es evidencia ni a favor ni en contra del PPR.
8. **α y β por dataset son un hiperparámetro declarado, no una trampa, pero hay que reportarlos completos**
   (tabla α × MRR × memoria) y mostrar el brazo podado también en los datasets chicos. Si el revisor lo
   objeta, el argumento es la Tabla 9 de A\*Net.
9. **Nosotros tuneamos y ellos no.** Batch, lr, épocas y β se eligen por `valid_mrr` de nuestro lado; los
   baselines corren con su config publicada. Va en la tabla de desviaciones, no en una nota al pie.
10. **MR en WN18RR es peor que A\*Net** (6 575 contra 5 854) aunque en FB15k-237 y YAGO es 2× mejor. La
    ventaja de MR no es una propiedad general del método.
11. **La semilla 1025 de NBFNet FB15k-237** se recuperó de un checkpoint de la época 8; el protocolo de
    medición sigue siendo no determinista (`index_add_`), por eso todo va con n ≥ 3.

---

## 6. Artefactos de este documento

- Scripts de análisis (en el repo, CPU, env `attention`, ~1 min cada uno): `analyze_overlap_fb237.py`
  (§3.3.1–3.3.4), `analyze_seeds_fb237.py` (§3.3.5), `analyze_overlap_yago.py` (§3.3.6 y §3.4.4),
  `analyze_overlap_wn18rr.py` (§3.3.7), `analyze_wn18rr_distance.py` (§3.4.5), `analyze_cardinality_fb237.py`
  (§3.4.1–3.4.3).
- Volcados: `astarnet_ranks_bias_s{1024,1025,1026}_test.pkl` (A\*Net FB, con `htr`),
  `astarnet_yago_ranks_test.pkl`, `gt_b{100,200}_s{42,43,44}_ranks.pt` (YAGO), `wn_b8lr1_b100*_ranks.pt`
  (GT WN18RR), `astarnet_wn_ranks_s{1024,1025,1026}_test.pkl` y `nbfnet_wn_ranks_s{1024,1025,1026}_test.pkl`
  (job 93301, `sbatch_dump_wn_ranks.sh`, 6 min de GPU).
- Vocabularios de torchdrug (orden de aparición train → valid → test, `load_tsvs`):
  `~/datasets/knowledge_graphs/torchdrug/{yago310,wn18rr}_{train,valid,test}.txt` y
  `~/datasets/knowledge_graphs/FB15k-237/FB15k-237/raw/`.
- Volcados usados: `nbfnet_ranks_fb237_s1024_test.pkl`, `astarnet_ranks_fb237_s1024_test.pkl`,
  `fb_b8lr1_ranks.pt`; vocabularios `data/fb15k-237/{entities,relations}.txt` y los raw de torchdrug en
  `~/datasets/knowledge_graphs/FB15k-237/FB15k-237/raw/` (vocabulario por orden de aparición
  train → valid → test, `torchdrug/data/dataset.py::load_tsvs`).
- Semillas nuevas (no registradas aún en `SESSION_NOTES.md`): jobs 93114, 93115, 93224–93226, 93139, 93140.
