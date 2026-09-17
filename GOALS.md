# GOALS.md — Attention (Graph Transformer para KGC)

Objetivos del proyecto. Estable; cambia poco. Para resultados de sesión ver
`SESSION_NOTES.md`; para el briefing operacional ver `CLAUDE.md`.

---

## 🚩 PRIORIDAD VIGENTE (desde la defensa de candidatura, 2026-08-18)

La defensa de candidatura se rindió el **2026-08-18** y **salió aprobada**. Ahí se acordó la
prioridad que manda de aquí en adelante:

| | |
|---|---|
| **Régimen prioritario** | **TRANSDUCTIVO** |
| **Entregable** | publicación enviada a **ACM WWW** |
| **Deadline** | **primera semana de octubre de 2026** |
| **Inductivo** | **pospuesto — se retoma DESPUÉS del envío** |

Todo lo que sigue en este archivo describe la **etapa anterior** (el sanity check inductivo).
Sigue siendo el registro válido de lo medido y de las restricciones de diseño, pero **la
pregunta central de abajo ya no fija la prioridad de cómputo**. Evaluar cada experimento
propuesto contra el objetivo transductivo y el deadline de octubre.

**Bloqueador #1**: no existe baseline **NBFNet transductivo** propio en ningún dataset ⇒ el
mejor número transductivo del proyecto no es interpretable. Ver la entrada **2026-08-18** de
`SESSION_NOTES.md` para el estado y la cola de trabajo.

---

## Objetivo general

Determinar si una arquitectura basada en **atención (Graph Transformer)** puede
**superar a NBFNet** en knowledge graph completion (KGC), con énfasis en el régimen
**inductivo** (grafo de test con entidades disjuntas). Una sola arquitectura debe servir
transductivo e inductivo; solo cambian hiperparámetros (requisito del manuscrito de tesis).

---

## Pregunta central de la etapa actual (reinicio 2026-06-21)

> Un **Graph Transformer con atención FULL (densa, all-pairs O(N²))** —el techo de
> expresividad de cualquier atención sparse— ¿supera a **NBFNet** en **FB15k-237
> inductivo v1**?

Lógica del sanity check: si la versión densa (el techo) **no** gana, una versión sparse
(que es una restricción de la densa) no tiene caso. Decide si la dirección "Graph
Transformer para KGC" sigue viva antes de invertir en variantes sparse/lineales.

**Estado: RESPONDIDA — NO, y con barra de error (2026-08-07).**

| FB15k-237 ind v1, 50 ép | test_mrr |
|---|---:|
| NBFNet (n=6) | **0.458 ± 0.004** |
| Sparse + edge_drop 0.2 (n=6) | 0.397 ± 0.019 |
| **Full attention (n=3)** | **0.3764 ± 0.0021** |

Brecha full vs NBFNet: **+0.0816 ± 0.0020 (t=39.9)**. La respuesta original (2026-06-21) era
n=1 a 20 épocas; la re-medición del 2026-08-07 con el protocolo vigente la confirma sin
cambiarla (0.375 → 0.3764). **La brecha del full (0.082) es mayor que la del sparse (0.061)**:
el techo de expresividad rinde peor que su propia restricción. Ver `SESSION_NOTES.md`
(2026-06-21, 2026-08-05, 2026-08-07 b).

---

## Restricciones de diseño (no negociables)

- **Inductivo puro**: sin embeddings de entidad, sin positional encoding de nodo. Lo
  único compartido train/test son embeddings de **relación**.
- **Labeling trick** (NBFNet): anclar la propagación a la fuente y condicionar a la
  relación de la query.
- **Loss = CE de grafo completo** (no BCE-k negative sampling).
- **Model selection por `valid_mrr`** (val = train graph); test sobre el grafo inductivo
  disjunto. Reportar siempre valid_mrr y test_mrr + hits@1/3/10, mr.
- La señal de no-redundancia, si la hay, debe venir de **composición/evidencia fuera del
  horizonte de propagación del message passing**, no de recombinar/reponderar lo que el
  MP ya entrega.

---

## Criterio de éxito

- **Éxito**: una arquitectura de atención que iguale o supere el test_mrr de NBFNet
  (**0.458 ± 0.004**, n=6 a 50 ép en FB15k-237 ind v1) **sin** romper inductividad y
  **transfiriendo** al grafo disjunto (no solo subir valid). Con σ de 0.002–0.027 según el
  modelo, "igualar" significa **un contraste con error estándar sobre n≥3 semillas**, no un
  número suelto.
- **Fracaso de una línea**: test_mrr < NBFNet de forma estructural (no cierra con tuning
  razonable), o valid↑ / test↓ (overfit estructural). En ese caso: parar, diagnosticar
  causa mecánica, registrar en `SESSION_NOTES.md` y en la lista negra de `CLAUDE.md`.

## Protocolo de medición (obligatorio desde 2026-08-05)

- **≥50 épocas** y **n≥3 semillas** (n≥6 para contrastes finos), reportando media ± sd.
- Comparar brazos con distinto número de épocas **invierte conclusiones** — ya pasó.
- Una diferencia de 0.01–0.05 con n=1 es ruido en el sparse (σ≈0.02–0.05); en el full y en
  NBFNet σ≈0.002–0.004. Aplicar la advertencia **fila por fila, no por familia**.

---

## Marco teórico

`transformer_vs_nbfnet.tex`:
- **Caso (a)** transformer ciego a la estructura (atención sobre embeddings de nodo sin
  aristas): rompe la inductividad. Inviable → NO hacer.
- **Caso (b)** graph transformer con labeling trick + adyacencia relacional: puede igualar
  a NBFNet en principio, pero "reimplementa message passing con agregación aprendida".
  Este proyecto mide empíricamente si esa flexibilidad extra mejora o no.

---

## Fuera de alcance (refutado — ver lista negra en `CLAUDE.md`)

1. Construir sobre KnowFormer (V-RMPNN/QK-RMPNN/RSPMM) — atención decorativa.
2. Expander graphs en inductivo — metían ruido (Exphormer-Max).
3. Embeddings de entidad / PE de nodo — rompen inductividad.
4. Atención ciega a la estructura — caso (a), inviable.
5. BCE-k negative sampling como loss principal.
6. Leer `h` acumulada en K/V de un stream separado — causa del overfit previo.
7. **PE estructural por nodo en la atención** (RWSE local, LapPE global, source_rw
   query-conditioned) — ninguno da mejora robusta ni supera a NBFNet; el bump de LapPE en v1
   no replicó en v2. Ver lista negra #7 de `CLAUDE.md` para la salvedad de medición.
8. **Expander graphs también en TRANSDUCTIVO** — Δ +0.0008 a config y batch idénticos, a
   ~1.67× de costo. Además el tipado de la arista falsa **daña más cuanto más "real" es la
   relación prestada** (ULTRA: −0.051 FB237 / −0.030 WN18RR). Línea cerrada en ambos regímenes.
