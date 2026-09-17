# Diseño: GT disperso sobre subgrafo podado por A\*Net

**Objetivo**: satisfacer el requisito duro del paper WWW — **escalar a ogbl-wikikg2**
(2.5 M entidades, 32.2 M aristas dirigidas) — sin reimplementar la poda de A\*Net desde cero.

**Estado**: especificación. Nada implementado. Escrito 2026-08-24, antes de que termine el
job 92064; **el número de ese job decide si esto se construye** (ver §7).

---

## 1. Por qué esta arquitectura y no otra

Medido en las entradas 2026-08-22 (b) y 2026-08-23:

| hecho medido | consecuencia |
|---|---|
| Nuestro GT sobre el grafo completo pide **~180 GB** a batch 1 en wikikg2 | no escala; NBFNet tampoco (OOM a batch 1) |
| Sobre el **0.2 % podado** pide **~5-15 GB** a batch 32 | **el muro es propagar sobre todo el grafo, no el modelo** |
| La poda de A\*Net es **sin pérdida en inferencia** (Δ +0.0002 entre 10 % y 100 %) | el subgrafo **contiene la evidencia que importa** |
| A\*Net aplica su score a **estados de NODO**, no a aristas | por eso conserva el kernel fusionado; nuestra atención no puede (2026-08-21 c) |

⇒ La poda es el único mecanismo conocido de escala, y ya está validada. Lo que aportamos
encima es la agregación.

---

## 2. Las tres piezas

```
query (h, r, ?)
      │
      ├─► [1] SELECTOR (A*Net congelado, sin gradiente)
      │        └─► subgrafo por query: unión de nodos/aristas visitados en sus 6 iteraciones
      │
      ├─► [2] GT DISPERSO sobre ese subgrafo  (entrenable)
      │        └─► score para cada nodo VISITADO
      │
      └─► [3] FALLBACK para los nodos NO visitados
               └─► score por defecto  ⇒  arregla el MR, ver §5
```

**Dos etapas, no end-to-end, a propósito.** Se pierde el *weight sharing* que en A\*Net le da
supervisión a la función de prioridad (`AStarNet/reasoning/model.py`, el score comparte pesos
con el predictor). Es una degradación aceptada para que quepa en el plazo; la versión
end-to-end es trabajo posterior, **no** parte de este diseño.

---

## 3. Pieza [1] — extracción del subgrafo

**Dónde engancharse**: `AStarNet/reasoning/model.py:258 select_edges(self, graph, score)`, que
ya devuelve el subgrafo por iteración usando `variadic_topks` con
`ks = node_ratio * num_nodes` y `es = degree_ratio * ks * num_edges / num_nodes` (línea 261-262).

**Qué hacer**: monkey-patch (mismo patrón que `dump_astarnet_ranks.py`, **sin tocar el repo**)
que acumule los índices de nodo y arista seleccionados en las 6 iteraciones y devuelva la
**unión**. El bucle de selección está en `forward` (≈línea 315: `edge_index = self.select_edges(graph, graph.score)`).

**Tamaño esperado por query** (`node_ratio 0.1` en FB15k-237, N=14 541):
1 454 nodos por iteración; la unión de 6 iteraciones da del orden de **3-6 K nodos**
(hay solapamiento fuerte: la selección es acumulativa alrededor de la fuente). **Medir esto
primero** — es el número que determina el costo de todo lo demás.

**En línea, no precomputado.** Guardar subgrafos para 272 115 triples de train es inviable
(~10⁹ enteros). El costo de correr A\*Net sin gradiente por batch es **~0.5 h/época** contra
las 3.7 h/época del GT ⇒ **+14 %**, aceptable.

⚠️ **A validar antes de confiar**: que `select_edges` sea determinista en eval. El config de
wikikg2 usa `break_tie: yes`, el de FB15k-237 no ⇒ puede haber empates resueltos
arbitrariamente. Si no es determinista, el subgrafo cambia entre corridas y ensucia la
comparación.

---

## 4. Pieza [2] — el GT sobre el subgrafo

**Relabeling**: el subgrafo trae un subconjunto de nodos; hay que remapear a `0..n_sub-1`,
correr el GT, y dispersar los scores de vuelta al espacio de entidades completo.

**El modelo no cambia**: es `SparseGraphTransformer` tal cual, con la config vigente
(`--loss bce --dependent --num_layer 6 --hidden_dim 32 --num_heads 8`). Lo único que cambia es
el grafo que recibe. Eso es deliberado: **un solo cambio a la vez**, para poder atribuir.

**Labeling trick**: `x⁰_v = emb(r_q)` si `v == head`. La fuente siempre está en el subgrafo
(es la semilla del selector), así que no hay caso degenerado.

---

## 5. Pieza [3] — nodos no visitados, y por qué es una oportunidad

Los nodos que el selector nunca visita **no tienen score**. Es exactamente la causa del MR
malo de A\*Net, que medimos en los dos datasets:

| dataset | NBFNet | A\*Net | factor |
|---|---:|---:|---:|
| WN18RR | 652.9 | 5853.7 | **9.0×** |
| FB15k-237 | 113.9 | 497.8 | **4.4×** |

⇒ Si asignamos un **score de fallback sensato** a los no visitados (constante aprendida, o un
prior por grado/frecuencia de relación), podríamos **conservar la eficiencia de la poda y
arreglar el MR**. Es un resultado chico, concreto y medible, que sale directamente de una
medición propia — y es la métrica donde un modelo que propaga por todo el grafo debería ganar.

**Ablation obligatorio**: fallback constante vs prior por grado vs `-inf` (el comportamiento de
A\*Net). Solo afecta la cola del ranking ⇒ mueve MR, casi no mueve MRR/H@10.

---

## 6. Entrenamiento

| | |
|---|---|
| selector | **congelado**, sin gradiente (`torch.no_grad()`) |
| GT | entrenable |
| loss | **BCE-k** (`--loss bce --num_negative 32 --adversarial_temperature 0.5`) — regla del proyecto |
| negativos | pueden caer fuera del subgrafo ⇒ reciben el score de fallback. **Verificar que eso no degenere** (si el fallback es constante, todos los negativos fuera del subgrafo dan el mismo gradiente) |
| protocolo | 20 épocas, n≥3 semillas, media ± sd |

⚠️ El punto de los negativos es el riesgo silencioso de esta sección: con `num_negative 32`
sobre 14 541 entidades, la mayoría de los negativos muestreados va a caer **fuera** del
subgrafo. Si todos comparten el mismo score constante, la loss deja de discriminar. **Medir
qué fracción de negativos cae fuera antes de entrenar**; si es alta, muestrear negativos
**dentro** del subgrafo (que además es lo que hace más difícil la tarea, no más fácil).

---

## 7. Experimento de validación — y el gate que decide si esto se construye

**GATE (hoy, job 92064)**: si el GT completo en FB15k-237 **no** es competitivo con NBFNet
(0.4140) ni muestra ventaja por estrato, **este diseño no se construye**: sería A\*Net con
pasos de más y peores números.

**V1 — ¿la poda rompe algo?** GT-sobre-subgrafo vs GT-sobre-grafo-completo, FB15k-237, misma
config, n≥3.
- Predicción falsable: como la poda es sin pérdida para A\*Net, el GT debería quedar **dentro
  del ruido** del GT completo. Si cae mucho, la poda de A\*Net **no** transfiere a otra
  agregación, y eso es en sí un resultado (su selector está entrenado para su propio modelo).

**V2 — ¿escala?** wikikg2, midiendo memoria y tiempo reales antes de entrenar en serio.
Estimación: 0.2 % de 32.2 M = ~64 K aristas ⇒ **~5-15 GB a batch 32**. Verificar con un probe.

**V3 — el MR.** Ablation de fallback (§5).

**Diferido a propósito**: la **compuerta atención/suma condicionada al grado**
(SESSION_NOTES 2026-08-23). Es un cambio **separado** y va después de V1. Meterlo junto con la
poda haría imposible atribuir un fallo — el error que esta bitácora corrigió cuatro veces.

---

## 8. Riesgos, en orden

1. **El GT no tiene ventaja que escalar** ⇒ todo esto es A\*Net más lento. Lo resuelve el gate de hoy.
2. **El selector de A\*Net no transfiere a otra agregación.** Su prioridad se entrenó con weight
   sharing contra *su* predictor. Lo resuelve V1.
3. **Los negativos fuera del subgrafo degeneran la loss** (§6).
4. **El plazo**: ~6 semanas al deadline. Implementación 1-2 semanas con el atajo, V1 ~3 días,
   wikikg2 días, escritura 1-2 semanas. Entra, **sin margen para una segunda arquitectura**.
5. **Contribución**: dos etapas sobre A\*Net invita a "¿qué aportan además de A\*Net?". Las
   respuestas honestas son la agregación por atención (si gana por estrato) y el arreglo del MR.
   Si ninguna se sostiene, no hay paper de arquitectura.

---

# ADENDA (2026-08-26): ESTADO DISPERSO — el requisito real para wikikg2

## Por qué hace falta (medido, job 92413)

El benchmark en YAGO3-10 (N=123 K, E=2.28 M) dio **0.807 it/s y 21.8 GB** a batch 8, y **OOM a
batch 32**. Yendo de FB15k-237 a YAGO el modelo se puso **10.9× más lento** con solo 4.1× de
aristas pero **8.5× de nodos** ⇒ **escala con N, no con E**.

⚠️ **Esto corrige el diagnóstico anterior.** El cuello no es el scoring `O(B·E)` que había
documentado: es el **estado DENSO de nodos**. Extrapolado a wikikg2 (N = 2.5 M, B=8, d=32, L=6):

```
estado (B,N,d) retenido por capa : 2.6 GB × 6 = 15.6 GB
scoring (B,E) por capa           : 1.1 GB × 6 =  6.7 GB
                                   ─────────────────────
                                   22 GB antes de la atención   →  ~19 días/semilla
```

**A\*Net no mantiene estado para los 2.5 M nodos.** `AStarNet/reasoning/model.py:318`:
`subgraph = graph.edge_mask(edge_index, compact=True)` ⇒ la capa corre sobre un subgrafo
**compactado**, y `graph.hidden` es un `VirtualTensor` (base + overrides dispersos), no un denso.

## Confirmaciones que da el código y hay que respetar

- **`short_cut`** (línea 321): `graph.hidden[node_out] = graph.hidden[node_out] + hidden[out_mask]`
  mientras `layer_input = F.sigmoid(subgraph.score) * subgraph.hidden` escala **solo lo que entra
  a la capa**. ⇒ confirma en la fuente el fix del desvanecimiento del 2026-08-25 (c): el score
  pesa el MENSAJE, nunca el estado persistente.
- **Solo se actualizan los nodos con `degree_out > 0`** en el subgrafo; el resto conserva estado.
- El score se recalcula **solo** para esos nodos (línea 325).

## Diseño propuesto: top-M de tamaño FIJO sobre nodos (no VirtualTensor)

Mismo truco que ya funcionó para aristas — evita la maquinaria variádica (`VirtualTensor`,
multi-key sort del Apéndice C) a cambio de algo de cómputo desperdiciado en padding.

**Estructuras** (S = cupo de slots, p.ej. `L_layers × M`):

| | forma | wikikg2 con M=5 000, S=30 000 |
|---|---|---|
| `active` (ids globales, ORDENADOS) | `(B, S)` int64 | 1.9 MB |
| `hidden` (estado por slot) | `(B, S, d)` | **30 MB** (contra 15.6 GB) |
| `n_active` | `(B,)` | — |

**Mapeo global→slot**: `torch.searchsorted` sobre `active` ordenado. **Nunca** materializar un
`(B, N)` — es justamente lo que hay que evitar.

**Grafo en CSR ordenado por fuente** (una vez por grafo): `indptr (N+1)`, `edges (E,)`. Las
salientes de un conjunto de nodos salen con `repeat_interleave` sobre los grados.

**Por capa:**
1. prioridad sobre los **activos** → top-K
2. salientes de esos K vía CSR → top-L por prioridad del **DESTINO** (§3.2)
3. destinos nuevos se **agregan** a `active` si hay cupo
4. atención sobre las L aristas, en espacio de **slots**
5. `hidden[slot] += salida` solo para destinos alcanzados (short_cut)

**Costo esperado en wikikg2**: L = 0.002 × 34.7 M ≈ 69 K aristas y M = 5 000 nodos por query ⇒
**la misma carga por paso que FB15k-237 hoy** ⇒ estimado ~4 h/semilla en vez de 19 días.

## 🔴 TEST DE DES-RIESGO, OBLIGATORIO ANTES DE SEGUIR

**Con `M = N` y `L = E` (todo activo), la versión con estado disperso debe reproducir la densa
EXACTAMENTE** (`max|diff| ~ 1e-6`), cargándole los mismos pesos. Es el patrón de
`diag_equiv.py`, que ya descartó una hipótesis equivocada en 10 minutos.

**Si no pasa, no se avanza.** Los tres fallos previos de esta línea fueron de *semántica de
selección* (ahora verificada contra la fuente); esto es **mecánico y verificable de forma
exacta**, que es una categoría de riesgo distinta y hay que explotarla.
