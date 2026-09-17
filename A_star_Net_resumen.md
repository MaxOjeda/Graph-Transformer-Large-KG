# A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs

**Autores:** Zhaocheng Zhu, Xinyu Yuan, Mikhail Galkin, Sophie Xhonneux, Ming Zhang, Maxime Gazeau, Jian Tang
**Venue:** NeurIPS 2023 · arXiv:2206.04798v5

---

## 1. Idea central y motivación

El razonamiento sobre knowledge graphs (KG) para la tarea de *knowledge graph completion* (predecir tripletas faltantes `(u, q, ?)` o `(?, q, u)`) se ha resuelto históricamente con dos familias de métodos:

- **Métodos de embeddings** (TransE, ComplEx, RotatE, etc.): aprenden un vector por entidad y por relación. Son escalables (se paralelizan bien, incluso a nivel multi-GPU), pero **no generalizan a entidades no vistas** (setting inductivo), porque el embedding está atado a la identidad de la entidad.
- **Métodos basados en caminos (path-based)**: predicen la relación entre dos entidades a partir de los caminos (secuencias de tripletas) que las conectan. Como la semántica de un camino depende solo de las *relaciones* involucradas (no de las entidades concretas), estos métodos **sí generalizan al setting inductivo**. El problema es que el número de caminos crece exponencialmente con su longitud, lo que los hace intratables en KGs grandes.

Dentro de los métodos basados en caminos, trabajos recientes (NeuralLP, DRUM, NBFNet, RED-GNN) usan una variante del **algoritmo de Bellman-Ford** para evitar enumerar caminos explícitamente: propagan representaciones nodo a nodo de forma iterativa con complejidad polinomial. Sin embargo, en cada iteración Bellman-Ford **visita todos los nodos y todas las aristas** del grafo (`|V|` y `|E|`), lo cual sigue siendo inviable en KGs de millones de entidades (p. ej. `ogbl-wikikg2`, con ~2.5M entidades y ~16M tripletas), donde NBFNet directamente sufre *out-of-memory* (OOM).

**Propuesta:** A*Net traslada la idea del **algoritmo A\*** (usado en búsqueda de caminos más cortos) al razonamiento sobre KGs. En lugar de propagar por todos los nodos como Bellman-Ford, en cada iteración A*Net **aprende una función de prioridad** que selecciona solo un subconjunto de `K` nodos y `L` aristas "importantes" por los cuales propagar, reduciendo drásticamente el costo de tiempo y memoria tanto en entrenamiento como en inferencia, sin resignar generalización inductiva.

---

## 2. Marco teórico: de caminos a Bellman-Ford a A*

### 2.1 Formulación de path-based reasoning

Dada una consulta `(u, q, ?)`, la representación de una tripleta candidata `(u, q, v)` se construye agregando la representación de **todos los caminos** `P` entre `u` y `v`:

```
h_q(u, v) = ⊕_{P ∈ P_{u→v}} h_q(P) = ⊕_{P ∈ P_{u→v}} ⊗_{(x,r,y)∈P} w_q(x, r, y)
```

- `⊕` (bigote: *L*): función de agregación permutation-invariant sobre caminos (p. ej. suma o max).
- `⊗` (bigote: *N*): función de agregación sobre aristas dentro de un camino, permutation-sensitive (p. ej. multiplicación de matrices).
- `w_q(x, r, y)`: representación de la tripleta `(x, r, y)` condicionada a la relación de consulta `q`, **independiente de las entidades** `x`, `y` — esto es lo que habilita la generalización inductiva.

El problema: calcular esta suma sobre el conjunto completo de caminos `P_{u→v}` es intratable (crece exponencial con la longitud).

### 2.2 Solución vía Bellman-Ford

En vez de enumerar caminos, se propaga iterativamente representaciones "de `t-1` saltos" a "de `t` saltos":

```
h_q^(0)(u, v) ← 1_q(u = v)                                    (condición de frontera)
h_q^(t)(u, v) ← h_q^(0)(u, v) ⊕  ⊕_{(x,r,v)∈E(v)} h_q^(t-1)(u, x) ⊗ w_q(x, r, v)
```

Esto tiene complejidad polinomial, pero cada iteración sigue necesitando recorrer `|V|` nodos y `|E|` aristas para calcular `h_q^(t)(u, v)` para todo `v`.

### 2.3 Caminos importantes

La observación clave de los autores: de todos los caminos entre `u` y `v`, solo un subconjunto pequeño `P_{u→v|q}` (los "caminos importantes" para la relación de consulta `q`) contribuye de forma no despreciable a `h_q(u,v)`:

```
h_q(u, v) = ⊕_{P∈P_{u→v}} h_q(P) ≈ ⊕_{P∈P_{u→v|q}} h_q(P)
```

Ejemplo del paper: para la consulta `Mother(a, ?)`, un camino como `a --Friend--> d --Mother--> e --Friend--> f` no aporta evidencia (usa relaciones irrelevantes), mientras que `a --Father--> b --Wife--> f` o `a <--Brother-- c --Mother--> f` sí son caminos de parentesco relevantes.

Los caminos en `P_{u→v}` forman una estructura de árbol, y se asume que **un camino no es importante si algún prefijo suyo tampoco lo es** — esto permite construir el conjunto de caminos importantes de forma iterativa mediante una función de selección de caminos `m_q`. Bajo el supuesto adicional de que caminos con igual longitud e igual nodo final pueden fusionarse (para evitar la explosión combinatoria del árbol), la selección de caminos se aproxima por una **selección de nodos** `n_uq^(t): 2^V → 2^V`, que en cada paso elige qué nodos "sobreviven" para seguir expandiéndose.

### 2.4 La iteración A*

Sustituyendo la selección de caminos por la selección de nodos, la recursión de Bellman-Ford se convierte en la **iteración A\***:

```
h_q^(t)(u, v) ← h_q^(0)(u, v) ⊕  ⊕_{x∈n_uq^(t-1)(V), (x,r,v)∈E(v)} h_q^(t-1)(u, x) ⊗ w_q(x, r, v)
```

La única diferencia con Bellman-Ford es que la suma ya no corre sobre *todos* los vecinos, sino solo sobre los nodos seleccionados por `n_uq^(t-1)(V)`. Los autores demuestran formalmente (con teoremas en el apéndice, que aquí se omiten) que si `n_uq^(t)` selecciona correctamente los nodos "importantes", esta iteración converge al mismo resultado que agregar sobre los caminos importantes.

El paralelismo con el A* clásico de búsqueda de caminos cortos es directo:
- A* clásico: prioridad `s(x) = d(u,x) ⊗ g(x,v)`, donde `d(u,x)` es la distancia actual recorrida y `g(x,v)` es una heurística de la distancia restante hasta el objetivo `v`.
- A*Net: no se conoce `v` de antemano (es justamente lo que se busca predecir), así que la heurística no puede depender de `v` directamente.

---

## 3. Metodología de A*Net (implementación)

### 3.1 Función de prioridad neural

Como no existe un oráculo (heurística manual) buena para KGs — los experimentos muestran que heurísticas hechas a mano (PageRank personalizado, grado del nodo, selección aleatoria) **dañan fuertemente el desempeño** — los autores diseñan una **función de prioridad neuronal**, entrenada end-to-end junto con la tarea de razonamiento.

Idea de reparametrización: en A* clásico, el nodo objetivo `v` se puede expresar como "nodo fuente + desplazamiento". En A*Net, el objetivo se reparametriza usando la entidad cabeza `u` y la relación de consulta `q` (ya que `v` es justo lo desconocido). Así:

```
s_uq^(t)(x) = h_q^(t)(u,x) ⊗ g([h_q^(t)(u,x), q])
```

- `h_q^(t)(u,x)`: representación actual del nodo `x` (juega el papel de "distancia recorrida" `d(u,x)`).
- `g(·)`: red feed-forward (1 capa) que, a partir de la concatenación `[h_q^(t)(u,x), q]`, estima la "distancia restante" — compara la representación actual contra el vector de la relación de consulta `q` (el "objetivo").
- `q`: representación aprendida de la relación de consulta (vector de embedding).

La puntuación final de prioridad se pasa por otra red feed-forward (2 capas) y una sigmoide:

```
s_uq^(t)(x) = σ(f(s_uq^(t)(x)))  ∈ [0, 1]
```

Intuición: si `h_q^(t)(u,x)` ya está "cerca" de `q`, la distancia restante estimada es pequeña y `x` es probablemente cercano a la respuesta correcta.

### 3.2 Selección de nodos y aristas (Top-K / Top-L)

En cada iteración `t` (de un total de `T` iteraciones, `T=6` en todos los experimentos):

1. **Selección de nodos:** se seleccionan los `K` nodos con mayor prioridad `s_uq^(t-1)(x)` de entre todos los nodos visitables → conjunto `X^(t)`.
2. **Selección de aristas:** de las aristas salientes de `X^(t)`, se seleccionan las `L` aristas con mayor prioridad (reutilizando la misma función de prioridad, evaluada en el nodo *destino* de la arista, no en uno nuevo) → conjunto `E^(t)`. Esto evita que nodos de grado muy alto (p. ej. una entidad como "Human" conectada a millones de personas) disparen el costo computacional.
3. Solo los nodos alcanzados por `E^(t)` participan en la actualización de representaciones de esa iteración (Ec. 12/23 del paper):

```
h_q^(t)(u,v) ← h_q^(0)(u,v) ⊕  ⊕_{x∈X^(t), (x,r,v)∈E^(t)(v)} s_uq^(t-1)(x) · [h_q^(t-1)(u,x) ⊗ w_q(x,r,v)]
```

Nótese que la prioridad `s_uq^(t-1)(x)` se usa además como **peso multiplicativo del mensaje**, no solo como criterio de selección — esto empuja al modelo a aprender pesos grandes para los nodos verdaderamente relevantes.

**Hiperparámetros de control coste/calidad:**
- `α = K / |V|`: *node ratio* (proporción máxima de nodos seleccionados).
- `β = L·|V| / (K·|E|)`: *average degree ratio* (razón de aristas por nodo seleccionado, relativa al grado promedio del grafo completo).
- El ratio máximo de aristas queda determinado por `α·β`.

Ajustando `α` y `β` se puede mover el punto de operación en el trade-off desempeño/eficiencia: por ejemplo, aceptando un desempeño similar al de ConE (un método de embeddings), basta con `α=1%` o `β=10%` para lograr un speedup de 8.7× respecto a NBFNet.

### 3.3 Entrenamiento: pesos compartidos

**Problema clave:** no hay supervisión directa para la función de prioridad — no se sabe de antemano cuáles son los "caminos importantes". La solución adoptada es **compartir los pesos** entre la función de prioridad y el predictor de la tarea de razonamiento (es decir, `s_uq^(t)(x)` se calcula reutilizando las mismas transformaciones que producen la puntuación final `p(v|u,q)`). La intuición: cualquier entidad respuesta positiva debe estar sobre al menos un camino importante, mientras que las negativas probablemente no — así la señal de la tarea de completado actúa como supervisión débil de la función de prioridad. El ablation study confirma que esta técnica es esencial (sin compartir pesos, MRR baja de 0.411 a 0.374 en FB15k-237).

**Función de pérdida:** entropía cruzada binaria estándar sobre tripletas positivas y negativas (siguiendo a RotatE):

```
L = − log p(u,q,v) − Σ_{i=1}^{n} (1/n) log(1 − p(u'_i, q, v'_i))
```

donde las negativas se generan corrompiendo cabeza o cola de una tripleta positiva.

### 3.4 Parametrización concreta usada en los experimentos

- `⊗` (agregación de aristas dentro de un camino): operación relacional de **DistMult** (multiplicación vectorial elemento a elemento).
- `⊕` (agregación entre caminos/mensajes): **PNA** (*Principal Neighborhood Aggregation*) para FB15k-237, WN18RR, YAGO3-10 y ogbl-wikikg2 en su versión transductiva completa; suma simple en el setting inductivo.
- Condición de frontera `1_q(u=v)`: parametrizada con un embedding de consulta `q`; en ogbl-wikikg2 se aumenta con una noción de "distancia suave" basada en PageRank personalizado, dado el tamaño del grafo.
- `w_q(x,r,v)`: función lineal de la relación de consulta `q` (`W_r·q + b_r`) en la mayoría de datasets; en WN18RR se usa directamente un embedding de la relación.
- Preprocesamiento: aumento del grafo con tripletas invertidas (`(y, r⁻¹, x)` por cada `(x,r,y)`), y *edge dropout* de las tripletas de consulta durante entrenamiento para evitar que el modelo "copie" la respuesta desde el propio grafo de entrada.
- Entrenamiento con 4 GPUs Tesla A100 (40GB).

### 3.5 Implementación eficiente: operaciones "padding-free"

Un desafío práctico: como distintas muestras de un batch seleccionan conjuntos `V^(t)` y `E^(t)` de tamaños diferentes, el enfoque estándar de *padding* (rellenar hasta un tamaño fijo) anularía buena parte de la ganancia de eficiencia de A*Net. Los autores introducen operaciones **top-k sin padding**: en vez de rellenar, concatenan todas las muestras del batch en un único tensor grande, junto con un identificador de muestra por elemento, y convierten el top-k por muestra en un problema de **ordenamiento multi-clave** (primero por ID de muestra, luego por valor), implementado mediante dos llamadas sucesivas a *sort* estable — esto permite paralelizar la operación en GPU sin desperdiciar cómputo en posiciones "vacías".

---

## 4. Configuración experimental

**Datasets** (2 transductivos completos en el cuerpo del paper, YAGO3-10 con detalle en apéndice, e inductivos):

| Dataset | #Relaciones | #Entidades | #Tripletas train |
|---|---|---|---|
| FB15k-237 | 237 | 14,541 | 272,115 |
| WN18RR | 11 | 40,943 | 86,835 |
| YAGO3-10 | 37 | 123,182 | 1,079,040 |
| ogbl-wikikg2 | 535 | 2,500,604 | 16,109,182 |

Para el setting inductivo se usan las 4 particiones (v1–v4) estándar de Teru et al. (GraIL), donde las entidades de test no aparecen en entrenamiento.

**Métricas:** Mean Reciprocal Rank (MRR) y Hits@K bajo el protocolo de *filtered ranking* estándar. Eficiencia medida como número promedio de mensajes por paso, tiempo por época y memoria pico.

**Baselines:** métodos de embeddings (TransE, ComplEx, RotatE, HAKE, RotH, PairRE, ComplEx+RP, ConE), GNNs (RGCN, CompGCN, GraIL), y métodos basados en caminos (MINERVA, Multi-Hop, CURL, NeuralLP, DRUM, NBFNet, RED-GNN).

---

## 5. Resultados principales

### 5.1 Setting transductivo (FB15k-237 y WN18RR)

A*Net **supera a todos los métodos de embeddings y GNNs**, y queda **a la par de NBFNet** (el estado del arte entre métodos basados en caminos), pese a explorar solo una fracción del grafo:

| Método | FB15k-237 MRR | WN18RR MRR |
|---|---|---|
| RotatE (embedding) | 0.338 | 0.476 |
| ComplEx+RP (embedding) | 0.388 | 0.488 |
| NBFNet | **0.415** | **0.551** |
| RED-GNN | 0.374 | 0.533 |
| **A*Net** | 0.411 | 0.549 |

En predicción de cola (tail prediction) frente a métodos de "path-finding" con RL (MINERVA, Multi-Hop, CURL), A*Net y NBFNet muestran ventajas similares, lo que sugiere que **agregar múltiples caminos** (como hacen A*Net/NBFNet) es superior a seleccionar un único camino vía RL.

### 5.2 Eficiencia en el setting transductivo

Esta es la contribución central del paper: A*Net logra el desempeño de NBFNet propagando solo por **~10% de los nodos y ~10% de las aristas** en cada iteración.

| Método | FB15k-237 #mensajes | FB15k-237 tiempo | FB15k-237 memoria | WN18RR #mensajes | WN18RR tiempo | WN18RR memoria |
|---|---|---|---|---|---|---|
| NBFNet | 544,230 | 16.8 min | 19.1 GiB | 173,670 | 9.42 min | 26.4 GiB |
| A*Net | 38,610 | 8.07 min | 11.1 GiB | 4,049 | 1.39 min | 5.04 GiB |
| **Mejora** | **14.1×** | **2.1×** | **1.7×** | **42.9×** | **6.8×** | **5.2×** |

Nótese que la reducción en tiempo/memoria es **menor** que la reducción en número de mensajes — esto se debe a que A*Net opera sobre subgrafos de tamaño dinámico, más difíciles de paralelizar eficientemente en GPU que la propagación densa de NBFNet. Los autores lo señalan explícitamente como limitación y línea de trabajo futuro (mejor co-diseño algoritmo/sistema).

A*Net también converge notablemente más rápido en tiempo real de entrenamiento (curvas de MRR de validación vs. tiempo de entrenamiento, Fig. 5 del paper).

### 5.3 Escalabilidad: ogbl-wikikg2 (escala de millones)

Este es el resultado más llamativo del paper. `ogbl-wikikg2` tiene 2.5 millones de entidades y 16 millones de tripletas — dos órdenes de magnitud más grande que los datasets previamente abordados por métodos basados en caminos.

| Método | Test MRR | Valid MRR | #Parámetros |
|---|---|---|---|
| TransE | 0.4256 | 0.4272 | 1,251 M |
| ComplEx | 0.4027 | 0.3759 | 1,251 M |
| RotatE | 0.4332 | 0.4353 | 1,251 M |
| PairRE | 0.5208 | 0.5423 | 500 M |
| ComplEx+RP | 0.6392 | 0.6561 | 250 M |
| **NBFNet** | **OOM** | **OOM** | **OOM** |
| **A*Net** | **0.6767** | **0.6851** | **6.83 M** |

Puntos clave:
- **NBFNet directamente no corre** (out-of-memory) incluso con batch size 1, al tener que propagar por todo el grafo.
- A*Net, propagando por solo **0.2% de nodos y 0.2% de aristas** por iteración, no solo logra correr sino que **establece un nuevo estado del arte**, superando incluso a los mejores métodos de embeddings.
- A*Net usa solo **6.83 millones de parámetros** — **36.6× menos** que ComplEx+RP — porque solo aprende parámetros para las *relaciones*, no una tabla de embeddings por cada una de las 2.5M entidades.
- A*Net converge sustancialmente más rápido que los métodos de embeddings (Fig. 1 del paper).
- A*Net es el **primer método no basado en embeddings** evaluado con éxito en ogbl-wikikg2.

### 5.4 Setting inductivo (FB15k-237 v1–v4, WN18RR v1–v4)

Los métodos de embeddings no pueden aplicarse aquí (no generalizan a entidades nuevas). A*Net queda **a la par de NBFNet** y **supera claramente** a GraIL, NeuralLP, DRUM y RED-GNN en la mayoría de las particiones. Por ejemplo en FB15k-237: A*Net obtiene MRR de 0.457/0.510/0.476/0.466 en v1–v4, frente a NBFNet con 0.422/0.514/0.476/0.453 — resultados muy similares, con A*Net incluso mejor en v1 y v4.

### 5.5 Estudios de ablación (FB15k-237 transductivo)

**(a) Elección de función de prioridad:**

| Función de prioridad | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| Personalized PageRank (PPR) | 0.266 | 0.212 | 0.296 | 0.371 |
| Grado del nodo (Degree) | 0.347 | 0.268 | 0.383 | 0.501 |
| Aleatoria (Random) | 0.378 | 0.288 | 0.413 | 0.556 |
| **Neural (propuesta)** | **0.411** | **0.321** | **0.453** | **0.586** |

La función de prioridad **aprendida** supera claramente a cualquier heurística manual, confirmando que no existe un buen "A* clásico" para KGs y que aprenderla end-to-end es necesario.

**(b) Compartir pesos entre la función de prioridad y el predictor:**

| Pesos compartidos | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| No | 0.374 | 0.282 | 0.413 | 0.557 |
| **Sí** | **0.411** | **0.321** | **0.453** | **0.586** |

Compartir pesos es **esencial** para poder entrenar una buena función de prioridad sin supervisión directa.

**(c) Trade-off desempeño/eficiencia:** variando `α` (node ratio) y `β` (degree ratio) se obtienen curvas de MRR vs. speedup relativo a NBFNet; con `α=1%` o `β=10%` se logra desempeño comparable a métodos de embeddings (ConE) con **8.7× speedup**.

### 5.6 Interpretabilidad

A diferencia de los métodos de embeddings (caja negra), A*Net permite **extraer los caminos importantes** aprendidos por la función de prioridad, vía *beam search* sobre las puntuaciones de prioridad de cada paso (fórmula de importancia normalizada de camino, Ec. 14 del paper). Ejemplo cualitativo: para la consulta `(Bandai, industry, ?)`, A*Net recupera caminos como `Bandai <--subsidiary-- Bandai Namco --industry--> video game` y `Bandai --industry--> media <--industry-- Pony Canyon --industry--> video game`, ambos consistentes con el sentido común humano.

---

## 6. Conclusiones y limitaciones declaradas por los autores

- A*Net logra el objetivo central: llevar el razonamiento basado en caminos (con su ventaja de generalización inductiva) a la escala de millones de entidades, algo que antes solo lograban los métodos de embeddings.
- La limitación principal reconocida es de **ingeniería de sistemas**: la reducción en número de mensajes (hasta 42.9×) no se traduce proporcionalmente en reducción de tiempo/memoria real (2–7× y 1.7–5.2× respectivamente), porque operar sobre subgrafos de tamaño dinámico es menos amigable para la paralelización en GPU que la propagación densa y uniforme de NBFNet. Los autores dejan como trabajo futuro el co-diseño algoritmo/sistema para cerrar esa brecha.
- Impacto social señalado: por el lado positivo, reduce tiempo de entrenamiento/inferencia (menor huella de carbono); por el lado negativo, modelos de razonamiento más escalables podrían usarse para descubrir relaciones sensibles en datos anonimizados.

---

# APÉNDICES B–F (agregado 2026-08-26) — detalles de implementación que SÍ nos afectan

## B. La segunda etapa de selección de aristas — confirmación textual

> *"E(t) ← TopL(s(t−1)_uq(v) | x ∈ X(t), (x,r,v) ∈ E(x))* — **each edge is picked according to
> the priority of node v, i.e., the tail node of an edge**. By doing so, we reuse the neural
> priority function and avoid introducing any additional priority function. The intuition is
> that if an edge (x,r,v) goes to a node with a higher priority, it is likely we are propagating
> towards the answer entities."*

⇒ Confirma que las aristas se ordenan por la prioridad del **DESTINO**, y **por qué**: es la
estimación de "distancia restante" del A\*. También confirma que **no hay función de prioridad
extra** — se reusa la misma. (Ya corregido en nuestra v3.)

Y el motivo del paso: *"some nodes may have very large degrees, e.g., the entity Human is
connected to every person"* ⇒ el top-L existe para que un hub no dispare el costo.

## 🚩 E. Tabla 9 — HIPERPARÁMETROS. Lo que más nos importa

| | FB15k-237 transd. | WN18RR transd. | **YAGO3-10** | ogbl-wikikg2 |
|---|---|---|---|---|
| #step (T) | 6 | 6 | 6 | 6 |
| hidden dim | 32 | 32 | 32 | 32 |
| aggregation | PNA | PNA | **PNA** | sum |
| g(·) #layer | 1 | 1 | 1 | 1 |
| f(·) #layer | 2 | 2 | 2 | 2 |
| f hidden dim | 64 | 64 | **64** | 64 |
| **node ratio α** | 10 % | 10 % | **10 %** | **0.2 %** |
| **degree ratio β** | 100 % | 100 % | **100 %** | 100 % |
| **batch size** | 256 | 256 | **40** | 128 |
| learning rate | 5e-3 | 5e-3 | 5e-3 | 5e-3 |
| **#epoch** | 20 | 20 | **0.4** | **0.2** |
| adv. temperature | 0.5 | 1 | 0.5 | 0.5 |
| #negative | 32 | 32 | 32 | **1 048 576** |

⚠️⚠️ **`#epoch 0.4` EN YAGO Y `0.2` EN wikikg2.** No entrenan ni media pasada sobre el dataset.
En YAGO: 0.4 × 1 079 040 / 40 = **~10 800 pasos a batch 40 = ~430 000 triples vistos**.

⚠️ **β = 100 % significa que NO truncan aristas por nodo**: toman TODAS las salientes de los
nodos seleccionados. Nuestro `--edge_cap` es una desviación.

⚠️ **wikikg2 usa 1 048 576 negativos**, no 32. Es otro régimen de loss por completo.

## E. Detalle que CORROBORA nuestro hallazgo sobre la normalización

> *"We observe that **PNA does not generalize well when degrees are dynamically determined by
> the priority function**. Therefore, we **precompute the degree for each node on the full
> graph**, and use them in PNA no matter how many nodes and edges are selected."*

⇒ Encontraron **el mismo fenómeno que medimos nosotros**: normalizar por lo que quedó
seleccionado no funciona; hay que usar la magnitud del grafo COMPLETO. Es exactamente la causa
que propusimos para que `--prune_attn sigmoid` (sin normalizar) le gane a `softmax` bajo poda
(SESSION_NOTES 2026-08-26). **Evidencia independiente, de otro grupo, del mismo mecanismo.**

## E. Otros detalles

- **Indicador en wikikg2**: no usan el 1-hot de siempre sino `1(u=v)q + 1(u≠v)p_uv`, con `p_uv`
  un embedding aprendido sobre el **PageRank personalizado** discretizado ⇒ "distancia suave".
  A esa escala la condición de frontera casi toda en ceros no les alcanzó.
- Quitan las aristas de las queries de train (igual que nuestro `graph_mask`).

## D. Tamaños de los splits

| | #Rel | #Ent | train | valid | test |
|---|---:|---:|---:|---:|---:|
| FB15k-237 | 237 | 14 541 | 272 115 | 17 535 | 20 466 |
| WN18RR | 11 | 40 943 | 86 835 | 3 034 | 3 134 |
| **YAGO3-10** | **37** | **123 182** | **1 079 040** | **5 000** | **5 000** |
| ogbl-wikikg2 | 535 | 2 500 604 | 16 109 182 | 429 456 | 598 543 |

## F. Resultados y eficiencia en YAGO3-10 (Tabla 10)

| método | MRR | H@1 | H@3 | H@10 |
|---|---:|---:|---:|---:|
| DistMult | 0.34 | 0.24 | 0.38 | 0.54 |
| ComplEx | 0.36 | 0.26 | 0.40 | 0.55 |
| RotatE | 0.495 | 0.402 | 0.550 | 0.670 |
| **NBFNet** | **0.563** | **0.480** | **0.612** | **0.708** |
| A\*Net | 0.556 | 0.470 | 0.611 | 0.707 |

Eficiencia en YAGO: A\*Net reduce **16.0× los mensajes**, 2.5× el tiempo y 2.0× la memoria
respecto de NBFNet.

## C. Top-k sin padding (Alg. 2) — por si alguna vez hace falta

Multi-key sort con dos `sort` estables: `indices = inputs.argsort()`;
`indices = sample_ids[indices].argsort(stable=True)`.

~~**No lo usamos**: nuestro top-L/top-M es de tamaño FIJO, que da la misma reducción de memoria
a cambio de algo de cómputo en padding.~~

⚠️ **ESE DESCARTE ERA INCORRECTO (corregido 2026-08-30). Costó +0.098 de MRR en YAGO.**

El error fue aplicar un hecho verdadero a la parte equivocada del pipeline. Hay DOS lugares
donde aparece el padding y sólo uno es de tamaño fijo:

- **La SALIDA del top-L**: sí es de tamaño fijo (cada query propaga exactamente L aristas).
  La nota era correcta *para esto*, y por eso resistió cada relectura.
- **El POOL DE ENTRADA del que el top-L elige**: para armar la matriz rectangular
  `(B, K·cap)` hay que fijar cuántas casillas tiene cada nodo ⇒ el padding **obliga** a un
  tope por nodo (`--edge_cap`). Eso no es cómputo desperdiciado: **descarta evidencia**, y la
  descarta tomando las primeras `cap` aristas de la fila CSR, o sea en el orden en que venían
  en el archivo del dataset (`src/model.py:1631`). Misma categoría que el desalojo por id
  global del 2026-08-27.

A\*Net no tiene tope por nodo: `neighbors()` (`reasoning/data.py:276`) devuelve TODAS las
salientes de los nodos seleccionados y el único recorte es global, vía `variadic_topks`.

Medido (YAGO3-10, todo lo demás igual): `edge_cap` 64 → **0.4498**, 1024 → **0.5478**.
**+0.098 y sin saturar** — más que el gap que nos separaba de NBFNet. Ver SESSION_NOTES
2026-08-28 (b) y 2026-08-30.
