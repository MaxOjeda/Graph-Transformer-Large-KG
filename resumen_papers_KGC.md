# Resumen de Papers: Modelos para Link Prediction en Knowledge Graphs

---

## 1. NBFNet — Neural Bellman-Ford Networks

**Zhu, Zhang, Xhonneux, Tang (Mila / Université de Montréal). NeurIPS 2021.**

### Motivación
Los métodos tradicionales de link prediction se dividen en (a) métodos basados en caminos (Katz index, Personalized PageRank, graph distance) — interpretables, inductivos, pero con heurísticas fijas no óptimas — y (b) GNNs tipo auto-encoder, que son transductivos salvo que existan features de nodo, y arquitecturas tipo SEAL/GraIL que materializan subgrafos por cada enlace, lo cual no escala. El objetivo es combinar lo mejor de ambos mundos: generalización inductiva, interpretabilidad, alta capacidad y escalabilidad.

### Metodología

**Formulación de caminos.** La representación de un par (u, v) condicionada a una relación de consulta q se define como una suma generalizada de las representaciones de todos los caminos entre u y v:

- h_q(u,v) = ⊕_{P∈P_uv} h_q(P)
- h_q(P) = ⊗_{i=1}^{|P|} w_q(e_i)

donde ⊕ es un operador de "suma" conmutativo y ⊗ un operador de "producto" (no necesariamente conmutativo). Esta formulación generaliza: Katz index (⊕=+, ⊗=×), Personalized PageRank, graph distance (⊕=min, ⊗=+), widest path (⊕=max, ⊗=min) y most reliable path (⊕=max, ⊗=×) — demostrado formalmente en los Teoremas 1–5.

**Algoritmo de Bellman-Ford generalizado.** Bajo el supuesto de que ⟨⊕,⊗⟩ forman un semiring, la formulación de caminos se puede resolver eficientemente (evitando la explosión exponencial de caminos) mediante:

- h⁽⁰⁾_q(u,v) ← 1_q(u=v)
- h⁽ᵗ⁾_q(u,v) ← [⊕_{(x,r,v)∈E(v)} h⁽ᵗ⁻¹⁾_q(u,x) ⊗ w_q(x,r,v)] ⊕ h⁽⁰⁾_q(u,v)

**Neural Bellman-Ford Networks (NBFNet).** Se relaja el supuesto de semiring parametrizando los operadores con 3 funciones neuronales: INDICATOR (reemplaza la condición de borde), MESSAGE (reemplaza ⊗) y AGGREGATE (reemplaza ⊕):

- h⁽⁰⁾_v ← INDICATOR(u,v,q)
- h⁽ᵗ⁾_v ← AGGREGATE({MESSAGE(h⁽ᵗ⁻¹⁾_x, w_q(x,r,v)) | (x,r,v)∈E(v)} ∪ {h⁽⁰⁾_v})

Diseño concreto:
- **INDICATOR**: 1(u=v) * q (embedding de la query en el nodo fuente, ceros en el resto).
- **MESSAGE**: operadores relacionales de embeddings de KG — suma (TransE), producto elemento a elemento (DistMult), o rotación (RotatE).
- **AGGREGATE**: sum, mean, max, o Principal Neighborhood Aggregation (PNA), seguido de transformación lineal + no linealidad.
- Representaciones de arista dependientes de la query: w_q(x,r,v) = W_r q + b_r (o solo b_r para grafos con pocas relaciones).
- Se agregan tripletas inversas ⟨v, q⁻¹, u⟩ para KGs, y self-loops para grafos homogéneos.
- Predicción: p(v|u,q) = σ(f(h_q(u,v))), con f un MLP de 2 capas.

**Complejidad temporal.** O(|E|d/|V| + d²) amortizado por tripleta — uno de los más bajos entre métodos inductivos (comparado con SEAL: O(|E|d²), NeuralLP/DRUM: O(|E|d/|V| + d²)).

**Interpretabilidad.** Se aproxima la importancia de cada camino mediante una expansión de Taylor de primer orden sobre p(u,q,v), aproximando la importancia del camino como la suma de importancias de arista (obtenidas por diferenciación automática); el top-k de caminos se obtiene con búsqueda tipo beam search de Bellman-Ford sobre el grafo de importancias de arista.

### Resultados

- **KG completion (transductivo)** en FB15k-237 y WN18RR: NBFNet obtiene el mejor MRR y H@10 de todos los métodos comparados (path-based, embeddings, GNNs). En FB15k-237: MR 114, MRR 0.415, H@1 0.321, H@10 0.599. En WN18RR: MR 636, MRR 0.551, H@1 0.497, H@10 0.666 — superando a RotatE, HAKE, LowFER, RGCN y GraIL.
- **Grafos homogéneos** (Cora, Citeseer, PubMed): mejores resultados en Cora y PubMed (AUROC 0.956 y 0.983 respectivamente), competitivo en Citeseer, a pesar de no usar features de nodo.
- **Inductive relation prediction** (HITS@10, splits v1–v4 de GraIL): NBFNet supera a NeuralLP, DRUM, RuleN y GraIL en las 8 combinaciones (FB15k-237 y WN18RR), con ganancia relativa promedio de 22% sobre GraIL (el mejor previo).
- **Ablaciones:** PNA > sum/mean/max como AGGREGATE; combinaciones que satisfacen la condición de semiring (TransE+max, DistMult+sum) son localmente óptimas entre las funciones simples. El desempeño aumenta monótonamente con el número de capas, saturando en 6 capas (sin el degradamiento típico de GNNs profundas).
- **Por categoría de relación:** NBFNet mejora tanto en casos fáciles (1-a-1) como en los difíciles (1-a-N, N-a-1, N-a-N), donde los métodos de embeddings fallan.

### Limitaciones
El supuesto de semiring no se sostiene formalmente por las no linealidades (no hay garantía teórica del error introducido por esta relajación). Solo se valida en edge prediction simple, no en consultas lógicas complejas (conjunciones/disyunciones).

---

## 2. ULTRA — Towards Foundation Models for Knowledge Graph Reasoning

**Galkin, Yuan, Mostafa, Tang, Zhu (Intel AI Lab / Mila). ICLR 2024.**

### Motivación
Los modelos de foundation (lenguaje, visión) transfieren gracias a un vocabulario fijo (tokens, píxeles). Los KGs no tienen esto: cada grafo tiene su propio vocabulario de entidades y relaciones, que típicamente no se solapan entre grafos. Los métodos inductivos previos (NBFNet, RED-GNN, NodePiece) transfieren a **nuevas entidades** pero requieren el **mismo conjunto de relaciones** en entrenamiento e inferencia. El objetivo de ULTRA es la transferencia *fully-inductive*: generalización zero-shot a grafos con entidades **y** relaciones completamente nuevas (V_train∩V_inf=∅, R_train∩R_inf=∅).

### Metodología

**Idea central — grafo de relaciones.** Aunque las relaciones cambien entre grafos, sus *interacciones estructurales* (qué relaciones comparten entidades como cabeza/cola) son invariantes y transferibles. ULTRA construye G_r = LIFT(G): un grafo donde cada nodo es un tipo de relación del grafo original, con aristas que capturan 4 interacciones fundamentales R_fund:
- **t2h** (tail-to-head), **h2h** (head-to-head), **h2t** (head-to-tail), **t2t** (tail-to-tail).

El tensor de adyacencia completo es A_r ∈ ℝ^{|R|×|R|×4}, calculable eficientemente vía multiplicación de matrices dispersas.

**Algoritmo de tres pasos** (dado una query (h,q,?) sobre G):

1. **Construcción del grafo de relaciones** G_r = LIFT(G).
2. **Representaciones relativas de relación** R_q condicionadas a la relación de consulta q, usando un *labeling trick* (INDICATOR_r(v,q) = 1_{v=q} * 1^d, vector de unos — se encontró empíricamente que generaliza mejor que un vector aprendible) seguido de una GNN tipo NBFNet (GNN_r) sobre G_r con función de mensaje DistMult no paramétrica y agregación suma. Los únicos parámetros aprendibles por capa son los embeddings de las 4 interacciones fundamentales R_fund ∈ ℝ^{4×d} y una capa lineal de update.
3. **Link prediction a nivel de entidad**: se usa otra instancia de NBFNet (GNN_e) sobre el grafo original G, inicializando el nodo h con R_q[q] (en vez de un embedding fijo de query), y transformando las representaciones de relación por capa mediante g^t(·) (MLP de 2 capas con ReLU).

Como ULTRA no aprende embeddings específicos de entidad/relación de ningún grafo, es directamente aplicable a cualquier grafo multi-relacional nuevo sin necesidad de features de entrada.

**Entrenamiento.** Loss de entropía cruzada binaria estándar sobre tripletas positivas/negativas (corrompiendo cabeza o cola), igual que NBFNet. Modelo pequeño: 177k parámetros totales (60k en GNN_r, 117k en GNN_e). Preentrenado en mezcla de 3 grafos (WN18RR, CoDEx-Medium, FB15k-237), 200.000 pasos, batch 64, AdamW, 2×A100.

### Resultados

**Evaluación en 57 KGs** distintos, categorizados en: transductivos (16), inductivos con solo entidades nuevas (18), e inductivos con entidades y relaciones nuevas (23).

- **Zero-shot vs. SOTA supervisado (51 datasets con baseline disponible):** ULTRA 0-shot ya supera en promedio al SOTA supervisado entrenado específicamente por dataset: MRR 0.395 vs 0.344, H@10 0.556 vs 0.486. Ganancias más grandes en grafos inductivos pequeños (FB-25, FB-50: ~3× mejor, +291% y +289%).
- **Fine-tuning** (1000–2000 pasos, gracias a la eficiencia de muestra de NBFNet): mejora a MRR 0.422 / H@10 0.592 en total, cerrando la brecha en grafos transductivos grandes donde el 0-shot es más débil (entrenamiento en grafos de 15-40k nodos, inferencia hasta 123k).
- **Comparación con entrenar de cero por grafo (train e2e):** el modelo preentrenado 0-shot rinde casi a la par (MRR 0.366 vs 0.393 promedio en 54 grafos), superando en datasets inductivos y quedando algo por detrás en transductivos grandes; con fine-tuning ULTRA supera al entrenamiento e2e.
- **Número de grafos en la mezcla de preentrenamiento:** el desempeño zero-shot se satura después de ~3 grafos en la mezcla (probado hasta 8).
- **Ablación clave (Tabla 3):** remover las 4 interacciones fundamentales (grafo de relaciones homogéneo) y usar una GNN no condicional (arquitectura R-GATv2 de InGram) degrada dramáticamente el desempeño — hasta -48% relativo en MRR (0.192 vs 0.366) con inicialización aleatoria de nodos. Esto confirma que las representaciones condicionales (tanto a nivel de relación como de entidad) son cruciales.

### Limitaciones y trabajo futuro
Más grafos en la mezcla de preentrenamiento no siempre mejora el desempeño — se hipotetiza que la capacidad del modelo (177k parámetros) es limitada, aunque escalar más allá de 200k parámetros tampoco mostró mejoras claras en experimentos preliminares (posible problema de normalización/optimización).

---

## 3. A*Net — A Scalable Path-based Reasoning Approach for Knowledge Graphs

**Zhu, Yuan, Galkin, Xhonneux, Zhang, Gazeau, Tang (Mila / Univ. de Montréal / Intel AI Lab / Peking Univ. / LG Electronics AI Lab). NeurIPS 2023.**

### Motivación
El razonamiento a gran escala sobre KGs ha sido dominado por métodos de embeddings (simples, escalables vía sistemas multi-GPU) porque los métodos basados en caminos, aunque poseen capacidad inductiva que los embeddings no tienen (generalizan a entidades nuevas), son intratables computacionalmente: el número de caminos crece exponencialmente con su longitud. Los métodos tipo Bellman-Ford (NeuralLP, DRUM, NBFNet, RED-GNN) logran complejidad polinomial, pero aún deben propagar mensajes a través de **todos** los |V| nodos y |E| aristas en cada iteración — inviable para grafos de millones de entidades (p. ej. ogbl-wikikg2, que hace que NBFNet directamente falle por out-of-memory). La observación central de A*Net es que, para responder una query dada, solo una pequeña fracción de caminos son realmente importantes.

### Metodología

**Formulación de caminos y algoritmo de Bellman-Ford (marco compartido con NBFNet).** Igual que en NBFNet, h_q(u,v) = ⊕_{P∈P_{u⇝v}} ⊗_{(x,r,y)∈P} w_q(x,r,y), resuelto vía iteración de Bellman-Ford:
- h⁽⁰⁾_q(u,v) ← 1_q(u=v)
- h⁽ᵗ⁾_q(u,v) ← h⁽⁰⁾_q(u,v) ⊕ ⊕_{(x,r,v)∈E(v)} h⁽ᵗ⁻¹⁾_q(u,x) ⊗ w_q(x,r,v)

**Algoritmo A\*** (para shortest path clásico): en vez de propagar uniformemente, prioriza nodos según una función de prioridad heurística s(x) = d(u,x) ⊗ g(x,v), donde d(u,x) es el costo actual desde u hasta x y g(x,v) estima el costo restante hasta el objetivo v (p. ej. distancia L1 en un grid).

**Caminos importantes (Sec. 3.1).** Se define formalmente P_{u⇝v|q} ⊆ P_{u⇝v} como el subconjunto de caminos importantes para la query q, tal que h_q(u,v) = ⊕_{P∈P_{u⇝v}} h_q(P) ≈ ⊕_{P∈P_{u⇝v|q}} h_q(P) — cualquier camino fuera de este subconjunto (p. ej. uno que use una relación irrelevante como "Friend" para predecir "Mother") contribuye de forma despreciable. Se prueba formalmente (Thm. A.1–A.4 en apéndice) que:
1. Existe una función de selección de caminos m_q que, aplicada iterativamente sobre la estructura de árbol de los caminos, cubre los caminos importantes (Ec. 6-7).
2. Esta selección iterativa de caminos puede aproximarse por una **selección iterativa de nodos** n⁽ᵗ⁾_{uq}: V ↦ 2^V (Ec. 8), asumiendo que caminos con igual longitud y mismo nodo final pueden fusionarse (Prop. A.3) — convirtiendo la búsqueda exponencial en programación dinámica polinomial.
3. Bajo el supuesto de semiring, la iteración A* resultante recupera exactamente la agregación de los caminos importantes (Ec. 9): h⁽ᵗ⁾_q(u,v) ← h⁽⁰⁾_q(u,v) ⊕ ⊕_{x∈n⁽ᵗ⁻¹⁾_{uq}(V), (x,r,v)∈E(v)} h⁽ᵗ⁻¹⁾_q(u,x) ⊗ w_q(x,r,v).

**Función de prioridad neuronal (Sec. 3.2).** Como no existe un oráculo (ni una heurística manual efectiva — ver ablación Tab. 6a) para n⁽ᵗ⁾_{uq}(V) en KGs (semántica relacional compleja), A*Net aprende s⁽ᵗ⁾_{uq}(x) end-to-end. Inspirado en s(x)=d(u,x)⊗g(x,v) del A* clásico, y notando que el nodo objetivo v puede reparametrizarse como (u,q) (la query misma), se define:
- s⁽ᵗ⁾_{uq}(x) = h⁽ᵗ⁾_q(u,x) ⊗ g([h⁽ᵗ⁾_q(u,x), **q**])   — g(·) es un FFN, [·,·] concatena, **q** es el embedding aprendido de la relación de consulta.
- s⁽ᵗ⁾_{uq}(x) = σ(f(s⁽ᵗ⁾_{uq}(x)))  — score final en [0,1], f(·) FFN.

Intuición: si h_q(u,x) ya está "cerca" del objetivo **q**, la representación restante g(·) será cercana a 0, indicando que x está cerca de la respuesta correcta.

**Integración en el mensaje (Ec. 12):** se pesa cada mensaje por la prioridad del nodo emisor en la iteración anterior: h⁽ᵗ⁾_q(u,v) ← h⁽⁰⁾_q(u,v) ⊕ ⊕_{x∈X⁽ᵗ⁾,(x,r,v)∈E(v)} s⁽ᵗ⁻¹⁾_{uq}(x)·(h⁽ᵗ⁻¹⁾_q(u,x) ⊗ w_q(x,r,v)). En cada iteración se seleccionan **top-K nodos** y, dentro de sus vecindarios, **top-L aristas** (para evitar explosión en nodos de grado muy alto), reutilizando la misma función de prioridad para ambas selecciones (Apéndice B) — sin introducir una función de prioridad adicional para aristas.

**Entrenamiento — weight sharing crítico.** No hay supervisión directa para la función de prioridad (no se conocen los caminos importantes verdaderos). La solución es **compartir los pesos** entre la función de prioridad y el predictor de la tarea de razonamiento: cualquier entidad respuesta positiva debe estar en al menos un camino importante, dando una supervisión débil pero efectiva. Se entrena minimizando entropía cruzada binaria estándar sobre tripletas positivas/negativas (igual que NBFNet/ULTRA).

**Implementación eficiente — operaciones sin padding ("padding-free").** Como distintas muestras de un batch seleccionan cantidades muy distintas de nodos/aristas, el padding tradicional para paralelizar en GPU anularía la ganancia de eficiencia. Se propone un top-k sin padding: se empareja cada elemento con un ID de muestra y se resuelve como un problema de *multi-key sort* (dos llamadas a sort estable), evitando el overhead de padding (Apéndice C, Alg. 2).

**Configuración de capas.** Igual que NBFNet: función de mensaje DistMult (multiplicación vectorial), agregación PNA o suma, 6 capas, 32 dims ocultas. Dos hiperparámetros clave: ratio de nodos α=K/|V| y ratio de grado promedio β=L|V|/K|E|.

### Resultados

- **Transductivo (FB15k-237, WN18RR — Tabla 1):** A*Net es comparable a NBFNet (FB15k-237: MRR 0.411 vs 0.415; WN18RR: MRR 0.549 vs 0.551) y supera a todos los métodos de embeddings (TransE, RotatE, HAKE, RotH, ComplEx+RP, ConE) y GNNs (RGCN, CompGCN), propagando solo **10% de nodos y 10% de aristas** por iteración.
- **Eficiencia (Tabla 3):** A*Net reduce el número de mensajes 14.1× (FB15k-237) y 42.9× (WN18RR) respecto a NBFNet, con reducciones de tiempo de 2.1×/6.8× y de memoria de 1.7×/5.2× (la reducción en tiempo/memoria es menor que en mensajes porque A*Net opera sobre subgrafos de tamaño dinámico, más difíciles de paralelizar en GPU).
- **Tail prediction vs. métodos de path-finding con RL (Tabla 2):** A*Net (y NBFNet) superan a MINERVA, Multi-Hop y CURL — la ventaja de agregar múltiples caminos frente a seleccionar solo uno vía RL.
- **ogbl-wikikg2 (Tabla 4 — 2.5M entidades, 16M tripletas):** NBFNet falla por out-of-memory incluso con batch size 1. A*Net, propagando solo **0.2% de nodos y 0.2% de aristas**, logra **nuevo estado del arte** (MRR test 0.6767 vs. 0.6392 de ComplEx+RP), siendo el primer método no basado en embeddings evaluado en este dataset a esta escala. Además usa solo 6.83M parámetros (36.6× menos que ComplEx+RP) y converge notablemente más rápido (Fig. 1).
- **Inductivo (Tabla 5, splits v1-v4 de FB15k-237/WN18RR):** A*Net es comparable a NBFNet y supera claramente a GraIL, NeuralLP, DRUM y RED-GNN (los métodos de embeddings no aplican al setting inductivo).
- **Ablaciones (Tabla 6):**
  - *Función de prioridad:* la versión neuronal (MRR 0.411) supera ampliamente a heurísticas hechas a mano — PPR (0.266), Degree (0.347) e incluso Random (0.378) — confirmando que no existe una heurística manual efectiva para KGs (a diferencia del grid-world clásico de A*).
  - *Weight sharing:* esencial — sin compartir pesos entre la función de prioridad y el predictor, MRR cae de 0.411 a 0.374.
  - *Trade-off desempeño/eficiencia (Fig. 6):* variando α y β se puede obtener un speedup de 8.7× respecto a NBFNet aceptando un desempeño similar al de métodos de embeddings (p. ej. ConE), fijando α=1% o β=10%.
- **Interpretabilidad (Sec. 4.4, Fig. 7 y 9):** se extraen los caminos importantes vía beam search sobre la función de prioridad por paso (Ec. 14); ejemplos como *(Bandai, industry, ?)* recuperan caminos consistentes con el sentido común (Bandai↔Bandai Namco→video game; Bandai→media↔Pony Canyon→video game).

### Limitaciones
El trabajo se centra en diseño algorítmico, no de sistema: la mejora real en tiempo/memoria es menor que la reducción en el número de mensajes, porque operar sobre subgrafos de tamaño dinámico es más difícil de paralelizar en GPU que el enfoque denso de NBFNet — se deja como trabajo futuro el co-diseño algoritmo-sistema.

---

## 4. A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs

**Huang, Romero (PUC Chile / CENIA), Ceylan, Barceló (PUC Chile / IMFD / CENIA). NeurIPS 2023.**

### Motivación
Existe buena comprensión teórica de la expresividad de las GNNs relacionales tipo R-MPNN (RGCN, CompGCN) para tareas a nivel de nodo, vía relational WL. Pero estos modelos son subóptimos para link prediction porque una buena representación de nodo no induce necesariamente una buena representación de arista. Arquitecturas específicas para link prediction (NBFNet, GraIL, PathCon) carecen de una teoría equivalente. El objetivo del paper es dar una teoría unificada de la capacidad expresiva de las arquitecturas que computan representaciones **pareadas** (pairwise) de nodos.

### Metodología — Marco teórico

**Conditional Message Passing Neural Networks (C-MPNN).** Generalización formal que unifica NBFNet y modelos relacionados (NeuralLP, DRUM). Computa representaciones h_{v|u,q} condicionadas a un nodo fuente fijo u y relación de consulta q:

- h⁽⁰⁾_{v|u,q} = INIT(u,v,q)
- h⁽ᵗ⁺¹⁾_{v|u,q} = UPD(h⁽ᶠ⁽ᵗ⁾⁾_{v|u,q}, AGG({MSG_r(h⁽ᵗ⁾_{w|u,q}, z_q) | w∈N_r(v), r∈R}), READ({h⁽ᵗ⁾_{w|u,q} | w∈V}))

con f la función de historia (f(t)=t estándar, f(t)=0 recupera una generalización de NBFNet). Se requiere **target node distinguishability**: INIT(u,u,q) ≠ INIT(u,v,q) para v≠u (relacionado al *Labeling Trick* de Zhang et al., pero aplicado solo al nodo fuente, no al par).

**Espacio de diseño explorado:** 3 inicializaciones (INIT1 sin query, INIT2 con embedding de query z_q, INIT3 con ruido gaussiano ε_u por nodo), 2 agregaciones (sum, PNA), 3 funciones de mensaje (MSG1: análoga a NBFNet/query-dependiente, MSG2: análoga a CompGCN, MSG3: análoga a RGCN con descomposición basis), 2 funciones de historia, con/sin readout global.

**Test relational asymmetric local 2-WL (rawl2).** Se define un test de coloreo de pares (u,v) que, a diferencia del (local) k-WL simétrico habitual, solo mira vecinos cambiando la *segunda* coordenada del par:

- rawl⁽ᵗ⁺¹⁾₂(u,v) = τ(rawl⁽ᵗ⁾₂(u,v), {{(rawl⁽ᵗ⁾₂(u,w), r) | w∈N_r(v), r∈R}})

**Teorema 5.1 (caracterización de expresividad):** (1) toda C-MPNN es acotada superiormente por rawl2 (rawl⁽ᵗ⁾₂ ⪯ h⁽ᵗ⁾_q); (2) existe una C-MPNN sin readout global que iguala exactamente a rawl2 para cualquier T y función de historia f — es decir, **la elección de f(t)=t vs. f(t)=0 no afecta la expresividad teórica** (aunque NBFNet usa f(t)=0).

**Caracterización lógica.** Se define la lógica rFO3_cnt (fragmento de FO con 3 variables y counting) y su extensión erFO3_cnt (permite moverse también a no-vecinos). **Teorema 5.2:** un clasificador binario lógico es capturado por C-MPNNs sin readout global **si y solo si** es expresable en rFO3_cnt. **Teorema 5.3:** todo clasificador expresable en erFO3_cnt es capturado por C-MPNNs (con readout global) — el readout global estrictamente aumenta el poder expresivo.

**Jerarquía de expresividad:** rawl2 ⊆ rwl2 (versión simétrica relacional de local 2-WL de Barceló et al.). Se demuestra formalmente que agregar relaciones inversas (rawl2⁺, rwl2⁺) produce tests estrictamente más expresivos — justificando teóricamente una práctica común pero nunca antes cuantificada formalmente.

### Resultados experimentales

Sobre WN18RR y FB15k-237 (splits inductivos v1-v4 de GraIL), 6 capas, 32 dims ocultas, entrenados 20 épocas, INIT2 salvo se indique:

- **Q1 (función de historia):** sin diferencia significativa entre f(t)=t y f(t)=0 para ninguna combinación de agregación/mensaje — confirma empíricamente el Teorema 5.1.
- **Q2 (mensaje/agregación):** en WN18RR (≤11 tipos de relación) no hay diferencias significativas entre funciones de mensaje. En FB15k-237, MSG2_r (tipo CompGCN) rinde peor consistentemente; MSG3_r (tipo RGCN con basis decomposition) es el más robusto, superando incluso a MSG1_r (la función de NBFNet) pese a no tener vector de query aprendible explícito. PNA tiende a ser ligeramente mejor que sum en WN18RR; en FB15k-237 la interacción agregación×mensaje es más compleja (PNA mejor con MSG1_r, sum mejor con MSG2_r/MSG3_r).
- **Q3 (inicialización):** las inicializaciones que no cumplen target node distinguishability rinden significativamente peor (detalle en apéndice).
- **Q4:** C-MPNNs superan a R-MPNNs (RGCN, CompGCN) por márgenes significativos en experimentos transductivos y en un dataset biomédico (apéndice).
- **Q5 (readout):** el readout global no ayuda significativamente en WN18RR y **degrada** el desempeño en 2 splits de FB15k-237. El **readout relation-specific** propuesto (suma separada sobre vecinos con arista entrante/saliente de tipo q) sí produce mejoras consistentes en FB15k-237 (estado del arte en esa configuración), pero no en WN18RR (relacionalmente disperso) — se atribuye a que FB15k-237 tiene muchos más tipos de relación, haciendo útil la información específica de relación.

### Conclusión y limitaciones
El estudio explica formalmente por qué NBFNet y modelos similares funcionan bien: computan invariantes binarios vía variantes locales de tests WL de orden superior. Limitado a tareas binarias (link prediction); no se extiende directamente a hipergrafos relacionales ni a predicción de subestructuras de tamaño k. Los tests/arquitecturas de orden k más alto son computacionalmente costosos.

---

## 5. TRIX — A More Expressive Model for Zero-shot Domain Transfer in Knowledge Graphs

**Zhang, Bevilacqua (Purdue), Galkin (Intel AI Lab), Ribeiro (Purdue). LoG 2024.**

### Motivación
Tres desafíos abiertos en modelos fully-inductive: (1) expresividad limitada de ULTRA (su grafo de relaciones solo cuenta *cuántas* entidades comparten un par de relaciones, no *cuáles*); (2) soporte insuficiente para tareas de **predicción de relación** (h, ?, t) — ULTRA/InGram requieren un forward pass por cada relación candidata; (3) exploración insuficiente de si los LLMs de contexto largo pueden resolver estas tareas directamente.

### Metodología

**Matriz de adyacencia de relaciones enriquecida (A_R).** A diferencia de ULTRA/InGram, que solo cuentan cuántas entidades comparten un par de relaciones (perdiendo *cuáles* entidades), TRIX construye A_R ∈ ℝ^{|R|×|R|×|V|×4}: para cada par de relaciones (r_i, r_j) y cada entidad v_k, se registran 4 roles distinguidos:
- **head-head** (A^hh_R[r_i,r_j,v_k] = E_h[v_k,r_i]·E_h[v_k,r_j]),
- **tail-tail** (A^tt_R),
- **head-tail** (A^ht_R),
- **tail-head** (A^th_R),

donde E_h, E_t ∈ ℝ^{|V|×|R|} cuentan cuántas veces cada entidad es cabeza/cola de tripletas de cada relación. Aunque la forma |R|×|R|×|V|×4 parece grande, es en la práctica una lista adicional de atributos de arista dispersa con overhead marginal.

**Actualización iterativa simultánea (a diferencia de ULTRA, que es secuencial: primero relaciones, luego entidades).** TRIX refina entidades y relaciones **simultáneamente** en cada ronda, usando capas NBFNet (mensaje DistMult, agregación suma, update MLP) tanto sobre A_V como sobre A_R, cada una alimentándose de las representaciones más recientes de la otra.

- **Entity prediction (h,r,?):** inicialización con labeling trick — X⁽⁰⁾_{h,r}(u)=INIT_V(h,u) (vector de unos solo para h), Z⁽⁰⁾_{h,r}(r')=INIT_R(r,r') (vector de unos solo para r). Se itera: X⁽ⁱ⁾ actualiza usando A_V y Z⁽ⁱ⁻¹⁾; Z⁽ⁱ⁾ actualiza usando A_R y X⁽ⁱ⁾ (5 rondas en la implementación).
- **Relation prediction (h,?,t):** inicialización simétrica — X⁽⁰⁾_{h,t}(h)=vector de unos, X⁽⁰⁾_{h,t}(t)=vector de menos-unos, todas las relaciones inicializadas a unos (Z⁽⁰⁾=1^d, sin relación de consulta ya que es justamente lo que se predice). Se itera en orden inverso: primero Z⁽ⁱ⁾ (usa A_R), luego X⁽ⁱ⁾ (usa A_V, con Z⁽ⁱ⁾ ya actualizado) — 3 rondas. Esto permite responder consultas de predicción de relación **en un solo forward pass**, en vez de uno por cada relación candidata como en ULTRA/InGram.

**Resultados teóricos de expresividad (Sección 4.3):**
- **Lema 1:** TRIX es al menos tan expresivo como ULTRA e InGram (ambos son casos particulares de TRIX con hiperparámetros específicos).
- **Lema 2:** existen tripletas no-isomorfas que TRIX distingue pero ULTRA/InGram no (demostrado por construcción de ejemplo).
- **Teorema 1:** TRIX es **estrictamente más expresivo** que ULTRA e InGram.

**Complejidad temporal.** O(|E| + |V|α²) por forward pass de TRIX vs. O(|E|+|V|+|R|²) de ULTRA para entity prediction (donde α es el máximo número de relaciones únicas conectadas a una entidad) — en la práctica ~10× peor que ULTRA en entity prediction, pero ~20× mejor en relation prediction (donde ULTRA necesita O((|E|+|V|+|R|²)|R|), un forward pass por relación).

### Resultados experimentales

Evaluación en **57 KGs** (mismo benchmark de ULTRA), preentrenado en WN18RR, CoDEx-Medium, FB15k-237; NBFNet layers con dim 32; 10 épocas de preentrenamiento (10.000 pasos/época), 3 épocas de fine-tuning; se entrena un modelo TRIX separado para entity prediction (5 rondas) y otro para relation prediction (3 rondas).

- **Entity prediction (Tabla 1):** TRIX 0-shot supera a ULTRA 0-shot en promedio ~3% absoluto de MRR (0.390 vs 0.366 total avg), y ~0.7% tras fine-tuning (0.418 vs 0.408). Por categoría: inductivo (e,r) 0.368 vs 0.345; inductivo (e) 0.455 vs 0.431; transductivo 0.339 vs 0.312.
- **Relation prediction (Tabla 2):** mejora más marcada — TRIX 0-shot supera a ULTRA en +7.4% absoluto en Hits@1 (0.687 vs 0.613 total avg); tras fine-tuning +4.7% (0.706 vs 0.659). Notablemente, el TRIX 0-shot ya supera al ULTRA fine-tuned.
- **Por dataset (Figura 2, 40 datasets de entity prediction):** TRIX supera a ULTRA en 34/40, comparable en 14, y peor en 6. Supera a baselines supervisados en 30/40.
- **Meta-learning zero-shot (WikiTopics, 11 dominios de Wikidata-5M):** el desempeño zero-shot mejora monótonamente al aumentar el número de dominios de preentrenamiento (Tabla 3: Hits@10 promedio pasa de 0.413 con 1 dominio a 0.483 con 4 dominios) — los embeddings de relación se vuelven más distintivos y expresivos con más dominios (visualizado con heatmaps de similitud coseno, Figura 3).
- **Comparación con LLMs (Gemini-1.5-Pro/Flash, en CoDEx-S, 30 muestras):**
  - *Task 1 (in-domain, nombres naturales):* el LLM predice relaciones casi tan bien como TRIX y mejor que ULTRA — pero esto se debe a información semántica de preentrenamiento, no a razonamiento estructural.
  - *Task 2 (out-of-domain, entidades/relaciones vecinas reemplazadas por tokens metasintácticos tipo foo/bar/baz):* el desempeño del LLM cae drásticamente — no logra razonar puramente sobre la estructura del grafo.
  - *Task 3 (doble-equivariancia, IDs numéricos permutados):* el LLM es muy sensible a la permutación de IDs — predicciones consistentes en solo 38.5% de las queries a través de 3 corridas, mientras que los modelos gráficos doblemente equivariantes (TRIX/ULTRA) dan desempeño consistente por diseño.
  - Conclusión: benchmarks de contexto largo tipo "needle in a haystack" pueden sobreestimar la capacidad real de razonamiento estructural de los LLMs.

### Conclusión
TRIX demuestra que mayor expresividad estructural (distinguir *qué* entidades comparten relaciones, no solo *cuántas*) se traduce en mejor desempeño empírico en 57 KGs, y que su esquema de actualización simultánea entidad-relación resuelve de forma nativa y eficiente la predicción de relaciones, una limitación clara de ULTRA e InGram.
