# A Survey of Graph Transformers: Architectures, Theories and Applications

**Autores:** Chaohao Yuan (The Chinese University of Hong Kong; SIGS, Tsinghua University), Kangfei Zhao (The Chinese University of Hong Kong), Ercan Engin Kuruoglu (SIGS, Tsinghua University; ISTI-CNR), Liang Wang (Institute of Automation, Chinese Academy of Sciences), Tingyang Xu (DAMO Academy, Alibaba Group; Hupan Lab), Wenbing Huang (Renmin University of China), Deli Zhao (DAMO Academy, Alibaba Group; Hupan Lab), Hong Cheng (The Chinese University of Hong Kong), Yu Rong* (DAMO Academy, Alibaba Group; Hupan Lab)

\* Yu Rong es el autor de correspondencia.

**Publicación:** ACM Comput. Surv., Vol. 1, No. 1, Article, Julio 2026. DOI: https://doi.org/10.1145/3834764
**Licencia:** Creative Commons Attribution 4.0 International License. © 2026 los autores.
**arXiv:** 2502.16533v3 [cs.LG], 27 Jul 2026

**CCS Concepts:** Computing methodologies → Machine learning algorithms; Artificial intelligence; Networks → Network design principles; Theory of computation → Graph algorithms analysis.

**Additional Key Words and Phrases:** Graph Transformers, Graph Neural Networks

---

## Abstract

Graph Transformers (GTs) han demostrado una fuerte capacidad para modelar estructuras de grafos abordando las limitaciones intrínsecas de las Graph Neural Networks (GNNs), como el over-smoothing y el over-squashing. Estudios recientes han propuesto arquitecturas diversas, explicabilidad mejorada y aplicaciones prácticas para los GTs. A la luz de estos rápidos desarrollos, se realiza una revisión exhaustiva de los GTs, cubriendo aspectos como sus arquitecturas, fundamentos teóricos y aplicaciones. En este survey, primero se categoriza la arquitectura de los GTs según sus estrategias para procesar información estructural, incluyendo tokenización de grafos, codificación posicional, atención estructural (structure-aware attention) y ensamble de modelos. Luego, desde la perspectiva teórica, se examina la expresividad de los GTs en las diversas arquitecturas discutidas y se contrastan con otros algoritmos avanzados de aprendizaje en grafos para descubrir sus conexiones. Para las aplicaciones, se organiza la literatura en torno a cuatro formas de organización de grafos: relacional, geométrica, dinámica y heterogénea. Una tabla de Guía Práctica mapea luego los componentes arquitectónicos a estas formas de grafos según la frecuencia de adopción, de modo que los practicantes puedan reducir qué familias de diseño considerar para una estructura de entrada dada. Finalmente, se discuten los desafíos actuales y las direcciones prospectivas en Graph Transformers para investigación futura.

---

## 1. Introducción

Los datos de grafos, una estructura de datos no euclidiana, se encuentran comúnmente en diversas aplicaciones del mundo real, incluyendo datos moleculares, interacciones de proteínas y redes sociales. Recientemente, las Graph Neural Networks (GNNs) [82] han demostrado capacidades impresionantes para modelar dichos datos. Un paradigma representativo para construir GNNs es el message-passing [51], que agrega iterativamente la información de los vecinos y actualiza el embedding de los nodos. Sin embargo, el paradigma de message-passing enfrenta varias limitaciones intrínsecas, como el over-smoothing [13, 130] y el over-squashing [83], lo que dificulta que las GNNs capturen efectivamente las dependencias de largo alcance dentro de los grafos.

En otra línea de investigación, el modelo Transformer [144] ha demostrado un rendimiento notable en diversas modalidades, incluyendo lenguaje natural [144], imágenes [35], video [35] y series de tiempo [157]. Una ventaja notable del modelo Transformer es su capacidad de capturar efectivamente dependencias de largo alcance, convirtiéndolo en una solución factible para abordar las limitaciones inherentes de las GNNs. Los Graph Transformers (GTs) adaptan la arquitectura Transformer para manejar tanto embeddings de nodos como estructuras de grafos, demostrando un rendimiento superior comparado con las GNNs de message-passing en una variedad de tareas, incluyendo la predicción de propiedades moleculares [129, 142, 173], simulación de dinámica molecular [46, 90, 91] y generación de grafos [148].

En este survey, se realiza una revisión sistemática y exhaustiva de los avances recientes en GTs, examinando los desarrollos desde las perspectivas de arquitecturas, teorías y aplicaciones. Primero, desde la perspectiva de la arquitectura, se categorizan los GTs en cuatro categorías según su forma de integrar estructuras de grafos en los Transformers:

1. **Tokenización Multi-nivel de Grafos (Multi-level Graph Tokenization).** Estos modelos utilizan mecanismos de tokenización para representar bordes, subgrafos y hops como tokens estructurales, permitiendo que el mecanismo de atención capture y aprenda efectivamente las relaciones topológicas intrínsecas.
2. **Codificación Posicional Estructural (Structural Positional Encoding).** Estos modelos mejoran la codificación posicional (PE), tradicionalmente usada para denotar relaciones de secuencia de tokens, para dilucidar las interrelaciones estructurales entre tokens.
3. **Mecanismos de Atención Estructural (Structure-aware Attention Mechanisms).** Dado que la matriz de atención captura inherentemente las relaciones aprendidas entre tokens, estos modelos modifican la matriz de atención usando información estructural derivada del grafo para incorporar interrelaciones entre nodos.
4. **Ensamble de Modelos entre GNNs y Transformers (Model Ensemble).** Además de la modificación directa de las arquitecturas de los Transformers, otra estrategia efectiva emplea GNNs de message-passing para codificar información estructural, seguida de la integración de GNNs con Transformers. Se sigue la taxonomía del estudio previo [113] para organizar esta categoría.

Adicionalmente, se investigan dos aspectos críticos de la arquitectura de Graph Transformer: los desafíos de escalabilidad en el procesamiento de grafos a gran escala y los diseños arquitectónicos especializados para grafos geométricos.

Después de revisar las arquitecturas de los GTs, es importante determinar qué arquitectura es más poderosa en general o en una tarea específica, ya que diferentes arquitecturas incluyen diferentes sesgos inductivos o modelan el grafo en distinta granularidad. Para ello, se investiga la capacidad expresiva de los GTs. Específicamente, se examinan los estudios que comparan la expresividad de los GTs usando el Test de Weisfeiler-Lehman. Estos conocimientos teóricos mejorarán la comprensión de cómo cada componente contribuye a la capacidad de los GTs para aprender datos de grafos. Además, se discuten las relaciones entre los GTs y otros algoritmos actuales de aprendizaje en grafos, lo que puede aclarar aún más las fortalezas y debilidades de los GTs.

Además, para revisar los GTs adaptados a tareas específicas, se organizan sus aplicaciones en cuatro formas de organización de grafos: grafos relacionales con semántica discreta, grafos geométricos y periódicos, grafos dinámicos y flujos de eventos, y grafos heterogéneos o multimodales. Para cada forma de organización, se revisan las plantillas de tareas comunes, las señales estructurales dominantes y las elecciones de diseño de GT que aparecen repetidamente en diferentes dominios. Esta organización evita introducir repetidamente definiciones de tareas similares para cada dominio y ayuda a conectar las aplicaciones con las arquitecturas de GT correspondientes. Además, aplicar GTs a tareas específicas puede requerir estrategias de pre-entrenamiento adicionales u objetivos de optimización, como la difusión. Estos aspectos también se incluyen en la revisión. En la última parte de este survey, se discuten direcciones futuras potenciales para los GT.

Este survey busca proporcionar un análisis exhaustivo de los GTs desde múltiples perspectivas, incluyendo sus arquitecturas de modelo, expresividad y aplicaciones. Mientras que revisiones recientes se han centrado principalmente en aspectos arquitectónicos [113, 117], este trabajo se distingue en tres aspectos: (1) perspectivas arquitectónicas exhaustivas, (2) análisis sistemático de la expresividad teórica de los GTs, (3) investigación extensa de aplicaciones cross-domain. Más importante aún, estas tres perspectivas se conectan a través del marco unificado en la Figura 2, que proporciona una manera más sistemática de entender las elecciones de diseño de GT y sus capacidades correspondientes. También se exploran direcciones de investigación prospectivas con avances recientes en GTs.

**Roadmap.** El resto del paper está organizado de la siguiente manera. La Sección 2 introduce los preliminares. La Sección 3 revisa las cuatro categorías de GTs desde la perspectiva de arquitectura. En la Sección 4, se resume la capacidad expresiva de los GTs y se conecta con las metodologías de aprendizaje en grafos. La Sección 5 revisa las aplicaciones de los GTs. Se discute la situación actual del aprendizaje en grafos y se señalan las direcciones de investigación futuras en la Sección 6, y se concluye el paper en la Sección 7.

### Figura 1 (descripción)

La figura muestra el desarrollo de las arquitecturas de Graph Transformer a lo largo del tiempo (2020-2026), donde el color azul indica ensamble de GNNs y Transformers, rojo indica atención estructural, verde indica codificación posicional estructural y amarillo indica tokenización multi-nivel:

- **2020:** GROVER, SE(3)-Trans
- **2021:** Graphormer, Laplacian PE, SAN, GraphiT
- **2022:** SAT, EGT, GMT, GKAT, TokenGT, GraphGPS, NodeFormer, TorchMD-Net
- **2023:** NAGphormer, SGFormer, GRIT, Graphormer-GD, Equiformer, Exphormer, Graph ViT/MLP-Mixer
- **2024:** EquiformerV2, Polynormer, CoBFormer, TGT, GraphGPT
- **2025:** DUALFormer, Primphormer, SwapGT, Q-GT
- **2026:** FACET, RELGT, HubGT, EquiformerV3, ParaFormer

La figura inferior ilustra aplicaciones representativas de GT bajo cuatro formas de organización de grafos (relacional, geométrico, dinámico y multimodal): Molécula (predicción de propiedades — ligand binding affinity, BBBP, toxicidad; generación 2D/3D), Proteína (docking proteína-proteína, predicción de propiedades — soluble, contact; diseño de proteínas), Texto (comprensión multimodal, generación, comprensión), Imagen (procesamiento 3D, clasificación), Social (recomendación, detección de rumores, citación), Dinámico (predicción de eventos, trayectoria) y Otro (cerebro, dependencias de largo alcance, material cristalino).

---

## 2. Preliminares

Un grafo se denota como G = (V, E), donde V representa el conjunto de nodos y E representa el conjunto de bordes. El conjunto de nodos V comprende *n* nodos, con matriz de características H ∈ ℝ^(n×d), donde *n* es el número de nodos y *d* es la dimensión de las características de los nodos. El conjunto de bordes E corresponde a una matriz de adyacencia A^G ∈ ℝ^(n×n), donde A^G_(u,v) = 1 si existe un borde (u,v) en E, y 0 en caso contrario.

Estas notaciones son adecuadas para representar un grafo básico con matriz de adyacencia y características de nodos. En escenarios más desafiantes, los grafos están asociados con más características para soportar diversas aplicaciones:

1. **Grafo Geométrico:** además de las características de propiedad de los nodos, los nodos tendrán características de coordenadas X⃗ ∈ ℝ³, que se manejarán independientemente para lograr propiedades de invariancia o equivariancia.
2. **Grafo con características de borde:** en lugar de simplemente convertir la información del borde en una matriz de adyacencia, los bordes ofrecen información suplementaria que denota relaciones más específicas entre los nodos.
3. **Grafo dinámico:** el grafo adicionalmente contiene el dominio temporal T comparado con grafos estáticos, representado como G = (V, E, T). Esto indica que los nodos y bordes pueden cambiar en el tiempo. El grafo se denota como G = {(v_i, v_j, τ)_n, n = 1,2,...,|E|}, donde cada tupla (v_i, v_j, τ) representa un borde entre el nodo v_i y el nodo v_j en un tiempo específico τ ∈ T, y E especifica el conjunto de bordes para ese paso temporal particular.

### Message Passing Graph Neural Networks

Las graph neural networks [82] son un marco fundamental para aprender representaciones en grafos vía el mecanismo de message passing [51]. Dados los embeddings de nodos H y la matriz de adyacencia A^G, una message passing neural network (MPNN) φ_θ actualiza los embeddings de nodos propagando información de los nodos vecinos:

```
m_ij = φ_msg(H_i, H_j, e_ij),    H_i = φ_upd(H_i, {m_ij}_{j∈N_i})
```
*(Ecuación 1)*

donde H y e representan embeddings de nodos y bordes, respectivamente. φ_msg(·) calcula el mensaje recibido por el nodo v_i de sus vecinos, mientras φ_upd(·) actualiza el embedding del nodo con los mensajes.

### Transformers

Los Transformers han demostrado un éxito notable en diversos campos, logrando alto rendimiento en procesamiento de lenguaje natural [32] y aplicaciones de visión por computadora [35]. Los Transformers emplean una arquitectura encoder-decoder, donde el bloque constructor es el mecanismo de self-attention [144]. El módulo de self-attention opera sobre una secuencia de *n* tokens, H ∈ ℝ^(n×d), representable por una función de transformación φ_θ(·): ℝ^(n×d) → ℝ^(n×d):

```
Q = HW_Q,  K = HW_K,  V = HW_V,
A = Softmax(QK^T / √d),
H̃ = AV
```
*(Ecuación 2)*

donde H se transforma primero en matrices query, key y value, Q, K, V ∈ ℝ^(n×d), mediante matrices de pesos lineales W_Q, W_K, W_V ∈ ℝ^(d×d). Luego, la matriz de atención A ∈ ℝ^(n×n) se calcula por el producto interno de query y key, seguido de la normalización de una función Softmax(·). Aquí, A_ij ∈ [0,1] indica la influencia de H_j sobre H_i. La matriz de atención se aplica finalmente a la matriz de valores para generar nuevos embeddings de los tokens H̃ ∈ ℝ^(n×d). Interpretando la matriz de atención A como una matriz de adyacencia alternativa, el mecanismo de self-attention puede verse como una variante de MPNNs con una matriz de adyacencia totalmente conectada:

```
m_ij = A_ij V_j,   H_i = Σ_{j=0}^{n} m_ij
```
*(Ecuación 3)*

Tras aplicar self-attention, los Transformers emplean redes feed-forward elemento a elemento (FFN), que comprenden dos capas lineales con activación ReLU:

```
FFN(H̃) = max(0, H̃W_1 + b_1)W_2 + b_2
```
*(Ecuación 4)*

donde W_1 ∈ ℝ^(d×d1), W_2 ∈ ℝ^(d1×d) son parámetros aprendibles. Además, las subcapas en la FFN incorporarán conexiones residuales y layer normalization [2].

En la práctica, la multi-head attention (MHA) se utiliza ampliamente en los Transformers para mejorar el aprendizaje de representaciones. Precisamente, las matrices Q, K, V obtenidas de la Ecuación (2) se dividen en H cabezas independientes, denotadas Q^(h), K^(h) y V^(h), respectivamente. El mecanismo MHA calcula representaciones como:

```
H̃ = ||_{h=1}^{H} Softmax(Q^(h)K^(h)T / √d_h) V^(h)
```
*(Ecuación 5)*

donde d_h = d/H representa la dimensión asignada a cada cabeza. MHA adquiere representaciones desde varias perspectivas dentro de cada cabeza y concatena los embeddings de cada cabeza a la representación final.

Otro componente crítico en los Transformers es la codificación posicional (positional encoding), diseñada para codificar la posición relativa de los tokens en una entrada secuencial. El paper original de Transformer [144] introduce una PE sinusoidal libre de parámetros. Sin embargo, avances posteriores, como la PE aprendida [50] y la PE rotatoria [138], han demostrado que la elección de la PE impacta significativamente el rendimiento de los Transformers, resaltando su importancia.

---

## 3. Arquitecturas

El Transformer vanilla es esencialmente una GNN especial, donde los mecanismos de self-attention entre todos los nodos operan en un grafo completamente conectado [117]. Dado que los Transformers ignoran la estructura de grafo original, el objetivo central de un GT es incorporar información de bordes en la arquitectura Transformer. Con base en los enfoques para incorporar este prior estructural, en esta sección se categorizan sistemáticamente los GTs existentes en cuatro clases:

1. Tokenización Multi-nivel de Grafos (Sección 3.1)
2. Codificación Posicional Estructural (Sección 3.2)
3. Mecanismos de Atención Estructural (Sección 3.3)
4. Ensamble de Modelos entre GNNs y Transformers (Sección 3.4)

La Figura 2 proporciona una visión general de las arquitecturas, mientras que la Tabla 1 resume las elecciones de diseño representativas. Bajo el marco unificado de la Figura 2, esta sección se centra en la perspectiva arquitectónica: la Tabla 1 registra qué priors estructurales incorpora cada GT, la Sección 4 analiza la capacidad expresiva implicada por estas elecciones, y la Sección 5 revisa su uso en diferentes formas de organización de grafos. Aparte de arquitecturas de modelo novedosas, las aplicaciones prácticas de los GTs requieren alta escalabilidad o equivariancia. Por lo tanto, también se discute el progreso avanzado en la mejora de la escalabilidad (Sección 3.5) y el logro de equivariancia (Sección 3.6) para los GTs.

### Figura 2 (descripción)

La figura ilustra la arquitectura general de Graph Transformers: un "General Graph Transformer" con bloques FFN, Multi-Head Attention (con codificación posicional absoluta/relativa) y GNN, alimentado por un Tokenizer que procesa la matriz de adyacencia. Se muestran cuatro submódulos:

- **Multi-level Tokenizer:** Node Token, Edge Token (TokenGT), Subgraph Token (SAT), K-hop neighbor Token (NAGphormer)
- **Positional Encoding:** Absolute PE (Laplacian PE), Relative PE (Spatial Encoding)
- **Modify Attention:** vía bias (Graphormer) o máscara (HetGT), sobre la matriz de atención y adyacencia (conectado/no conectado)
- **Ensemble Transformers and GNNs (GROVER):** combinaciones secuenciales/paralelas de bloques Trans y GNN

### Tabla 1: Resumen de arquitecturas de GT existentes

Para cada modelo se documenta la tokenización de grafo, la implementación de codificación posicional (PE), la utilización de atención estructural (Attn) y la inclusión de un MPNN adicional (Ens). Abreviaturas: PE: Positional Encoding, Attn: Attention, Ens: ensemble, Sub.: Sub-Graph, K-hop: K-hop Neighbor, RW: Random Walk, Deg.: Degree, Lap: Laplacian.

| Modelo | Token | PE | Attn | Ens |
|---|---|---|---|---|
| SE(3)-Transformer [46] | Node | | | ✓ |
| Graphormer [173] | Node | Deg. | | ✓ |
| Dwivedi et al. [36] | Node | Lap. | | ✓ |
| SAN [83] | Node | Lap. | | ✓ |
| GraphiT [112] | Sub. | RW | | ✓ |
| TokenGT [81] | Edge | Lap. | | |
| SAT [14] | Sub. | Hybrid | | ✓ |
| TorchMD-Net [142] | Node | | | ✓ |
| GMT [114] | Node | | | ✓ |
| GraphGPS [128] | Node | Hybrid | | ✓ |
| GKAT [26] | Node | | | ✓ |
| NodeFormer [160] | Node | | | ✓ |
| EGT [69] | Edge | SVD | | ✓ |
| Exphormer [137] | Node | Lap. | | ✓ |
| NAGphormer [16] | K-hop | | | |
| GRIT [105] | Node | Deg. | | |
| Graph ViT/MLP-Mixer [58] | Sub. | RW | | |
| SGFormer [161] | Node | | | ✓ |
| Graphormer-GD [182] | Node | | | ✓ |
| Equiformer [90] | Node | | | ✓ |
| EquiformerV2 [91] | Node | | | ✓ |
| Polynormer [31] | Node | Lap. | | ✓ |
| CoBFormer [164] | Sub. | | | ✓ |
| TGT [70] | Edge | | | ✓ |
| GraphGPT [48] | Edge | | | ✓ |
| DUALFormer [210] | Node | | | ✓ |
| Primphormer [57] | Node | | | ✓ |
| ParaFormer [178] | Node | | | ✓ |

### 3.1 Tokenización Multi-nivel de Grafos

En el ámbito de los GTs, la tokenización juega un papel crucial en la transformación de datos de grafo a un formato secuencial para su procesamiento por Transformers. A diferencia de los Transformers basados en texto convencionales donde la tokenización es trivial, los datos de grafo presentan desafíos únicos debido a su complejidad estructural. Esta sección explora cuatro niveles distintos de tokenización en GTs de fino a grueso: a nivel de nodo, de borde, de hop y de subgrafo.

#### 3.1.1 Tokenización a Nivel de Nodo

La tokenización a nivel de nodo es el enfoque más granular [53, 173], que trata a cada nodo del grafo como un token individual. Además, el nodo involucrado en la atención puede seleccionarse mediante métodos como el contrastive learning [19]. Este enfoque es particularmente efectivo cuando los modelos se enfocan en características específicas de nodos o cuando la topología del grafo es menos crítica que los atributos de nodos individuales. Al capturar información detallada sobre cada nodo, la tokenización a nivel de nodo es adecuada para tareas como la clasificación de nodos.

#### 3.1.2 Tokenización a Nivel de Borde

La tokenización a nivel de borde extiende el concepto de tokenización a las conexiones entre nodos [48, 81]. Aquí, cada borde del grafo se trata como un token, haciendo este enfoque ideal para tareas donde las interacciones entre nodos son de interés primario. La tokenización a nivel de borde puede capturar la dinámica de estas interacciones, esencial para tareas como predicción de enlaces o comprensión del flujo de información a través del grafo. Al enfocarse en bordes, este enfoque puede resaltar la importancia de los patrones de conectividad en el grafo.

#### 3.1.3 Tokenización a Nivel de Subgrafo

La tokenización a nivel de subgrafo trata un parche local inducido como la representación de token [14], de modo que cada token resume el arreglo interno de nodos y bordes dentro de ese parche en lugar de solo los atributos de un nodo central. Este diseño es efectivo cuando la topología de una estructura local es en sí misma informativa, por ejemplo cuando motivos, ciclos, clusters u otros patrones de orden superior importan. Notablemente, los nodos en subgrafos correspondientes a diferentes tokens pueden superponerse, permitiendo una representación flexible y comprehensiva del grafo.

#### 3.1.4 Tokenización a Nivel de Hop

La tokenización a nivel de hop mantiene un nodo ancla central y construye una secuencia ordenada de tokens a partir de sus vecindarios en diferentes radios [16-18, 186]. Su señal clave es por lo tanto la distancia al ancla y la progresión de 1-hop a 2-hop y contexto de orden superior, en lugar de la identidad completa de un subgrafo inducido. Por lo tanto, la tokenización a nivel de hop no debe considerarse simplemente como un caso especial de tokenización a nivel de subgrafo con un k diferente: los tokens de subgrafo enfatizan la topología interna de un parche, mientras que los tokens de hop enfatizan cómo se organiza la información a través de múltiples capas de distancia. Además, en lugar de realizar self-attention de complejidad cuadrática entre todos los nodos, la tokenización a nivel de hop aplica self-attention a la secuencia de hop de cada nodo. Este diseño permite el entrenamiento por mini-batch y hace el modelo escalable a grafos grandes.

### 3.2 Codificación Posicional Estructural

En los Transformers estándar, el módulo de codificación posicional indica la posición de los tokens en una secuencia. Para extender este módulo a los GTs, es natural desarrollar métodos para representar los embeddings posicionales de los nodos en un grafo. Dado que los enfoques de tokenización a nivel de borde, hop y subgrafo ya incorporan información estructural, en esta sección se asume que los métodos de PE se aplican a nivel de nodo.

Los métodos de PE existentes pueden categorizarse en PE absoluta y PE relativa. Similar a los Transformers estándar, la PE absoluta asigna un embedding posicional único a cada nodo. Este embedding, aprendido o libre de parámetros, se agrega o concatena posteriormente con el embedding original del nodo. En contraste, la PE relativa se enfoca en capturar relaciones pairwise entre nodos y se aplica directamente a la matriz de atención. Por lo tanto, la discusión de PE relativa se deja para la siguiente sección.

El objetivo principal de la PE absoluta puede formularse como utilizar una función *f* para extraer la información estructural subyacente del grafo, típicamente de la matriz de adyacencia A:

```
P = f(A),   H̃ = g(H, P)
```
*(Ecuación 6)*

Aquí, la función *f* extrae la PE absoluta P ∈ ℝ^(n×dp), donde *n* denota el número de nodos y *d_p* representa la dimensión del embedding posicional para cada nodo. La función *g* integra la PE absoluta con las características originales del nodo H, ya sea por concatenación o empleando un MLP para alinear las dimensiones de P y H antes de sumarlas.

**Laplacian PE** aprovecha los eigenvectores y eigenvalores obtenidos de la descomposición de la matriz Laplaciana como PE:

```
UΛU^T = I − D^(−1/2) A D^(−1/2)
```
*(Ecuación 7)*

donde D denota la matriz de grados, I es la matriz identidad, Λ y U representan eigenvalores dispuestos en una matriz diagonal y eigenvectores. Dado que el signo de los eigenvectores precalculados es arbitrario, los enfoques de Laplacian PE ajustan aleatoriamente el signo de los eigenvectores durante el entrenamiento. La primera Laplacian PE [36] propone utilizar los *k* eigenvectores no triviales más pequeños de un nodo como su PE. Otro trabajo, SAN [83], introduce una Laplacian PE aprendida. Para un nodo dado v_j, SAN usa {λ_i, U_ij}_{i=0}^m como características de entrada para redes neuronales para aprender la PE del nodo v_j, donde m es un hiperparámetro que determina el número de eigenvectores considerados.

A pesar de la efectividad de la Laplacian PE, enfrenta dos desafíos subyacentes:

1. **Descomposiciones propias no únicas.** Hay diferentes eigendescomposiciones del mismo Laplaciano. Si un vector v es un eigenvector, entonces −v también lo es. Hay soluciones no únicas para eigenvectores con multiplicidades de eigenvalores.
2. **Sensibilidad a perturbaciones.** Perturbaciones menores en la estructura del grafo pueden afectar significativamente el resultado de los eigenvectores, generando inestabilidad considerable en la Laplacian PE.

Para abordar el primer desafío, **SignNet** [92] introduce una red sign-invariant, *f*, que opera sobre eigenvectores como:

```
f(U_1,...,U_k) = ρ(||_{i=1}^{k} [φ(U_i) + φ(−U_i)])
```
*(Ecuación 8)*

donde ρ y φ son redes neuronales. Esta formulación asegura que la red neuronal permanezca invariante a los embeddings respecto del signo de los eigenvectores. Además, para abordar la ocurrencia de múltiples elecciones de eigenvectores cuando hay eigenvalores repetidos en la matriz Laplaciana, **BasisNet** [92] propone un método para extraer PE consistente de estas matrices.

Para abordar el desafío de estabilidad en Laplacian PE, se introduce **Stable and Expressive PE (SPE)** [68], formulada como:

```
P(U, Λ) = ρ(U(φ_1(Λ))U^T, U(φ_2(Λ))U^T, ..., U(φ_m(Λ))U^T)
```
*(Ecuación 9)*

donde la entrada consiste en los *k* eigenvalores más pequeños λ y sus eigenvectores correspondientes V. En lugar de implementar una división estricta de eigensubespacios, SPE utiliza una agregación ponderada de eigenvectores contingente a los eigenvalores para asegurar estabilidad.

**Singular Value Decomposition (SVD) PE** [69] proporciona un alcance más amplio de aplicaciones comparado con Laplacian PE, ya que puede manejar grafos dirigidos y ponderados. La SVD PE se calcula por:

```
A^SVD ≈ UΣV^T = (U√Σ)·(V√Σ)^T = Û V̂^T,   P = Û ∥ V̂
```
*(Ecuación 10)*

donde U, V ∈ ℝ^(n×r) contienen los r vectores singulares izquierdo y derecho como sus columnas respectivas, cada uno asociado con los valores singulares más altos r en la matriz diagonal Σ ∈ ℝ^(r×r). Similar a Laplacian PE, la SVD PE involucra el volteo aleatorio de signo de los eigenvectores durante el entrenamiento. En consecuencia, construyendo sobre el concepto de SignNet, desarrollar una SVD PE sign-invariant podría ser una dirección potencial para investigación futura.

**Random Walk PE (RWPE)** [38] representa una PE derivada del proceso de difusión de un random walk. La RWPE para un nodo v_i se puede expresar mediante un random walk de k pasos:

```
P_i = (M_ii, M²_ii, ..., M^k_ii) ∈ ℝ^k
```
*(Ecuación 11)*

donde M = AD^(−1) representa el operador de random walk. A diferencia de Laplacian PE, RWPE no sufre de ambigüedad de signo. Bajo la condición de que cada nodo posea un vecindario topológico k-hop único para un k suficientemente grande, RWPE proporciona una representación de nodo distintiva. Como estudio futuro potencial, los investigadores pueden explorar el reemplazo de la difusión de random walk por procesos de difusión de grafo alternativos para derivar PE.

**Graphormer** [173] introduce un método heurístico que aprovecha los grados de los nodos para la codificación de centralidad. Específicamente, cada nodo recibe un vector aprendible basado en su grado, que se incorpora luego a las características del nodo como la capa de entrada:

```
h_i^(0) = x_i + z⁻_{deg⁻(v_i)} + z⁺_{deg⁺(v_i)}
```
*(Ecuación 12)*

donde z⁻, z⁺ ∈ ℝ^d representan vectores de embedding aprendibles definidos respectivamente por el grado de entrada deg⁻(v_i) y el grado de salida deg⁺(v_i). Para grafos no dirigidos, deg⁻(v_i) y deg⁺(v_i) se simplifican a deg(v_i). Al incorporar la codificación de centralidad en los componentes query y key del mecanismo de atención, Graphormer mejora la capacidad de la atención para reconocer efectivamente tanto la importancia como las relaciones entre nodos.

La información de grado también puede inyectarse como post-procesamiento. Por ejemplo, **GRIT** [105] actualiza la representación del nodo después del Transformer vía:

```
x_i^{out'} := x_i^{out} ⊙ θ_1 + (log(1 + D_i) · x_i^{out} ⊙ θ_2)
```
*(Ecuación 13)*

donde θ_1, θ_2 son pesos aprendibles. Similarmente, **SAT** [14] también incorpora información de grado en la conexión residual como: x_i^{out'} = x_i + 1/√D_i · x_i^{out}.

### 3.3 Mecanismos de Atención Estructural

### Tabla 2: Resumen de métodos de mecanismo de atención estructural

e_ij son las características del borde, c_ij representa el camino más corto entre el nodo i y el nodo j, M = AD^(−1) y D es la matriz de grados, R indica la matriz de distancia de resistencia y L es la matriz Laplaciana.

| | Attention Bias | | | Attention Mask | | |
|---|---|---|---|---|---|---|
| **Método** | Graphormer [173] | Graphormer-GD [182] | GRIT [105] | GT [36] | HetGT [172] | GraphiT [112] |
| **Término Bias/Mask** | MLP(e_ij)+c_ij | MLP(D_ij) | ‖_{k=n}(M^k_ij) | MLP(e_ij) si A^G_ij=1, si no, 0 | MLP(e_ij) si A^G_ij=1, si no, 0 | e^(−βL) & (I−γL)^p |

En los bloques Transformer, la matriz de atención gobierna las interacciones entre nodos, mientras que la tokenización y la PE absoluta aumentan los embeddings de nodo. Estos embeddings aumentados permiten a los Transformers incorporar el prior estructural en los mecanismos de atención. En este sentido, la modificación directa de la matriz de atención es un enfoque más directo para capturar el sesgo inductivo estructural.

Ajustar la matriz de atención comienza con capturar interacciones pairwise entre nodos en el grafo. Para esto, GT aprovecha la estructura del grafo para primero generar una matriz de estructura que codifica patrones de conectividad de nodos. Esta matriz de estructura puede integrarse en la matriz de atención de tres maneras: por attention bias, por attention mask y por tokenización a nivel de borde. Los enfoques más prevalentes que integran información estructural en el mecanismo de atención se resumen en la Tabla 2.

#### 3.3.1 Matriz de Estructura como Attention Bias

Mediante attention bias, la información estructural se incorpora al mecanismo de atención agregando una matriz de bias b ∈ ℝ^(n×n) al producto interno de las matrices query y key:

```
A = Softmax((HW_Q)(HW_K)^T / √d + b)
```
*(Ecuación 14)*

donde el attention bias b se especifica por diferentes enfoques, siendo esencialmente una PE relativa. La PE relativa se calcula a partir de la estructura del grafo, buscando entender interacciones pairwise entre nodos. Se define una matriz de relación P̂ ∈ ℝ^(R×R) como el attention bias b. P_ij está determinado por la función φ(H_i, H_j, e_ij), que codifica las relaciones entre cualquier par de nodos, utilizando sus embeddings H_i, H_j y opcionalmente incorporando el embedding de borde e_ij.

**Graphormer** [173] introduce la distancia del camino más corto (SPD) de un camino más corto SP_ij = [e_1, e_2, ..., e_N] conectando v_i a v_j en el mecanismo de atención. Graphormer incorpora dos tipos de attention bias. El primer bias espacial φ(v_i, v_j) codifica la longitud de SP_ij, y el segundo, edge encoding c_ij, agrega los embeddings de borde en SP_ij:

```
A_ij = Softmax((H_iW_Q)(H_jW_K)^T / √d + b_φ(v_i,v_j) + c_ij),
c_ij = (1/N) Σ_{n=1}^{N} x_en (w_n^E)^T
```
*(Ecuación 15)*

donde b_φ(v_i,v_j) es un escalar aprendible indexado por φ(v_i, v_j) y permanece consistente en todas las capas. x_en y w_n^E denotan la característica del n-ésimo borde e_n en SP_ij y su peso correspondiente, respectivamente.

Además, basado en SPD, **HDSE** [102] introduce codificación estructural de distancia jerárquica, capturando distancias estructurales multi-escala y mejorando significativamente la capacidad del Transformer para modelar topologías complejas. Aunque SPD es expresiva, su precómputo O(n³) suele ser prohibitivo para grafos grandes. **HubGT** [88] aborda esto introduciendo una nueva indexación basada en hub labeling, que permite PE relativa basada en SPD eficiente en grafos de escala de millones con una sobrecarga computacional significativamente reducida.

Sin embargo, **Graphormer-GD** [182] identifica que SPD es incapaz de distinguir adecuadamente ciertas perturbaciones en la estructura del grafo. Para abordar esta limitación, Graphormer-GD introduce una PE relativa más robusta basada en la distancia de resistencia (RD). La matriz de atención con esta PE relativa se representa como:

```
A = φ_1(R) ⊙ Softmax(HW_Q(HW_K)^T + φ_2(R))
```
*(Ecuación 16)*

donde R ∈ ℝ^(n×n) representa la matriz de distancia con R_ij = {|SP_ij|, SP_ij}. El análisis teórico demuestra que RD-WL exhibe un poder discriminativo superior comparado con SPD-WL, para diferenciar grafos distancia-regulares no isomorfos.

Extendiendo el attention bias a física específica del dominio, **Si-GT** [64] introduce un mecanismo de atención Intra-Inter Net (IIN) para análisis de circuitos integrados. En lugar de depender únicamente de métricas de distancia, aplica biases de atención distintos para diferenciar conexiones intra-net (representando resistencia de cable a lo largo de un camino de señal) y conexiones inter-net (representando capacitancia de acoplamiento entre cables adyacentes). Esto demuestra cómo el attention bias puede codificar efectivamente interacciones físicas heterogéneas.

**GRIT** [105] introduce un enfoque para aprender la PE relativa mediante la inicialización de probabilidades de random walk como: P_i,j = [I, M, M², ..., M^(K−1)]_i,j ∈ ℝ^K, donde M = AD^(−1) denota la matriz de probabilidad de transición del random walk. La inicialización de P, combinada con procesamiento MLP, se demuestra que aproxima SPD. Además, el graph-diffusion Weisfeiler-Lehman (GD-WL) con P es estrictamente mejor que GD-WL basado en SPD.

#### 3.3.2 Matriz de Estructura como Attention Mask

Un enfoque alternativo para incorporar atención estructural es realizar multiplicación elemento a elemento entre una matriz de atención y una matriz de máscara, en lugar de tratar la matriz de estructura como un bias de atención:

```
A = Softmax((HW_Q)(HW_K)^T / √d ⊙ M)
```
*(Ecuación 17)*

donde M ∈ ℝ^(n×n) representa la matriz de máscara, que puede ser la matriz de adyacencia u otra matriz que codifique la estructura del grafo.

Para integrar información de bordes en la matriz de máscara de un **GT** [36], M_ij se define en función de la característica del borde. Si existe una conexión, M_ij = W_E e_ij, donde e_ij denota la característica del borde conectando el nodo v_i y el nodo v_j, y W_E representa un peso aprendible. M_ij se establece en −∞ si los nodos v_i y v_j no están conectados. Esta máscara de atención opera de manera similar al attention bias, ya que ambos solo cambian los valores de atención entre nodos conectados.

Otra elección clásica para la matriz de máscara M se define por la matriz de adyacencia, donde M_ij = 0 si los nodos v_i y v_j no están conectados. Al truncar los valores de atención para nodos desconectados, la atención se ve forzada a enfocarse en nodos vecinos locales. Aunque esto puede reducir un GT a una GNN respecto a capturar información del vecindario local, **GMT** [114] y **HetGT** [172] emplean máscaras de atención distintas entre diferentes cabezas, obligando al GT a aprender desde diferentes perspectivas de la estructura del grafo.

Similar al attention bias, la PE relativa también puede usarse como máscaras de atención. **GraphiT** [112] propone kernels definidos positivos en grafos como PE relativa para máscaras de atención. Específicamente, GraphiT explota el kernel de difusión M = e^(−βL) y el kernel de random walk de p pasos M = (I−γL)^p, donde L es la matriz Laplaciana, β y p son hiperparámetros. GraphiT demuestra la efectividad de estas PE relativas como máscaras de atención en varios conjuntos de datos.

A pesar de la efectividad de la difusión de grafo clásica, escalarla con el mecanismo de atención estándar a grafos grandes es desafiante debido a la complejidad cuadrática respecto al número de nodos. Además, como máscaras de atención, las PE relativas no son directamente aplicables a Transformers lineales de bajo rango que no construyen explícitamente una matriz de atención. Para esto, **GKAT** [26] propone un nuevo Random Walks Graph-Nodes Kernel (RWGNK) con complejidad sub-cuadrática. RWGNK opera como un enmascaramiento de bajo rango directamente sobre las matrices query, key y value en el mecanismo de atención, evitando el cómputo explícito de la matriz de atención y así evitando complejidad cuadrática.

Para unificar attention bias y attention mask, **EGT** [69] diseña un marco que incorpora ambos como una fórmula general:

```
A = Softmax((HW_Q)(HW_K)^T / √d + E_e) ⊙ G_e
```
*(Ecuación 18)*

donde G_e, E_e ∈ ℝ^(n×n) son embeddings de bordes generados por transformaciones lineales.

#### 3.3.3 Token a Nivel de Borde como Entrada de Atención

La atención a nivel de borde puede explotarse de dos formas. La primera se enfoca en calcular la atención solo mediante tokens de borde para generar representaciones de borde mejoradas, que luego se fusionan con embeddings de nodo usando las técnicas discutidas anteriormente. La segunda forma incorpora tokens de borde y de nodo simultáneamente en el mecanismo de atención para desarrollar atención estructural. Se presentan brevemente los representantes de la primera forma, i.e., **EGT** [69] y **TGT** [70], y los de la segunda forma, i.e., **TokenGT** [81] y **Edgeformers** [74].

**EGT** [69] introduce attention bias y máscara de bordes a nodos. Además, las características de borde pueden actualizarse desde la matriz de atención en cada capa como:

```
E_e = f((HW_Q)(HW_K)^T / √d + E_e)
```
*(Ecuación 19)*

donde f es una función aprendible con capas feed-forward y layer normalization.

**TGT** [70] emplea interacciones triplete de bordes para actualizar aún más los embeddings de borde. Dados e_ij, e_ik y e_jk como los embeddings de borde de los tres bordes (v_i,v_j), (v_i,v_k) y (v_j,v_k) en un triángulo, los vectores query, key y value se calculan mediante proyecciones lineales sobre e_ij, e_ik y e_jk respectivamente. Luego, la atención triplete calcula la matriz de atención y actualiza los embeddings de borde:

```
A_ijk = Softmax_k((1/√d) q_ij · k_jk + b_ik × σ_1(g_ik)),
e_ij = σ_2(Σ_{k=1}^{N} A_ijk v_jk)
```
*(Ecuación 20)*

donde A_ijk denota el peso de atención que el borde (v_i,v_j) asigna al borde (v_j,v_k). σ_1 y σ_2 son dos MLPs, y b_ik, g_ik son dos escalares derivados de transformaciones MLP sobre e_ik.

**TokenGT** [81] calcula la atención entre todos los nodos y bordes concatenando la matriz de entrada como Ĥ = H||E, entonces el cálculo de la matriz de atención se representa como:

```
A = Softmax((ĤW_Q)(ĤW_K)^T / √d)
```
*(Ecuación 21)*

Para incorporar adicionalmente la información estructural, en TokenGT, el embedding del nodo v se combina con un identificador de nodo P_v ∈ ℝ^d mediante la concatenación [H_v, P_v, P_v, T^V], y el embedding de cada borde (u,v) ∈ E se aumenta como [E_(u,v), P_u, P_v, T^E]. Aquí, T^V y T^E son dos identificadores aprendibles para distinguir nodo y borde. Similar a la PE, los identificadores de nodo P pueden definirse vía PE como Laplacian PE.

**Edgeformers** [74] consisten en dos variantes distintas: Edgeformer-E y Edgeformer-N, especializadas en capturar embeddings de borde y de nodo, respectivamente. En este marco, los bordes se representan como datos textuales que comprenden múltiples tokens. Específicamente, Edgeformer-E combina estos tokens de borde con los tokens de sus nodos asociados como entrada, y procesa la entrada usando self-attention. A diferencia de usar todo el grafo como entrada, Edgeformer-N analiza el ego-grafo centrado en el nodo v. Emplea Edgeformer-E para modelar cada borde incidente a v, y luego aplica una función de agregación para generar la representación final del nodo H_v.

### 3.4 Ensamble de Modelos entre GNNs y Transformers

El enfoque más directo para diseñar GTs implica combinaciones estratégicas de GNNs y Transformers, aprovechando tanto patrones de estructura local como relaciones contextuales globales. Como se ilustra en la Figura 2, estas arquitecturas de ensamble pueden dividirse sistemáticamente en cuatro categorías según el posicionamiento relativo de los bloques GNN y Transformer:

1. **GNN-a-Transformer Secuencial:** alimentar la salida de una GNN a un Transformer.
2. **Transformer-a-GNN Secuencial:** alimentar la salida de un Transformer a una GNN.
3. **Intercalar bloques GNN y Transformer.**
4. **GNN y Transformer en Paralelo:** alimentar el grafo a GNN y Transformer concurrentemente, y fusionar las representaciones de salida en una representación.

En la primera categoría, los GTs primero procesan el grafo alimentándolo a una GNN, lo cual puede considerarse como tokenizar el grafo a nivel de subgrafo. La GNN agrega información de vecindarios locales para refinar los embeddings de nodo. Luego, los embeddings de nodo aumentados se alimentan a un Transformer, permitiendo al modelo aprender de tokens de subgrafo, como se discutió en la Sección 3.1.

La segunda categoría de arquitecturas se emplea comúnmente cuando los bloques Transformer han sido preentrenados. Por ejemplo, en el dominio de datos de proteínas, los Transformers [94] han demostrado capacidades efectivas para capturar representaciones de residuos de aminoácidos. Los marcos Protein GT típicamente explotan Transformers preentrenados para generar una representación inicial de nodo, seguida del refinamiento mediante una GNN respecto a la estructura espacial del grafo.

Intercalar bloques GNN y Transformer, como tercera categoría de ensamble de modelos, es una arquitectura simple pero efectiva. Por ejemplo, **Mesh Graphormer** [93] intercala bloques GNN y Transformer para reconstruir poses humanas, y **GROVER** [129] adopta una estrategia híbrida de combinar GNN y Transformer para aprender representaciones moleculares.

Al paralelizar GNN y Transformer, los GTs pueden aprender adaptativamente la importancia tanto de la información local como global. **GraphGPS** [128] utiliza la arquitectura paralela que combina las salidas de una MPNN y un Transformer. Además, GraphGPS aprovecha MPNN para actualizar los embeddings de bordes, que pueden utilizarse para actualizar aún más la PE.

**SGFormer** [162] demuestra teóricamente que una atención de una sola capa es suficientemente expresiva para capturar las interacciones globales entre nodos. En consecuencia, SGFormer propone una arquitectura GT simplificada que incorpora un mecanismo de atención lineal de auto-bucle de una sola capa junto con bloques GCN. Al combinar las representaciones finales del Transformer y la GNN, SGFormer exhibe escalabilidad considerable y rendimiento competitivo en tareas de predicción de propiedades de nodo.

**CoBFormer** [164] busca abordar el problema de over-globalization en los GTs. Para ello, CoBFormer paraleliza bloques GCN con el Transformer y propone una estrategia de entrenamiento colaborativo para suplementar el conocimiento de estructura de grafo local del GCN al Transformer. Específicamente, CoBFormer incorpora una función de pérdida adicional para alinear las representaciones de salida del GCN y el Transformer, permitiendo así supervisión mutua entre los dos módulos.

### 3.5 Hacia la Escalabilidad en Graph Transformer

Recordando que el mecanismo de self-attention en los Transformers introduce una complejidad computacional cuadrática respecto al número de nodos. Dado que los grafos del mundo real pueden contener millones o incluso miles de millones de nodos, los Transformers a menudo luchan para escalar eficientemente a grafos grandes. Por lo tanto, diseñar mecanismos de atención eficientes para grafos a gran escala sigue siendo un desafío significativo para la escalabilidad de los GTs.

Para reducir la complejidad del mecanismo de atención a lineal, un enfoque directo es integrar GNNs con Transformers lineales. Por ejemplo, **GraphGPS** [128] adopta Transformers establecidos que utilizan mecanismos de atención lineal, e.g., combinando Performer [27] y BigBird [180] con otros módulos GNN. Sin embargo, experimentos en GraphGPS revelan que aunque los mecanismos de atención lineal mejoran la escalabilidad, tienden a degradar el rendimiento. **SGFormer** [162], un GT alternativo basado en ensamble, introduce un mecanismo de atención lineal con propagación de auto-bucle. El análisis teórico demuestra que una sola capa de atención es suficiente para capturar interacciones globales, permitiendo a SGFormer lograr escalabilidad y precisión competitiva en tareas de clasificación de nodos. Otro GT lineal, **Polynormer** [31] implementa un mecanismo de atención local-a-global, que aprende polinomios de alto grado a partir de las características de entrada, incluyendo características de nodo y estructura de grafo.

**CobFormer** [164] presenta un módulo de atención global bi-nivel dirigido a mitigar el problema de over-globalization mientras reduce simultáneamente la complejidad del modelo. Inicialmente, CobFormer particiona todo el grafo en clusters. Posteriormente, un mecanismo de atención bi-nivel opera tanto a nivel intra-cluster como inter-cluster, lo que reduce significativamente el consumo de memoria. De manera similar, **Polynormer** [31] introduce un marco lineal mediante red polinomial, donde cada elemento de salida se representa como una función polinomial de las características de entrada. Para permitir equivariancia de permutación y combinar información local y global, calcula atención local sobre nodos vecinos y atención global sobre todo el grafo como los coeficientes en la red polinomial.

Una limitación notable del mecanismo de atención estructural, como se discutió en la Sección 3.3, es su difícil aplicabilidad a la atención lineal. Esto se debe a que los mecanismos de atención lineal no construyen explícitamente una matriz de atención, dificultando la incorporación de información estructural mediante attention bias o attention mask. Para esto, **NodeFormer** [160] introduce una pérdida de regularización a nivel de borde como se muestra en la Ecuación (22), que fomenta que los valores de atención entre nodos conectados en un grafo sean cercanos a 1.0:

```
L_e(A, A^G) = −(1/NL) Σ_{l=1}^{L} Σ_{(u,v)∈E} (1/d_u) log A_uv^(l)
```
*(Ecuación 22)*

donde L denota el número total de capas en NodeFormer, y d_u representa el grado del nodo u. Dado que esta función de pérdida solo requiere cálculos sobre bordes, NodeFormer gestiona eficientemente la complejidad de la regularización de bordes en O(|E|), manteniendo la complejidad general del modelo como O(|V|+|E|).

**Exphormer** [137] incorpora un mecanismo de atención sparse para lograr complejidad lineal. En esencia, este mecanismo combina la matriz de adyacencia, el expander graph y el nodo virtual. Un expander graph conecta nodos aleatoriamente y asegura que cada nodo mantenga un grado igual, resultando en que el expander posea un número de bordes lineal respecto a los nodos. A pesar de su complejidad lineal, el expander graph preserva la aproximación espectral de un grafo completo. Además, Exphormer logra un rendimiento competitivo comparado con la atención densa. **SP_Exphormer** [136] impulsa aún más esta dirección explorando una sparsificación más agresiva de la atención de grafo. De manera similar, **ANS-GT** [191] muestrea subgrafos con diferentes estrategias y emplea coarsening de grafo para reducir la complejidad computacional.

Un enfoque alternativo evita alimentar todo el grafo al Transformer. **NAGphormer** [16] transforma el vecindario k-hop N_k(v) en un embedding de vecindario x_v^k usando un operador de agregación φ. Este embedding agregado se trata luego como un token dentro del Transformer, permitiendo al modelo aprender el embedding del nodo v. Al agregar nodos vecinos antes del procesamiento por el Transformer, NAGphormer evita la necesidad de alimentar un gran número de nodos al Transformer. Además, **NAGphormer+** [18] mejora la característica de x_v^k enmascarando aleatoriamente una porción de vecinos para lograr mejor rendimiento. Adicionalmente, **VCR-Graphormer** [45] emplea random walk para reconectar el grafo mediante nodos virtuales. Al mantener un grafo con nodos virtuales, VCR-Graphormer controla su complejidad, manteniéndola lineal respecto al número de nodos mientras aún captura dependencias de largo alcance.

### 3.6 Geometric Graph Transformers

Dada la amplia gama de aplicaciones científicas del mundo real que involucran GTs, el estudio de GTs geométricos es crucial para modelar datos de grafo 3D, como sistemas moleculares y estructuras de proteínas. El principio de diseño central de estos marcos radica en asegurar la invariancia y/o equivariancia 3D del modelo. Esta sección revisa brevemente los GTs equivariantes de última generación que se han aplicado con éxito al modelado de grafos 3D.

El enfoque más directo para aprender las relaciones estructurales es incorporar la distancia relativa 3D como un embedding de borde adicional, que permanece invariable bajo transformaciones Euclidianas. Por ejemplo, **Graphormer** [173] introduce codificación espacial, donde se usa un MLP para codificar la distancia relativa entre átomos, capturando efectivamente las relaciones estructurales. Este paradigma ha demostrado su eficacia en varios marcos para aprender representaciones moleculares [203]. Además, otras características invariantes, como el ángulo entre bordes [169], pueden incluirse para representar información de orientación. Estas características invariantes usualmente se codifican usando funciones kernel, como la Radial Basis Function [28], para mejorar la expresividad del modelo.

**TorchMD-Net** [142] representa otro modelo equivariante que incorpora la distancia interatómica r_ij en su marco. El proceso comienza proyectando r_ij en dos filtros multidimensionales distintos, denotados D^K y D^V, usando las siguientes expresiones:

```
D^K = σ_1(r_ij),   D^V = σ_2(r_ij)
```
*(Ecuación 23)*

donde σ_1 y σ_2 son dos MLPs. Posteriormente, TorchMD-Net reemplaza la función Softmax tradicional con la función SiLU para calcular la matriz de atención:

```
A = SiLU((HW_Q)(HW_K)^T D^K) · φ(d_ij)
```
*(Ecuación 24)*

donde φ denota una función de corte que asigna el valor de 0 siempre que d_ij exceda un umbral predefinido. La representación final se calcula entonces por:

```
Z = σ_3(AVD^V)
```
*(Ecuación 25)*

donde σ_3 representa otra transformación lineal aprendible.

Para tareas como la generación de conformación, donde el modelo necesita generar coordenadas atómicas, **Uni-Mol** [203] propone una cabeza SE(3)-equivariante simple, representada como:

```
X̂_i = X̂_i + (1/n) Σ_{j=1}^{n} (X̂_i − X̂_j) c_ij
```
*(Ecuación 26)*

donde c_ij representa el embedding de relación aprendido entre el nodo v_i y v_j. Para mejorar la eficiencia, Uni-Mol actualiza las coordenadas solo en la capa final del modelo.

**GVP-Transformer** [62] representa un marco encoder-decoder basado en la arquitectura Transformer, diseñado para la tarea de plegamiento inverso de proteínas (protein inverse folding). El modelo está estructurado para tomar estructuras de proteínas y posteriormente generar las secuencias de proteínas correspondientes. Como encoder, GVP-Transformer utiliza GVP-GNN [78], capaz de extraer características que son invariantes a la traslación y equivariantes a la rotación, para modelar efectivamente estructuras de proteínas. Esto va seguido de la aplicación de un decoder Transformer para producir secuencias de proteínas válidas.

Más allá del modelado de un solo confórmero, **FACET** [120] introduce un marco escalable para el aprendizaje de conjuntos de confórmeros. Aprovecha un GT diferenciable para aproximar la distancia Fused Gromov-Wasserstein (FGW), permitiendo una agregación eficiente y geometry-aware de múltiples conformaciones 3D.

Ejemplos de GTs steerable de alto orden incluyen SE(3)-Transformer [46], Equiformer [90], EquiformerV2 [91], TetraGT [43], Q-GT [127] y EquiformerV3 [89]. Estos modelos emplean mecanismos de atención equivariante utilizando representaciones steerable de mayor grado de características [56], que caen fuera del enfoque de este survey.

---

## 4. Teorías

Más allá de la efectividad empírica de los GTs, también es importante entender sus fundamentos teóricos. Esta sección primero revisa la capacidad expresiva de los GTs desde las perspectivas de tokenización y codificación posicional (Sección 4.1), y luego discute las relaciones entre los GTs y otros paradigmas de aprendizaje en grafos (Sección 4.2).

### 4.1 Expresividad

Siguiendo el orden de la Sección 3, se discute la expresividad de la tokenización estructural y la codificación posicional. Estos dos componentes juegan roles diferentes: la tokenización determina qué objetos estructurales se tratan como tokens para la atención, mientras que la PE proporciona relaciones estructurales adicionales entre estos tokens. Por lo tanto, la cuestión teórica no es solo si un GT es más fuerte, sino también qué tipos de ambigüedades de grafo puede distinguir y a qué costo computacional o estadístico.

#### 4.1.1 Tokenización Estructural

La tokenización determina el espacio en el que opera un Transformer. La tokenización a nivel de nodo y borde permanece cercana a las entidades primitivas del grafo, y su expresividad por lo tanto todavía depende del sesgo estructural adicional introducido por la PE, características de bordes, o restricciones arquitectónicas. En contraste, la tokenización de orden superior expone directamente objetos multi-nodo a la atención. **TokenGT** [81] es un ejemplo representativo que muestra que el inventario de tokens en sí mismo puede mejorar la expresividad teórica de un Transformer puro más allá de una vista de token de nodo vanilla.

Esto puede explicarse aún más por el marco de [118]. Si a un Transformer se le dan tokens de entrada k-tupla adecuados X^(0,k) ∈ ℝ^(n^k × d), entonces su capa t-ésima puede emular la t-ésima iteración de un procedimiento WL de k-orden. Este resultado indica que el comportamiento más fuerte que 1-WL no surge de la atención sola, sino de permitir al modelo atender sobre objetos o relaciones de orden superior.

También es útil distinguir la tokenización a nivel de subgrafo y de hop, aunque a menudo se discuten juntas. La tokenización a nivel de subgrafo trata un parche inducido como un objeto [14]. Su identidad depende del arreglo interno de nodos y bordes dentro del parche, y por lo tanto es adecuada cuando el tipo de motivo, la estructura de ciclo u otras configuraciones locales de orden superior son importantes. La tokenización a nivel de hop en cambio mantiene un nodo ancla central y organiza su contexto en una secuencia ordenada de vecindarios indexados por distancia [16, 18, 186]. Su sesgo inductivo clave es la descomposición radial: el modelo aprende cómo se distribuye la evidencia a través de las capas 1-hop, 2-hop, ... alrededor de un ancla, en lugar de aprender la identidad de un parche completo. En este sentido, la tokenización a nivel de hop no debe considerarse simplemente como tokenización a nivel de subgrafo con un k diferente; la primera enfatiza patrones de propagación multi-hop y escalabilidad, mientras que la última enfatiza el tipo de isomorfismo de una estructura local.

Esta distinción también ayuda a explicar cuándo son necesarias las señales ≥2-WL. Tales señales son especialmente útiles cuando el objetivo depende de relaciones entre múltiples nodos que las actualizaciones de nodo estilo 1-WL no pueden distinguir, como roles simétricos, participación en motivos, o estructuras locales que comparten el mismo multiset de características de vecinos. En estos casos, la tokenización consciente de subgrafo, borde o tupla puede exponer más directamente la estructura faltante. En contraste, cuando los atributos de nodo o borde ya rompen estas simetrías, o cuando la tarea depende principalmente de evidencia de corto alcance, la tokenización de orden superior puede introducir computación adicional sin traer ganancias claras. Esto también explica por qué muchos GTs a nivel de nodo, e incluso baselines fuertes de MPNN, siguen siendo competitivos en benchmarks prácticos a pesar de una expresividad de peor caso más débil.

#### 4.1.2 Codificación Posicional

La PE no cambia el token en sí, pero proporciona relaciones estructurales para las capas de atención. Desde esta perspectiva, la PE absoluta asigna a cada token una coordenada estructural, típicamente derivada de espectros, difusión o estadísticas de grado. La PE relativa mantiene la señal en forma pairwise y la inyecta directamente en la atención, permitiendo al modelo comparar dos tokens mediante distancia de camino más corto, distancia de resistencia, afinidad de random walk u otros puntajes de relación.

Esta distinción lleva a varias observaciones teóricas importantes. Black et al. [6] muestran, a través de una conversión basada en 2-equivariant graph networks [108], que en grafos sin características de nodo, el poder de distinción de grafos de la PE absoluta y relativa puede ser equivalente. Una vez que las características de nodo están presentes, sin embargo, convertir PE relativa en PE absoluta puede perder información. Intuitivamente, comprimir una relación pairwise en dos etiquetas separadas a nivel de nodo es más débil que exponer esa relación directamente al mecanismo de atención. Por lo tanto, la PE relativa puede ser más adecuada cuando la estructura pairwise juega un papel central.

Esta comparación también indica qué puede y no puede reemplazar la PE. Una PE absoluta más fuerte ayuda a romper simetrías de grafo y proporciona a los tokens a nivel de nodo un sistema de coordenadas global más informativo. Una PE relativa más fuerte determina qué pruebas estructurales pairwise pueden realizarse directamente en la atención. Sin embargo, incluso una PE poderosa usualmente enriquece un modelo con señales unarias o pairwise, en lugar de identidades arbitrarias de subgrafo. En otras palabras, la PE puede compensar por la tokenización a nivel de nodo solo cuando la tarea está gobernada principalmente por posiciones de nodo o relaciones pairwise; si la evidencia clave radica en la identidad de motivo o topología local de orden superior, los tokens de orden superior siguen siendo la elección más directa.

Fundamentalmente, la PE y la Tokenización sirven roles ortogonales en los GTs. La Tokenización define el espacio de observación (las entidades a las que el modelo puede atender), mientras que la PE establece el sistema de coordenadas dentro de ese espacio. Si la tarea downstream requiere fundamentalmente razonamiento sobre estructuras topológicas de orden superior (e.g., motivos químicos específicos), imponer un sistema de coordenadas espectral poderoso (PE) sobre un espacio de observación a nivel de nodo sigue siendo sub-óptimo. En tales casos, cambiar explícitamente el espacio de observación mediante tokenización a nivel de subgrafo o de borde es un enfoque matemáticamente más directo y empíricamente más eficiente.

Estudios teóricos recientes también ayudan a explicar la brecha entre el poder expresivo y el rendimiento empírico [85]. Primero, los análisis estilo WL conciernen la distinguibilidad de peor caso, mientras que la precisión de benchmark también está influenciada por la optimización, la eficiencia de muestra y la alineación de supervisión. Segundo, las señales posicionales teóricamente más fuertes no siempre son las más robustas en la práctica: la PE basada en Laplaciano sufre de ambigüedad de signo y base, las características espectrales pueden ser sensibles a perturbaciones, y algunas PE relativas son difíciles de combinar con atención sparse o lineal. Tercero, muchos datasets ya contienen atributos de nodo ricos u objetivos principalmente locales, por lo que los casos que separan 1-WL de pruebas más fuertes pueden no dominar la pérdida final.

Por lo tanto, una PE teóricamente más fuerte no se traduce uniformemente en superioridad empírica [52]; los investigadores deben navegar un trade-off estricto entre poder expresivo, presupuesto computacional y estabilidad ante perturbaciones, como se resume en la Tabla 3. Las ganancias prácticas aparecen solo cuando la tarea realmente requiere las distinciones estructurales adicionales y el modelo puede explotarlas dentro de su presupuesto de cómputo y muestra.

#### 4.1.3 Más allá de los Grafos Pairwise: Extensiones a Grafos Firmados, Dirigidos e Hipergrafos

Las cuatro formas de organización de grafo revisadas en la Sección 5 se enfocan principalmente en grafos no dirigidos, pairwise, con atributos opcionales. Varias extensiones naturales, como grafos dirigidos, grafos firmados (signed) e hipergrafos, merecen mención, ya que pueden abordarse dentro del mismo marco teórico sin requerir una nueva taxonomía fundamental.

Los **grafos dirigidos** ya están cubiertos por la forma de grafo relacional. Los grafos de conocimiento y grafos AMR, que aparecen en la Sección 5.2, son inherentemente dirigidos: los bordes codifican semántica específica de orientación (e.g., sujeto→predicado→objeto). Los mismos mecanismos de attention-bias y proyección específica de tipo de borde discutidos para grafos relacionales manejan directamente la direccionalidad.

Los **grafos firmados**, donde los bordes llevan signos positivos o negativos, también encajan en el marco relacional sin modificación. Redes de confianza/desconfianza sociales y grafos de calificación de usuarios son las instancias típicas. Desde la perspectiva de un GT, un borde firmado es simplemente una relación tipada con valores en {+1, −1}; la misma maquinaria de attention-bias y codificación de tipo de borde cubierta en la Sección 3 aplica. La elección clave de diseño es si el signo modula el puntaje de atención (como en las GNNs firmadas) o se trata como un atributo separado a nivel de token.

Los **hipergrafos** incorporan hiperbordes que conectan un conjunto arbitrario de nodos en lugar de un par. Ejemplos de aplicación incluyen redes de co-autoría, recomendación basada en sesión y modelado de complejos de proteínas. Desde la perspectiva de expresividad de GT, un hiperborde es precisamente la tupla-k de entrada estudiada en TokenGT [81] y el marco de emulación k-WL de [118]: codificar una relación k-aria como un token de orden superior eleva al modelo más allá de la expresividad 1-WL. Esto es conceptualmente consistente con la fila de tokenización de orden superior de la Tabla 4, donde datos con estructura de hipergrafo se beneficiarían directamente de tokenización a nivel de subgrafo, de borde o tupla.

En resumen, las tres extensiones se mapean naturalmente a las categorías arquitectónicas y marcos teóricos ya revisados en las Secciones 3 y 4.1, reforzando la generalidad de la perspectiva de forma de organización y frecuencia de componentes.

### Tabla 3: Trade-offs Teóricos y Prácticos de las Codificaciones Posicionales de Grafo

| Estrategia PE | Límite de Expresividad | Complejidad | Estabilidad | Restricciones de Aplicabilidad |
|---|---|---|---|---|
| Degree | 1-WL | O(\|E\|) | Alta | Falla en grafos regulares |
| Laplacian PE | >1-WL (Espectral) | O(\|V\|³) | Baja | Ambigüedad de signo/base; grafos no dirigidos |
| SVD PE | >1-WL (Espectral) | O(\|V\|² r) | Moderada | Soporta grafos dirigidos/ponderados |
| Random Walk PE | Consciente de subgrafo | O(k\|E\|) | Alta | Depende de vecindarios k-hop únicos |
| Shortest Path (Rel.) | >1-WL (Distancia) | O(\|V\|³) | Alta | Difícil de adaptar a atención lineal |

### 4.2 Relación con Otros Métodos de Aprendizaje en Grafos

Las características de los GTs pueden dilucidarse comparándolos con otros métodos de aprendizaje en grafos. En esta sección, se examinan estudios que comparan GT con MPNN, aprendizaje de estructura de grafo y graph attention networks (GATs).

#### 4.2.1 MPNN

Comparados con las MPNNs, los GTs integran mecanismos de self-attention y PE. Un estudio reciente [85] muestra que la self-attention puede mejorar la tasa de convergencia de los GTs, mientras que la PE ayuda a identificar el vecindario central para cada nodo y así mejora la generalización. Los GTs con distancia de camino más corto como PE relativa también pueden ser teóricamente más expresivos que las MPNNs clásicas [6]. Sin embargo, una mayor expresividad teórica no implica necesariamente una ventaja empírica uniforme.

Un enfoque alternativo para infundir información global en cada nodo es introducir un nodo virtual conectado a todos los nodos en un grafo. A pesar de la simplicidad de esta idea, la MPNN con el nodo virtual [8] sirve sorprendentemente como un baseline fuerte en el Long Range Graph Benchmark [39]. Un estudio reciente [131] muestra además que ningún algoritmo único supera consistentemente a los demás al comparar GTs y MPNNs con un nodo virtual.

Además, el problema de over-smoothing [130], caracterizado en MPNNs profundas, también existe en los Transformers [133], llevando a embeddings de nodo indistinguibles en capas más profundas. Como el Transformer es una forma especial de Graph Attention Networks (GAT) [146], comparte el mismo fenómeno de over-smoothing que GAT, llevando a una degeneración exponencial del poder expresivo respecto al número de capas. Para mitigar el over-smoothing, **ParaFormer** [178] propone un mecanismo de Generalized PageRank Attention para preservar la información de frecuencia diversa en la estructura del grafo desde la perspectiva del procesamiento de señales de grafo.

#### 4.2.2 Aprendizaje de Estructura de Grafo

El Graph Structure Learning (GSL) está estrechamente relacionado con los GTs, que busca refinar automáticamente estructuras de grafo cuando el grafo de entrada es ruidoso o incompleto, o inferir estructuras de grafo implícitas cuando la estructura explícita no está disponible [87], de manera parametrizada. Los GTs pueden considerarse una forma especial de GSL, lograda mediante self-attention que aprende una estructura de grafo "suave" completamente conectada [145]. Al utilizar técnicas orientadas a la atención, como la máscara de atención en la Sección 3.3.2 y el muestreo de estructura discreta en NodeFormer [160], la estructura de grafo aprendida puede ser esparsificada para reflejar la topología del mundo real.

---

## 5. Aplicaciones

Los GTs han sido adoptados en muchos dominios. En esta sección se revisa el panorama a través de la lente de la forma de organización de grafo: relacional, geométrica, dinámica y heterogénea o multimodal. Esta agrupación reduce las descripciones repetitivas de configuraciones de tarea similares y vincula los casos de uso downstream más directamente con los componentes arquitectónicos cubiertos en la Sección 3. La Figura 3 da una visión general de las tareas representativas y modelos bajo cada forma.

### Figura 3 (descripción — Visión general de aplicaciones de GT bajo cuatro formas de organización de grafos)

**Relational (§5.2)** — Predicción y Generación: MAT, R-MAT, GROVER, LiGhT, CoAtGIN, BatmanNet, DMP, MolSpectra, DiGress, CDGS, Uni-Mol, Uni-Mol+, Uni-Mol2, Mudiff, JODO, GTMGC, GraphDiT, BrainNetTF, Cai et al., THC, TSEN, ALTER, BioBGT. Grafo-a-Texto y Razonamiento: Zhu et al., Cai et al., HetGT, ASAG, GraphFormer, KGTransformer, Relphormer, TG-Transformer, KG-R3, GT-BEHRT.

**Geometric & Periodic (§5.3)** — Predicción de Propiedades: Equiformer, EquiformerV2, TorchMD-NET, SE(3)-Transformer, GNS-TAT, TGT, TransFun, HEAL, scMoFormer, Stability Oracle, ProstT5, Saprot, Matformer, CrystalFormer, CrysGraphFormer, DPA-2, MatterSim, OMat24, ComFormer, Mesh Graphormer, PoseGTAC, GTRS, Graformer, 3DMOTFormer, SGraFormer, SGFormer. Interacción y Docking: GraphSite, RTMScore, IGT, GeoT, GGT, GraphormerDTI, AttentionMGT-DTA, GTAMP-DTA, Graph-BERT, HGIN, Uni-Mol, GeoDock, EBMDock. Generación y Diseño: JODO, MUDiff, GTMGC, Uni-Mol, Uni-Mol+, Uni-Mol2, GVP-Transformer, LM-Design, ProRefiner, FAIR, PocketGen.

**Dynamic (§5.4)** — Forecasting: Social Attention, Trajectron, TrafficPredict, STAR, SSAGCN, Trafformer, LLGformer, IGT, GMAN, HS-GT, HST-GT, GCT-TTE, HPST-GT. Propagación y Eventos: StA-HiTPLAN, DGTR, Lgt, HeteroSGT, GCNs-MT, PHAROS, PSGT, GT-BEHRT.

**Heterogeneous / Multimodal (§5.5)** — Predicción de Interacción: PMGT, GMT, GFormer, LightGT, MGFormer, TransGNN, SIGFormer, Rankformer. Aprendizaje de Representación: GNN-nested Transformer, Edgeformer, Heterformer, GTP, AMIGO, MulGT, MG-Trans, CGT, IGT, SpaFormer, HEAT. Grounding y QA: M-DGT, DUET, Multimodal GT, mDT, KGEMT.

### 5.1 Una Visión Unificada de las Formas de Organización de Grafo

Bajo las cuatro formas, una gran fracción de aplicaciones de GT puede capturarse mediante las plantillas de tarea compartidas:

```
y = φ_θ(G, X)                          (Ecuación 27)
s = φ_θ(G_a, X_a, G_b, X_b)            (Ecuación 28)
Ĝ, X̂ = φ_θ(G, X, c)                    (Ecuación 29)
T̂ = f(φ_θ(G, X))                       (Ecuación 30)
```

Aquí, G denota la estructura del grafo, X denota información lateral opcional como geometría, tiempo o características específicas de modalidad, y es un objetivo de predicción, s es un puntaje para un par de grafos, (Ĝ, X̂) denota contenido de grafo generado o refinado, y T̂ es texto decodificado. Con base en esta formulación, se pueden resumir las señales estructurales dominantes y las elecciones de diseño de GT correspondientes a través de las diferentes formas de organización de grafo.

### 5.2 Grafos Relacionales con Semántica Discreta

Los grafos relacionales consisten principalmente en entidades discretas y relaciones tipadas. La dificultad central en este contexto es modelar dependencias semánticas no locales sin depender de geometría Euclidiana. Los GTs tienden a rendir bien aquí cuando se equipan con biases de atención conscientes de relación, codificaciones de camino más corto o centralidad, y agregación de contexto a nivel de subgrafo. Instanciaciones comunes incluyen grafos moleculares 2D, redes de conectividad cerebral, grafos semánticos AMR y grafos de conocimiento, abarcando tareas desde predicción a nivel de grafo único y generación molecular hasta decodificación grafo-a-texto y razonamiento basado en enlaces.

#### 5.2.1 Organización de Datos y Plantillas de Tarea Compartidas

La entrada estándar es un grafo G = (V, E) con atributos de nodo y borde. Los objetivos comunes son la predicción a nivel de grafo y = φ_θ(G), la generación discreta de grafo Ĝ = φ_θ(G, c), y la decodificación condicionada al grafo T̂ = f(φ_θ(G)). Comparado con aplicaciones conscientes de geometría, un problema central es cómo exponer tipos de borde, dependencias multi-hop, y composición simbólica a las capas de atención.

#### 5.2.2 Predicción y Generación de Grafo Único

El modelado molecular bidimensional es una motivación común para esta forma. Cuando las moléculas se tratan como grafos químicos sin coordenadas explícitas, los GTs sirven principalmente para propagar información entre grupos funcionales distantes y caminos químicamente significativos. **MAT** [110] y **R-MAT** [111] inyectan información estructural como biases de atención, mientras que **GROVER** [129] y **CoAtGIN** [189] combinan message passing local con atención global para cubrir dependencias tanto a nivel de motivo como de grafo. Esta plantilla relacional también cubre el análisis de conectividad cerebral, donde los grafos se definen por correlaciones en lugar de geometría. **BrainNetTF** [79], Cai et al. [10], **THC** [30], **ALTER** [176], **BioBGT** [124] y **LGC-SGT** [184] usan todos GTs para aprender representaciones de sujeto a nivel de grafo a partir de patrones de conectividad ponderados.

La generación discreta y el aprendizaje auto-supervisado encajan en la misma organización. Para la generación de moléculas, **DiGress** [148], **CDGS** [66] y **GraphDiT** [95] generan o refinan grafos moleculares discretos preservando la sparsity del grafo y el control condicional. **GraphXForm** [125] demuestra que los backbones de GT se extienden al diseño molecular asistido por computadora más allá de la predicción de propiedades estándar. Para el pre-entrenamiento, **MolT5** [40], **GROVER** [129], **LiGhT** [86], **BatmanNet** [153] y **DMP** [208] optimizan objetivos de enmascaramiento o consistencia cross-view en vistas moleculares simbólicas. **Transformer-M** [101], **MoleBLEND** [175], **MolSpectra** [150] y **RELGT** [37] alinean el andamio relacional con modalidades complementarias durante el pre-entrenamiento. En este contexto, los GTs son más efectivos cuando la semántica del grafo depende de cadenas de relación largas o motivos de orden superior; cuando los grafos son muy pequeños y las dependencias son fuertemente locales, las MPNNs más simples siguen siendo competitivas.

#### 5.2.3 Grafo-a-Texto Semántico y Razonamiento

Los grafos semánticos como los grafos AMR y grafos de conocimiento instancian la misma forma de organización, pero sus salidas son texto o respuestas simbólicas en lugar de etiquetas de grafo. Para la generación grafo-a-texto, la plantilla compartida es un pipeline encoder-decoder T̂ = f(φ_θ(G)). Los modelos existentes de AMR-a-texto codifican relaciones estructurales mediante información de camino más corto y biases de atención específicos de grafo, como en Zhu et al. [207], Cai et al. [9] y **HetGT** [172]. **ASAG** [1] usa la misma idea para la calificación de respuestas de estudiantes convirtiendo ambas respuestas en grafos AMR antes de la comparación consciente de grafo.

El razonamiento sobre grafos de conocimiento encaja en esta forma, ya que la señal estructural primaria sigue siendo llevada por relaciones tipadas. **GraphFormer** [171] codifica relaciones de camino más corto como biases de atención, **KG-Transformer** [97] combina pre-entrenamiento consciente de grafo con enrutamiento MoE para razonamiento complejo, y **Relphormer** [5] refuerza el modelado enmascarado con señales relacionales de orden superior. **KG-R3** [123] sigue además esta plantilla recuperando un subgrafo candidato y luego aplicando un GT sobre nodos y bordes para responder consultas sujeto-relación. Una lente centrada en relaciones conecta así tareas superficialmente diferentes como generación AMR y completación de KG.

### 5.3 Grafos Geométricos y Periódicos

Los grafos geométricos aumentan la topología con coordenadas, distancias, direcciones y ángulos; los grafos periódicos añaden además estructura de red (lattice). Los GTs que operan sobre estos grafos deben capturar dependencias de largo alcance mientras respetan la consistencia geométrica, lo que hace que las representaciones de pares, la atención consciente de distancia y las capas equivariantes sean mucho más centrales que en configuraciones puramente relacionales. Instanciaciones comunes incluyen conformaciones moleculares 3D, estructuras de proteínas, materiales cristalinos periódicos y mallas o nubes de puntos geométricas de visión, abarcando tareas desde predicción de propiedades geométricas y docking pairwise hasta generación de estructura 3D y diseño inverso.

#### 5.3.1 Organización de Datos y Plantillas de Tarea Compartidas

Se denota un grafo geométrico como G⃗ = (V, E, X⃗), donde X⃗ almacena coordenadas u otros descriptores geométricos. Para materiales periódicos, se requiere una matriz de red adicional L. Las tareas dominantes son la predicción de propiedades y = φ_θ(G⃗), el modelado de pares s = φ_θ(G⃗_a, G⃗_b), y la generación o refinamiento de estructura (Ĝ, X̂) = φ_θ(G⃗, c). Dado que estas tareas dependen de la geometría en sí más que solo de la conectividad, un problema de diseño importante es cómo se incorpora la geometría en la atención, las características de pares y las actualizaciones equivariantes.

#### 5.3.2 Predicción de Propiedades en Grafos de Objetos Estructurados

La predicción de propiedades moleculares tridimensionales es la representante de esta forma. **Equiformer** [90], **EquiformerV2** [91], **TorchMD-NET** [142] y **SE(3)-Transformer** [46] usan capas GT equivariantes para que las predicciones moleculares permanezcan estables bajo transformaciones rígidas. La misma forma de organización aparece en la predicción de propiedades de proteínas, donde **TransFun** [7], **HEAL** [55], **Stability Oracle** [33], **scMoFormer** [141], **Saprot** [139] y **ProstT5** [61] fusionan información de residuos derivada de secuencia con grafos de proteína espaciales. Los GTs encajan bien aquí porque residuos distantes a lo largo del backbone pueden estar aún fuertemente acoplados en el espacio 3D.

La predicción de propiedades de cristales sigue la misma plantilla geométrica pero añade periodicidad. **Matformer** [169], **CrystalFormer** [152] y **CrysGraphFormer** [140] codifican información de distancia y angular directamente en la atención. **ComFormer** [168] combina capas invariantes y equivariantes, mientras que **MatterSim** [170] y **OMat24** [3] demuestran que el pre-entrenamiento a gran escala en materiales puede mejorar aún más la transferibilidad. Las tareas de visión por computadora geométrica también pueden interpretarse de la misma manera una vez que las imágenes se convierten en mallas o nubes de puntos. **Mesh Graphormer** [93], **PoseGTAC** [209], **Graformer** [196], **GTRS** [199], **3DMOTFormer** [34], **SGraFormer** [187] y **SGFormer** [103] operan todos sobre relaciones espaciales explícitas en lugar de sobre semántica específica de dominio.

#### 5.3.3 Interacción Pairwise y Docking

Cuando el objetivo depende de la interacción entre dos grafos geométricos, la tarea compartida es el modelado de pares. La afinidad proteína-ligando, el docking proteína-proteína y las interacciones fármaco-diana caen todas aquí, y el modelo debe aprender representaciones de pares conscientes de geometría, no dos embeddings de grafo independientes. **RTMScore** [132], **IGT** [96], **GeoT** [115], **GGT** [25], **GraphormerDTI** [47], **AttentionMGT-DTA** [158] y **GTAMP-DTA** [143] mejoran todos la atención con características espaciales pairwise o embeddings bioquímicos preentrenados. **HGIN** [194], **Graph-BERT** [72] y **GraphSite** [179] inyectan de manera similar información pairwise de residuo o distancia para modelar interacciones biológicas. Para el docking, **Uni-Mol** [203], **GeoDock** [29] y **EBMDock** [159] refinan las representaciones de pares en poses finales o estructuras consistentes con energía.

#### 5.3.4 Generación de Estructura y Diseño Inverso

La generación y el diseño inverso son también aplicaciones importantes bajo esta forma de organización. En el modelado molecular, **JODO** [67], **MUDiff** [65], **Uni-Mol** [203] y **GTMGC** [165] generan o refinan conformaciones 3D acoplando grafos químicos discretos con coordenadas continuas. El diseño de proteínas puede verse bajo la misma formulación, donde la estructura del grafo está dada y el modelo predice la secuencia de aminoácidos o la estructura de bolsillo faltante. **GVP-Transformer** [62], **LM-Design** [202], **ProRefiner** [204], **FAIR** [192] y **PocketGen** [193] explotan todos los GTs para alinear el contexto estructural con el diseño de secuencia.

El aprendizaje auto-supervisado en grafos geométricos también sigue esta forma de organización. **Uni-Mol** [203], **GNS-TAT** [181], **Frad** [44] y **SliDe** [122] denoisean coordenadas, geometría de enlace o perturbaciones relacionadas con fuerzas antes de la predicción o generación downstream. Estos objetivos son especialmente útiles cuando la supervisión downstream es escasa. Dicho esto, los GTs no siempre superan a las GNNs geométricas más simples: si el grafo es pequeño, las interacciones son fuertemente locales y la memoria es limitada, una MPNN equivariante bien ajustada puede ser el mejor trade-off.

### 5.4 Grafos Dinámicos y Flujos de Eventos

Los grafos dinámicos están indexados por tiempo. El desafío aquí es modelar conjuntamente la estructura del grafo y la evolución temporal, y decidir cómo deben interactuar las señales espaciales y temporales. Los GTs son adecuados porque el self-attention captura naturalmente dependencias temporales de largo alcance, mientras que los módulos conscientes de grafo preservan la estructura de interacción local. Instanciaciones comunes incluyen predicción de trayectoria multi-agente, predicción de flujo de tráfico y eventos, detección de propagación de rumores y razonamiento de flujo de eventos en registros médicos electrónicos, abarcando tareas desde forecasting de estado futuro y clasificación hasta comprensión a nivel de secuencia de cascadas de propagación.

#### 5.4.1 Organización de Datos y Plantillas de Tarea Compartidas

Sea {G_t}_{t=1}^T una secuencia de grafos o estados de grafo inducidos por eventos. Las tareas dominantes son el forecasting Ŷ_{T+1:T'} = φ_θ({G_t}_{t=1}^T) y la predicción a nivel de secuencia y = φ_θ({G_t}_{t=1}^T). Comparado con grafos estáticos, preguntas de diseño importantes incluyen si la atención espacial y temporal deben acoplarse o desacoplarse, cómo debe codificarse el orden de eventos, y cómo puede controlarse la complejidad cuando crece el horizonte de observación.

#### 5.4.2 Forecasting Temporal

Los sistemas de tráfico motivan esta forma más visiblemente. En la predicción de trayectoria, el grafo evoluciona mientras los agentes se mueven y sus interacciones cambian con el tiempo. **Social Attention** [147], **Trajectron** [71], **TrafficPredict** [106] y **SSAGCN** [104] acoplan explícitamente bordes espaciales y temporales. **STAR** [174] separa bloques Transformer espaciales y temporales, mientras que **Trafformer** [77] y **LLGformer** [76] inyectan información posicional estructural y temporal directamente en la atención.

La predicción de eventos de tráfico usa la misma forma de organización pero típicamente a mayor escala y a menudo con tipos de nodo heterogéneos. **GMAN** [198] paraleliza la atención espacial y temporal, **IGT** [205] construye subgrafos bipartitos específicos de tipo antes de la fusión Transformer, **GCT-TTE** [109] combina encoders GNN y Transformer, y **HS-GT** [41], **HST-GT** [197] y **HPST-GT** [151] extienden GTs heterogéneos al forecasting temporal. Estos estudios sugieren que los GTs son particularmente útiles cuando la dependencia temporal de largo alcance importa o cuando el modelo necesita fusionar múltiples escalas de interacción.

#### 5.4.3 Propagación y Comprensión de Eventos

La detección de rumores es otra instancia de razonamiento en grafo dinámico porque la predicción depende del despliegue temporal de un árbol de propagación en lugar de un grafo social estático. **StA-HiTPLAN** [80] modela relaciones de tweets como biases de atención, **DGTR** [154] combina Transformers estructurales y temporales, y **Lgt** [163] integra componentes GNN y Transformer para razonamiento conjunto local-global. **HeteroSGT** [190], **GCNs-MT** [12], **PHAROS** [121] y **PSGT** [206] muestran además que las tareas de propagación a menudo requieren atención heterogénea o enmascarada por topología.

El razonamiento de flujo de eventos también aparece fuera de las redes sociales. **GT-BEHRT** [126], por ejemplo, modela registros médicos electrónicos como grafos de eventos específicos del paciente con atención consciente de relación. Esto captura lo que la taxonomía de forma de organización aporta: las cascadas de rumores y los historiales de pacientes provienen de diferentes dominios, pero ambos involucran problemas conscientes de secuencia en grafos y se benefician de patrones de GT similares. En la práctica, los GTs dinámicos necesitan control cuidadoso de la complejidad; cuando el horizonte es corto o las interacciones son casi locales, los baselines recurrentes o de message-passing todavía se sostienen bien.

### 5.5 Grafos Heterogéneos y Multimodales

La última forma cubre grafos que integran múltiples tipos de nodo, tipos de borde o modalidades. Los GTs son útiles aquí no solo por su campo receptivo global, sino también porque pueden fusionar de manera flexible tokens de objetos estructuralmente diferentes. Las proyecciones específicas de tipo, la atención heterogénea, la cross-attention y el pooling de grafo son patrones de diseño recurrentes. Instanciaciones comunes incluyen grafos de recomendación usuario-ítem, redes ricas en texto, grafos inducidos por imágenes de láminas de patología e imágenes celulares, y grafos multimodales para grounding visual y respuesta a preguntas, abarcando tareas desde predicción de interacción y aprendizaje de representación hasta inferencia cross-modal.

#### 5.5.1 Organización de Datos y Plantillas de Tarea Compartidas

Se escribe la entrada como G = (V, E, M), donde M denota características específicas de tipo o modalidad. Los objetivos más comunes son la predicción de interacción ŷ_{u,v} = φ_θ(G, u, v), el aprendizaje de representación de grafo z = φ_θ(G), y la inferencia cross-modal ŷ = φ_θ(G^(1), G^(2), M). A diferencia de las formas anteriores, el desafío principal aquí no es solo el rango estructural, sino también el desajuste semántico entre tipos de nodo y modalidades.

#### 5.5.2 Predicción de Interacción en Grafos Bipartitos y Heterogéneos

Los sistemas de recomendación proporcionan el ejemplo canónico. Las interacciones usuario-ítem forman grafos bipartitos o heterogéneos, a menudo aumentados con información lateral textual, acústica o visual. **PMGT** [98], **GMT** [114] y **MGFormer** [15] usan muestreo de grafo, enmascaramiento y bias posicional para aprender representaciones usuario-ítem escalables. **GFormer** [84], **LightGT** [155], **SIGFormer** [21] y **TransGNN** [188] combinan encoders de grafo locales con bloques Transformer para fusionar señales estructurales y multimodales. **Rankformer** [23] además alinea la arquitectura con objetivos de ranking, mientras que **HIRE** [42] muestra que el modelado heterogéneo basado en atención es especialmente útil en escenarios de cold-start.

#### 5.5.3 Aprendizaje de Representación en Grafos Inducidos por Texto e Imagen

En muchas aplicaciones, un grafo se construye primero a partir de otra modalidad y luego se alimenta a un GT para el aprendizaje de representación. Los grafos ricos en texto son un caso canónico: **GNN-nested Transformer** [171] alterna procesamiento local y global, **Edgeformer** [74] modela explícitamente contenido textual de borde, y **Heterformer** [75] asigna proyecciones específicas de tipo a nodos con y sin texto. Una vez que los tokens de grafo llevan descripciones de lenguaje natural largas, la fusión heterogénea tiende a importar más que la topología de grafo pura.

Las imágenes de láminas completas y celulares siguen un patrón similar, con grafos inducidos a partir de parches de imagen, células o biomarcadores en lugar de darse a priori. **GTP** [201] usa GCNs y pooling para reducir el número de parches de imagen antes del modelado con Transformer; **AMIGO** [119], **MulGT** [195], **MG-Trans** [134] e **IGT** [135] aprenden sobre vistas de grafo específicas de biomarcador o tarea; **SpaFormer** [156], **CGT** [149] y **HEAT** [11] inyectan además información de random-walk, Laplaciana, espacial y de borde heterogénea. Comparados con los vision transformers estándar, estos modelos explotan el hecho de que el contexto relevante está estructurado por la adyacencia celular y la organización del tejido en lugar de por el orden raster.

#### 5.5.4 Grounding Cross-Modal, Navegación y Respuesta a Preguntas

Las tareas cross-modales combinan la estructura de grafo con una modalidad adicional en tiempo de inferencia. **M-DGT** [24] realiza grounding visual construyendo un grafo sobre regiones de imagen y condicionándolo al texto. **DUET** [22] usa memorias de navegación conscientes de grafo junto con cross-attention textual para navegación embodied. **Multimodal GT** [59], **mDT** [60] y **KGEMT** [200] usan todos grafos semánticos o estructurales para mejorar la respuesta a preguntas multimodal y la comprensión de redes sociales. Estas tareas difieren de la recomendación o el aprendizaje de grafo de texto porque el grafo es solo una parte del contexto de razonamiento; sin embargo, se reutilizan los mismos mecanismos de GT heterogéneos.

### Tabla 4: Frecuencia de Componentes Arquitectónicos de GT a través de las Cuatro Formas de Organización de Grafo

Basado en las Secciones 5.2–5.5. ●●● ●●○ y ●○○ indican frecuencia alta, media y baja respectivamente, y ○○○ significa que el módulo no es aplicable en esa aplicación.

| Componente | Relacional | Geométrico | Dinámico | Heterogéneo |
|---|---|---|---|---|
| **Tokenización Multi-nivel de Grafos (Sec. 3.1)** | | | | |
| Tokenización a nivel de nodo | ●●● | ●●● | ●●● | ●●● |
| Tokenización a nivel de borde | ●●○ | ●○○ | ●○○ | ●●○ |
| Tokenización a nivel de subgrafo | ●●○ | ●○○ | ●○○ | ●○○ |
| Tokenización a nivel de hop | ●●○ | ○○○ | ●○○ | ●○○ |
| **Codificación Posicional Estructural (Sec. 3.2)** | | | | |
| Shortest Path (SPD) | ●●● | ●●○ | ●○○ | ●○○ |
| Random Walk PE | ●●● | ●○○ | ●○○ | ●○○ |
| Degree PE | ●●○ | ●○○ | ●○○ | ●○○ |
| Laplacian PE | ●○○ | ●●○ | ●○○ | ○○○ |
| Temporal PE | ○○○ | ○○○ | ●●● | ○○○ |
| **Atención Estructural (Sec. 3.3)** | | | | |
| Attn Bias / Mask | ●●● | ●●○ | ●●○ | ●●○ |
| Distance-based Attn | ○○○ | ●●● | ●○○ | ○○○ |
| Cross-Attention | ●○○ | ●●○ | ●○○ | ●●● |
| **Ensamble GNN-Transformer (Sec. 3.4)** | | | | |
| Local GNN + Global Trans. | ●●○ | ●○○ | ●●○ | ●●● |
| Type-specific Projections | ●●○ | ●○○ | ●○○ | ●●● |
| **Mejora de Escalabilidad (Sec. 3.5)** | | | | |
| Sparse / Linear Attn | ●○○ | ●○○ | ●●○ | ●○○ |
| Sampling / Pooling | ●●○ | ●○○ | ●○○ | ●●○ |
| **Logro de Equivariancia (Sec. 3.6)** | | | | |
| Equivariant Updates | ○○○ | ●●● | ○○○ | ○○○ |
| Pairwise Geometric Feature | ○○○ | ●●● | ●○○ | ○○○ |
| Angle Encoding | ○○○ | ●●○ | ○○○ | ○○○ |

### 5.6 Guía Práctica: Qué Usar Cuándo

A través de las Secciones 5.2–5.5, emerge un patrón consistente: las elecciones de diseño de GT más efectivas se predicen por la forma de organización del grafo de entrada en lugar del dominio de aplicación. Al rastrear qué tan a menudo aparece cada componente arquitectónico en modelos de GT exitosos de cada forma, las opciones de diseño pueden organizarse en tres niveles: frecuencia alta (casi todos los modelos fuertes bajo esta forma adoptan el componente), frecuencia media (muchos trabajos lo incluyen, y usualmente aporta ganancias no triviales), y frecuencia baja o emergente (usado por unos pocos trabajos muy recientes y vale la pena explorar, pero aún no estándar). La Tabla 4 resume este mapeo.

En resumen, la Tabla 4 muestra el patrón. La tokenización a nivel de nodo es el default universal, mientras que la tokenización a nivel de subgrafo y borde son más comunes para grafos relacionales. Para grafos relacionales, los componentes marcados ●●● son el attention bias con SPD y RWPE. Para grafos geométricos, ●●● cae en la atención basada en distancia, características geométricas pairwise y actualizaciones equivariantes. Para grafos dinámicos, ●●● se concentra en PE temporal y attention bias. Para grafos heterogéneos, ●●● se centra en proyecciones específicas de tipo, cross-attention y arquitecturas híbridas local-global. Los componentes marcados ●○○ indican direcciones emergentes que podrían convertirse en estándar a medida que el campo madura. La tabla está pensada como referencia a probar, no como una prescripción fija.

---

## 6. Discusión

El rápido avance de los GTs ha abierto varias vías prometedoras para investigación futura en diversos dominios científicos. En esta sección, primero se reflexiona sobre las limitaciones de las GNNs de message-passing, luego se delinean direcciones fundacionales y emergentes para el desarrollo de GT.

### 6.1 Reflexiones sobre el Paradigma Actual de Message Passing

**La estructura del grafo se trata como verdad fundamental, no como un prior.** El paradigma de message-passing restringe el flujo de información a los bordes del grafo de entrada, lo que implícitamente asume que la estructura dada es ideal, ya que la agregación ocurre si y solo si dos nodos están conectados. Esta suposición es razonable para grafos moleculares y de cristal, pero menos para grafos construidos, como redes cerebrales [79] y grafos de parches de imagen [201], cuyas estructuras pueden ser ruidosas. Una estructura ruidosa introduce un sesgo inductivo incorrecto en el modelo y degrada el rendimiento. Para reducir la dependencia de la estructura, los GTs relajan esta suposición mediante atención completamente conectada, y reintroducen la información estructural como un prior suave a través de codificaciones posicionales [36, 83] (Sección 3.2) y biases de atención [173, 182] (Sección 3.3).

**La fusión de características puede costar más de lo que aporta.** La agregación de vecindario inyecta información estructural pero también mezcla las características originales del nodo. Esta dilución es una propiedad intrínseca del operador de agregación, y subyace a varios de los problemas comúnmente atribuidos al message passing. El over-smoothing [13, 130] es el ejemplo más prominente: lo que eventualmente destruye son las características originales del nodo en lugar de la estructura. Si esto es dañino depende de cuánta información relevante para la tarea llevan las características originales del nodo. En redes de citación y sociales, atributos fijos de alta dimensión a menudo proporcionan la señal dominante, haciendo que la dilución sea dañina. En contraste, cuando las características de nodo son categóricas, como el tipo de átomo en grafos moleculares, indexan una tabla de embedding aprendible como lo hace un token en NLP. El embedding se optimiza conjuntamente con el modelo, y la mayor parte de la información relevante para la tarea reside en la topología, por lo que la misma dilución cuesta mucho menos. Desde esta perspectiva, muchos estudios sobre GNNs profundas no tratan realmente de profundidad, ya que DropEdge [130], los residuales iniciales de APPNP [49] y GCNII [20], y las jumping knowledge networks [167] adoptan mecanismos diferentes pero todos preservan las características originales del nodo. El over-smoothing y el over-globalization [164] son dos formas del mismo fallo: la señal local informativa de un nodo se lava una vez por agregación de vecindario excesiva y otra vez por atención global excesiva. Los GTs alivian la dilución mediante conexiones residuales, pero cuánta información de característica original retiene finalmente un modelo rara vez se cuantifica o reporta en los benchmarks actuales.

**La expresividad se mide por isomorfismo, no por generalización.** La jerarquía WL es una construcción combinatoria para probar isomorfismo de grafos [118], que examina si dos grafos no isomorfos pueden distinguirse en el peor caso. GIN [166] y Morris et al. [116] fueron los primeros en adoptar esta construcción para medir el poder expresivo de las GNNs, mostrando que las GNNs de message-passing son a lo sumo tan poderosas como el test 1-WL, y ha permanecido como la medida dominante desde entonces. Sin embargo, esta medida difiere del objetivo del aprendizaje en grafos, que es generalizar sobre una distribución de datos. Una GNN parametrizada tampoco es exactamente equivalente a un test WL, por dos razones: la equivalencia entre GIN y 1-WL asume un espacio de características contable que los atributos de nodo continuos violan, y los resultados WL conciernen la existencia de parámetros que realizan una función en lugar de si la optimización puede encontrarlos [85]. Como se discutió en la Sección 4, un modelo teóricamente más fuerte no es necesariamente más preciso, porque los componentes que elevan el poder WL a menudo introducen inestabilidad. Los GTs se analizan con la misma medida, que diagnostica útilmente las señales estructurales a las que una arquitectura es ciega (Sección 4.1.2) pero no predice el rendimiento empírico.

### 6.2 Direcciones Futuras

**Escalar Graph Transformers.** A pesar del éxito notable logrado al escalar Transformers, queda la pregunta de si escalar los GTs mejoraría el rendimiento de manera similar. **Uni-Mol2** [73] escala el GT a miles de millones de parámetros, mostrando mejoras en tareas downstream moleculares. Esta escalabilidad en Uni-Mol2 es factible debido a la abundancia de datos de grafo molecular. Sin embargo, escalar GTs en dominios con datos estructurados en grafo limitados sigue siendo un obstáculo significativo. Enfoques innovadores como **LM-design** [202] utilizan GNNs como adaptadores estructurales para modelos de lenguaje de proteínas preentrenados, integrando tanto información estructural limitada como datos de secuencia abundantes de conjuntos de datos de proteínas existentes. A pesar de estos avances, el campo aún carece de un marco integral que aborde efectivamente los desafíos fundamentales de escalar GTs en entornos con restricción de datos.

**Graph Transformers para Modelado de Datos Multimodales.** Integrar una GNN con un Transformer preentrenado del dominio del lenguaje permite al modelo generar descripciones (captions) para grafos. **MolCA** [99], un ejemplo de GT cross-modal, emplea una GNN molecular para codificar representaciones moleculares y alimenta estos embeddings a un decoder Transformer preentrenado en datos de lenguaje para generar descripciones para la molécula. En el contexto de sistemas de grafo complejos, e.g., proteínas [177], utilizar GTs para integrar estructuras de grafo ofrece una oportunidad para mejorar la comprensión de propiedades de proteínas mediante modelos de lenguaje preentrenados, indicando así una dirección prometedora.

**Enfoques Alternativos para Capturar Dependencias de Largo Alcance.** Recientemente, la arquitectura Transformer ha encontrado mayor competencia, evidenciada por modelos como Mamba [54]. Notablemente, **Graph Mamba** [4] ha demostrado un rendimiento muy competitivo comparado con los GTs. Como el modelo Transformer presenta limitaciones, incluyendo problemas de escalabilidad y over-smoothing como se detalló en el survey, el éxito de Graph Mamba sugiere la posibilidad de otro modelo capaz de capturar efectivamente interacciones globales dentro de datos de grafo. Una dirección prometedora involucra el desarrollo de un modelo que podría desviarse del paradigma de self-attention y capturar más eficientemente interacciones globales, evitando los problemas inherentes al Transformer.

**Uniendo Teoría y Arquitectura Más Allá de la Codificación Espectral.** La teoría existente de GTs se ha enfocado principalmente en codificaciones posicionales espectrales y la expresividad WL de la tokenización multi-nivel (Sección 4). Otros componentes arquitectónicos carecen de un tratamiento teórico correspondiente. Los mecanismos de attention bias y mask (Sección 3.3) codifican estructura mediante varias estrategias distintas, pero su poder representacional comparado entre sí no ha sido caracterizado formalmente. Los arreglos GNN-Transformer (Sección 3.4), ya sean seriales, paralelos o intercalados, actualmente se eligen por heurística, sin teoría que prediga qué arreglo beneficia a qué tipo de sesgo inductivo. Los GTs equivariantes (Sección 3.6) descansan sobre fundamentos sólidos de teoría de grupos para grafos geométricos, pero extender estos fundamentos a otros grupos de simetría o a formas no geométricas sigue sin explorarse.

**Graph Transformers como Modelos Fundacionales de Grafo.** El Transformer ha sido una piedra angular en la construcción de modelos fundacionales en varios dominios, como el procesamiento de lenguaje natural y la visión por computadora. Sin embargo, la importancia crítica de la representación de datos estructurados en grafo en el modelado científico y el análisis de redes sociales ha llevado a un interés significativo en los Graph Foundational Models [107]. Trabajos pioneros como **GROVER** [129] y **DPA-2** [183], preentrenados en dominios moleculares y cristalinos, establecen un nuevo paradigma para el machine learning científico. Estudios actuales sugieren que la tokenización escalable consciente de nodo o borde, las codificaciones estructurales relativas y los módulos de atención sparse o híbrida son elecciones prometedoras para los modelos fundacionales de grafo. Estos desarrollos resaltan el potencial de los GTs como bloques constructores fundamentales para construir la próxima generación de Graph Foundational Models en diversos dominios de aplicación.

---

## 7. Conclusión

En este survey, se presenta una revisión exhaustiva de los avances recientes en Graph Transformers. Se comienza examinando estrategias para incorporar información estructural en la arquitectura Transformer, incluyendo tokenización multi-nivel de grafo, codificaciones posicionales estructurales, mecanismos de atención estructural, y modelos híbridos que integran GNNs con Transformers. También se discuten dos desafíos prominentes en la investigación de GT: la escalabilidad, que concierne a mejorar la eficiencia arquitectónica para manejar grafos a gran escala, y la equivariancia, que se enfoca en diseñar modelos que respeten las restricciones de simetría inherentes a dominios de datos específicos. Adicionalmente, se examina la expresividad teórica de los GTs y se conectan sus elecciones de diseño con paradigmas de aprendizaje en grafos más amplios (MPNN, GSL, GAT). Finalmente, se revisan las aplicaciones de los GTs organizadas en torno a cuatro formas de organización de grafo —relacional, geométrica, dinámica y heterogénea/multimodal— proporcionando una Guía Práctica que mapea componentes arquitectónicos a formas de grafo según frecuencia de adopción, y se discuten los desafíos actuales y las direcciones de investigación futuras del campo.

---

*Nota: Este documento es una transcripción en Markdown del contenido completo del PDF original "A Survey of Graph Transformers: Architectures, Theories and Applications" (arXiv:2502.16533v3), publicado bajo licencia Creative Commons Attribution 4.0 International (CC BY 4.0). Las referencias numéricas entre corchetes (e.g., [173]) corresponden a la bibliografía original del paper, no incluida aquí.*
