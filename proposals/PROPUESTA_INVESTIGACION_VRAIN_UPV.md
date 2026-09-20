# PROPUESTA DE PROYECTO DE INVESTIGACIÓN Y TRANSFERENCIA TECNOLÓGICA

**Instituto Universitario de Investigación en Inteligencia Artificial de la Comunitat Valenciana (VRAIN)**  
**Universitat Politècnica de València (UPV)**

---

## FICHA RESUMEN DEL PROYECTO

| Campo | Detalle |
| :--- | :--- |
| **Título del Proyecto** | **PhaseAttention:** Mecanismos de Atención de Fase en el Círculo Unitario para Inteligencia Artificial Embebida de Ultra-Bajo Consumo (*Multiplier-Free Green AI*) |
| **Acrónimo** | **PHASE-EDGE** |
| **Investigador Proponente** | Mario Carbonell |
| **Línea Temática VRAIN** | Green AI, Edge Computing, Hardware-Aware Machine Learning, Ciberfísicos y Sistemas Inteligentes |
| **Duración Propuesta** | 24 - 36 meses |
| **Convocatorias Objetivo** | European Chips Act, Horizon Europe (Cluster 4: Digital, Industry & Space), PERTE Chip (MICIU), Ayudas CIDEGENT / Convocatorias UPV |

---

## 1. RESUMEN EJECUTIVO (EXECUTIVE SUMMARY)

La proliferación del paradigma de los *Transformers* ha transformado la inteligencia artificial moderna, pero ha generado una dependencia insostenible de aceleradores de hardware masivos (GPUs y TPUs) que consumen cientos de vatios por inferencia. En el extremo opuesto, el despliegue de modelos de atención en dispositivos de borde (*Edge AI* y *TinyML* en microcontroladores de menos de 2 € como ARM Cortex-M, ESP32 o FPGAs de bajo coste) se encuentra actualmente bloqueado por dos barreras físicas infranqueables:
1. La complejidad de memoria cuadrática $O(N^2)$ y los grandes estados intermedios que desbordan la memoria SRAM interna (típicamente limitada a 32–256 KB).
2. La necesidad de unidades de multiplicación de punto flotante masivas (MACs) en el cálculo de la afinidad $Q K^\top$, cuyo consumo energético es incompatible con baterías de botón o nodos de sensorización energéticamente autónomos (*energy harvesting*).

El proyecto **PHASE-EDGE** propone una ruptura conceptual fundamentada en la geometría de la información: **colapsar la atención token-a-token a una dimensión escalar en el grupo unitario complejo $U(1)$**. 

### Hitos Científicos Demostrados Preliminarmente:
1. **Resolución de la Barrera Monótona 1D:** Se demuestra que mientras la atención escalar real en $\mathbb{R}^1$ colapsa a un clasificador monótono (*Soft-Ranker*, 56% de precisión), el círculo unitario $U(1)$ recupera ortogonalidad continua ($\cos = 0$) e inhibición ($\cos = -1$), logrando **100.00% de precisión asociativa con solo 659 parámetros** (V375).
2. **Atención Causal Lineal $O(N)$ sin Softmax:** Se demuestra analíticamente que $\cos(\theta_q - \theta_k) = \phi(q)^\top \phi(k)$ es un **kernel separable exacto de rango 2** (sin aproximaciones de Taylor ni Random Fourier Features). Almacena el contexto en un estado recurrente diminuto de solo **$2 \times d_v$ floats por cabeza (16 números)**, donde la **interferencia destructiva de ondas cancela el ruido de fondo sin requerir la matriz cuadrática $N \times N$ ni el operador Softmax** (V377).
3. **Silicio Multiplier-Free:** Se demuestra que la sustitución del coseno por una **onda triangular periódica** ($\text{tri}(\Delta\theta) = 1 - \frac{2}{\pi}|\Delta\theta|$) o una **ROM discreta de 16 palabras** alcanza el **100.00% de precisión** requiriendo únicamente restadores enteros y rectificadores de signo (`abs`), **eliminando el 100% de las multiplicaciones de punto flotante en el kernel de atención** (V378).

Esta propuesta sitúa al VRAIN y a la UPV a la vanguardia europea del diseño conjunto de algoritmos y silicio (*algorithm-hardware co-design*), con un potencial de transferencia industrial inmediato hacia los sectores productivos de la Comunitat Valenciana y de España.

---

## 2. MOTIVACIÓN Y ESTADO DEL ARTE

### 2.1 El Dilema Actual del Edge Computing
La industria se enfrenta a una elección excluyente:
- **Enfoque NPU/Acelerador Dedicado:** Chips como Nvidia Jetson, Google Coral o Hailo-8 cuestan entre 50 € y 150 €, consumen de 5 a 15 W y requieren disipación activa. Inviables para sensores distribuidos o dispositivos médicos desechables.
- **Enfoque Cloud-Offloading:** Enviar la señal bruta por 4G/5G a servidores en la nube. Supone costes recurrentes de datos de 5–20 €/mes por sensor, problemas de latencia crítica en procesos industriales y vulnerabilidades de privacidad.
- **Enfoque Microcontrolador Clásico (TinyML):** Chips de 1 € (STM32, ESP32) con consumo en microamperios, pero relegados a modelos toscos (MLPs diminutos o filtros estadísticos) incapaces de modelar relaciones contextuales de largo alcance.

### 2.2 Por Qué Falla la Atención Convencional en el Borde
En un Transformer estándar (Vaswani et al., 2017), cada cabeza proyecta a subespacios vectoriales $d_k \ge 64$. Para una secuencia de $N$ tokens, el cálculo $Q K^\top$ exige $N^2 \cdot d_k$ multiplicaciones de punto flotante y almacenar una matriz densa de $N \times N$, saturando instantáneamente la memoria SRAM de cualquier microcontrolador. 

Las técnicas de compresión actuales (cuantización INT8/INT4 en BitNet o poda de pesos) reducen el tamaño de los pesos estáticos, pero **no eliminan la necesidad de multiplicadores hardware en la afinidad ni eliminan la matriz $N \times N$**.

---

## 3. HIPÓTESIS CIENTÍFICA Y NOVEDAD

### Hipótesis 1: Suficiencia Dimensional en Variedades Compactas $S^1$
*La necesidad de espacios vectoriales multidimensionales ($d_k \ge 64$) en atención asociativa es un artefacto de operar en espacios euclídeos no compactos $\mathbb{R}^d$. Proyectar representaciones al círculo unitario compacto $U(1)$ permite discriminar claves arbitrarias mediante fases angulares relativas, alcanzando capacidad asociativa completa con $d=1$.*

### Hipótesis 2: Separabilidad Analítica Exacta e Interferencia Destructiva
*La afinidad angular en $U(1)$ no requiere normalización Softmax. Al ser $\cos(\Delta\theta)$ un producto interno separable en $\mathbb{R}^2$, la acumulación causal recurrente $S_t = S_{t-1} + \phi(k_t) v_t^\top$ cancela las señales incoherentes por interferencia de fase destructiva, permitiendo complejidad computacional estrictamente lineal $O(N)$ y memoria de estado $O(1)$.*

### Hipótesis 3: Independencia de Funciones Trascendentes
*La selectividad atencional en fase depende de la simetría periódica y de los ceros en cuadratura, no de la derivada trascendente del coseno. Por tanto, aproximaciones lineales por tramos (onda triangular) o discretas (tablas ROM de 16 palabras) preservan la precisión asociativa con cero unidades de multiplicación en silicio.*

---

## 4. PLAN DE TRABAJO Y METODOLOGÍA (WORK PACKAGES)

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                             CRONOGRAMA DE TRABAJO                                 │
├────────────────────────────────┬──────────────────────────────────────────────────┤
│ WP1: Fundamentos Algorítmicos  │ Meses 1 - 12 (Modelado matemático y convergencia)│
│ WP2: Co-Diseño de Silicio      │ Meses 6 - 20 (Verilog/VHDL, FPGA y emulación)    │
│ WP3: Benchmarking Industrial   │ Meses 12 - 28 (Audio, Bio-señales y Visión Micro)│
│ WP4: Transferencia y Patentes  │ Meses 20 - 36 (Pilotos industriales y Spin-off)  │
└────────────────────────────────┴──────────────────────────────────────────────────┘
```

### WP1: Fundamentos Algorítmicos y Escalabilidad Toroidal (Meses 1–12)
- Formalización de la capacidad de almacenamiento asociativo en toroides $T^H = S^1 \times \dots \times S^1$ frente a la dimensión de embedding.
- Estudio de regularizadores entrópicos angulares para mitigar la aglomeración de fase (*angular crowding*) en vocabularios extensos.
- Análisis de estabilidad numérica del acumulador de fase lineal $O(N)$ en horizontes de contexto ultra-largos ($N > 10.000$ tokens).

### WP2: Co-Diseño Hardware/Silicio y Síntesis Multiplier-Free (Meses 6–20)
- Implementación de núcleos de hardware en Verilog/VHDL (`PhaseAttention Core`):
  - Unidad de afinidad triangular con restadores enteros de 8/16 bits y rectificadores de signo.
  - Módulo microcode ROM de 16 palabras (LUT-16) para arquitecturas ASIC.
- Implementación de referencia en C bare-metal optimizada para instrucciones ARM Cortex-M (CMSIS-NN) y RISC-V.
- Síntesis y perfilado en FPGAs de bajo coste (Lattice iCE40, Xilinx Artix-7), midiendo ahorro de celdas lógicas y consumo en microvatios ($\mu\text{W}$).

### WP3: Validación en Señales del Mundo Real y Benchmarking (Meses 12–28)
- **Audio y Detección de Comandos Vocales (Keyword Spotting):** Evaluación en *Google Speech Commands v2*, comparando latencia y memoria frente a MobileNet y Conformer.
- **Bio-señales Cardíacas y Neurológicas:** Detección en tiempo real de arritmias (PhysioNet ECG) y precursores de crisis epilépticas (EEG) basadas en *Phase-Locking Value* (PLV).
- **Visión Artificial Embebida (Micro-ViT):** Clasificación visual de defectos industriales en sensores de cámara de bajo consumo (ESP32-CAM).

### WP4: Transferencia Tecnológica, Patentes y Explotación (Meses 20–36)
- Redacción y depósito de solicitud de patente europea sobre la arquitectura de atención de fase sin multiplicadores.
- Despliegue de pilotos industriales en empresas colaboradoras de la Comunitat Valenciana.
- Elaboración del plan de negocio para la creación de una *spin-off* tecnológica VRAIN-UPV.

---

## 5. TRANSFERENCIA TECNOLÓGICA AL TEJIDO INDUSTRIAL VALENCIANO

La UPV lidera el sistema universitario español en patentes y contratos con el sector productivo. **PHASE-EDGE** ofrece vías de transferencia inmediata hacia sectores estratégicos de la región:

1. **Sector Automoción y Fabricación Inteligente (Ford Almussafes y clúster AVIA):**
   - Sensores acústicos y de vibración de bajo coste para control de calidad en líneas de estampado y soldadura robotizada.
2. **Sector Cerámico y Vidrio (Clúster de Castellón):**
   - Monitorización continua de motores de atomizadores, prensas y hornos cerámicos en ambientes de alta temperatura (120 °C–150 °C), donde la electrónica densa falla pero los microcontroladores robustos con lógica de fase operan con fiabilidad.
3. **Sector Sanitario y Biomédico (Colaboración con IIS La Fe y Hospital Clínico):**
   - Parches desechables adhesivos inteligentes para monitorización cardiológica ambulatoria durante 14 días con batería de botón, democratizando el acceso a diagnósticos continuos.
4. **Infraestructuras y Redes de Distribución Eléctrica:**
   - Detección de anomalías en transformadores de media tensión para operadores eléctricos, reduciendo apagones imprevistos.

---

## 6. SINERGIAS Y ENCAJE ESTRATÉGICO EN EL VRAIN / UPV

Esta línea de investigación fortalece de manera directa los objetivos estratégicos del VRAIN:
- **Alineamiento con la Estrategia Europea de Green AI:** Reducción del impacto de carbono de la IA sustituyendo computación de fuerza bruta en centros de datos por cómputo inteligente en el extremo sensorial.
- **Participación en el PERTE Chip (Microelectrónica y Semiconductores):** España cuenta con más de 12.000 millones de euros de fondos europeos para el diseño de chips y arquitecturas de silicio. Una IP como `PhaseAttention` posiciona al VRAIN para liderar proyectos en el pilar de diseño de procesadores neuromórficos y aceleradores de IA de ultra-bajo consumo.
- **Convocatorias Europeas Horizon Europe:** Propuesta natural para consorcios en *Cluster 4 (Digital, Industry and Space)*, convocatorias *Green and Soft Computing* y consorcios público-privados con fabricantes europeos de microelectrónica (STMicroelectronics, NXP, Infineon).

---

## 7. PLAN DE PUBLICACIÓN Y DIFUSIÓN DE ALTO IMPACTO

| Nivel | Foro / Revista | Justificación |
| :--- | :--- | :--- |
| **Tier 1 - Sistemas y ML** | **MLSys** (*Conference on Machine Learning and Systems*) | Presentación del co-diseño de arquitectura de atención lineal y métricas de silicio. |
| **Tier 1 - TinyML** | **TinyML Research Symposium** | Publicación de referencia para la comunidad de aprendizaje en microcontroladores. |
| **Tier 1 - Silicio y Circuitos** | **IEEE Transactions on Circuits and Systems (TCAS)** | Demostración de la atención sin multiplicadores en lógica digital. |
| **Tier 1 - Procesamiento de Señal** | **IEEE ICASSP** | Publicación de los benchmarks de audio, acústica y sincronización de fase. |

---

## 8. CONCLUSIÓN Y PERFIL DEL INVESTIGADOR

Esta propuesta demuestra la capacidad de **inventar primitivas algorítmicas desde primeros principios matemáticos**, validarlas rigurosamente con código reproducible y traducirlas a requerimientos de silicio y transferencia industrial. 

El proyecto **PHASE-EDGE** proporciona al VRAIN un activo científico diferencial: **la demostración de que la inteligencia atencional puede ejecutarse con costes energéticos y de silicio dos órdenes de magnitud inferiores a los estándares actuales.**
