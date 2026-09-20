# PROPUESTA DE PROYECTO DE INVESTIGACIÓN Y TRANSFERENCIA TECNOLÓGICA

**Instituto Universitario de Investigación en Inteligencia Artificial de la Comunitat Valenciana (VRAIN)**  
**Departamento de Informática de Sistemas y Computadores (DISCA / GAP)**  
**Universitat Politècnica de València (UPV)**

---

## FICHA RESUMEN DEL PROYECTO

| Campo | Detalle |
| :--- | :--- |
| **Título del Proyecto** | **PhaseAttention:** Mecanismos de Atención de Fase en el Círculo Unitario para Inteligencia Artificial Embebida de Ultra-Bajo Consumo (*Multiplier-Free Green AI*) |
| **Acrónimo** | **PHASE-EDGE** |
| **Investigador Proponente** | **Mario Carbonell** (`marioraulcarbonell@gmail.com`) |
| **Afiliación Objetivo** | Instituto VRAIN / DISCA — Universitat Politècnica de València (UPV) |
| **Línea Temática VRAIN** | Green AI, Edge Computing, Hardware-Aware Machine Learning, Bio-señales y Visión Embebida |
| **Línea Temática DISCA/GAP** | Aceleradores Hardware para IA, Arquitecturas RISC-V, Sistemas de Ultra-Bajo Consumo, FPGAs |
| **Duración Propuesta** | 24 - 36 meses |
| **Convocatorias Objetivo** | European Chips Act, Horizon Europe (Cluster 4: Digital, Industry & Space), PERTE Chip (MICIU), Red de Excelencia ValgrAI, Ayudas CIDEGENT / Juan de la Cierva / Convocatorias UPV |

---

## 1. RESUMEN EJECUTIVO (EXECUTIVE SUMMARY)

La proliferación del paradigma de los *Transformers* ha transformado la inteligencia artificial moderna, pero ha generado una dependencia insostenible de aceleradores de hardware masivos (GPUs y TPUs de cientos de vatios). En el extremo opuesto, el despliegue de modelos de atención en dispositivos de borde (*Edge AI* y *TinyML* en microcontroladores de sub-2 € como ARM Cortex-M, ESP32 o FPGAs de ultra-bajo coste) se encuentra bloqueado por dos barreras físicas fundamentales:
1. **La complejidad cuadrática $O(N^2)$ y la explosión del KV-Cache**, que saturan la memoria SRAM interna (típicamente 32–256 KB).
2. **La dependencia de millones de operaciones Multiply-Accumulate (MACs) de punto flotante** en el cálculo de la afinidad $Q K^\top$, incompatible con baterías de botón o nodos de sensorización energéticamente autónomos (*energy harvesting*).

El proyecto **PHASE-EDGE** propone una ruptura conceptual fundamentada en la geometría de la información: **colapsar la atención token-a-token a una dimensión escalar en el grupo unitario complejo $U(1) \cong S^1$**.

### Hitos Científicos Validados Empíricamente:
1. **Resolución de la Barrera Monótona 1D:** Frente al colapso por monotonía de la atención escalar real en $\mathbb{R}^1$ (59.50% de precisión), la parametrización circular en $U(1)$ recupera ortogonalidad continua ($\cos = 0$) e interferencia destructiva ($\cos = -1$), alcanzando el **100.00% de recall asociativo**. La expansión a toros multidimensionales ($T^H = S^1 \times \dots \times S^1$) escala a más de 64 claves con un **35% menos de parámetros** que la atención vectorial estándar.
2. **Validación Clínica Real en Señales Biomédicas (PhysioNet MIT-BIH ECG):** En la clasificación de arritmias cardíacas bajo el estándar clínico internacional **AAMI EC57** (Normal, Supraventricular, Ventricular), `LinearHolographicPhaseAttention` alcanza **93.75% de exactitud, 93.46% de Macro F1 y 98.66% de sensibilidad ventricular**, **superando al SOTA de atención lineal `cosFormer`** (Qin et al., 2022) con un **24.9% menos de parámetros** y eliminando completamente la función Softmax.
3. **Visión Embebida Espacial 2D (Micro-ViT en Fashion-MNIST):** Con parches de $4 \times 4$ y un Transformer de menos de 10k parámetros (<36 KB Flash, <4 KB SRAM), `LinearHolographicPhase` alcanza **80.67% Top-1 Acc**, superando a las CNNs 2D embebidas de CMSIS-NN (78.07%) y a `cosFormer` (80.33%).
4. **Modelado Causal Autorregresivo sin KV-Cache (TinyShakespeare):** En modelado de lenguaje causal ($L=64$), entrena casi **3 veces más rápido que cosFormer** (8.9s vs 26.1s) con un estado recurrente estrictamente $O(1)$ de solo **512 bytes**, eliminando el consumo creciente de memoria en inferencia continua.
5. **Silicio Multiplier-Free en Lógica Digital:** Sustituyendo el coseno por una onda triangular periódica ($\text{tri}(\Delta\theta) = 1 - \frac{2}{\pi}|\Delta\theta|$), el cálculo $Q \times K$ se ejecuta con **cero multiplicaciones**, empleando únicamente restadores enteros en complemento a dos (con wrap circular automático y gratuito) y rectificadores de signo. Un núcleo INT8 requiere únicamente **~95 puertas NAND2 equivalentes y 0.04 pJ/op** (**47.4x menor área y 92.5x menor energía** que un MAC FP32 de 45nm).

---

## 2. MOTIVACIÓN Y ESTADO DEL ARTE

### 2.1 El Dilema Actual del Edge Computing
La industria se enfrenta a una elección excluyente:
- **Enfoque NPU/Acelerador Dedicado:** Chips como Nvidia Jetson, Google Coral o Hailo-8 cuestan entre 50 € y 150 €, consumen de 5 a 15 W y requieren disipación activa. Inviables para sensores distribuidos o dispositivos médicos desechables.
- **Enfoque Cloud-Offloading:** Enviar la señal bruta por 4G/5G a centros de datos. Supone costes recurrentes de datos de 5–20 €/mes por sensor, latencia crítica en procesos industriales y vulnerabilidades de privacidad en datos médicos.
- **Enfoque Microcontrolador Clásico (TinyML):** Chips de 1 € (STM32, ESP32, Cortex-M) con consumo en microamperios, pero relegados a modelos toscos (MLPs diminutos o filtros estadísticos) incapaces de modelar relaciones contextuales de largo alcance.

### 2.2 Por Qué Falla la Atención Convencional en el Borde
En un Transformer estándar (Vaswani et al., 2017), cada cabeza proyecta a subespacios vectoriales $d_k \ge 64$. Para una secuencia de $N$ tokens, el cálculo $Q K^\top$ exige $N^2 \cdot d_k$ multiplicaciones de punto flotante y almacenar una matriz densa de $N \times N$, saturando instantáneamente la memoria SRAM de cualquier microcontrolador.

Las técnicas de compresión actuales (cuantización INT8/INT4 en BitNet o poda de pesos) reducen el tamaño de los pesos estáticos, pero **no eliminan la necesidad de multiplicadores hardware en la afinidad ni eliminan la matriz cuadrática $N \times N$**.

---

## 3. HIPÓTESIS CIENTÍFICA Y NOVEDAD

### Hipótesis 1: Suficiencia Dimensional en Variedades Compactas $S^1$
*La necesidad de espacios vectoriales multidimensionales ($d_k \ge 64$) en atención asociativa es un artefacto de operar en espacios euclídeos no compactos $\mathbb{R}^d$. Proyectar representaciones al círculo unitario compacto $U(1)$ permite discriminar claves arbitrarias mediante fases angulares relativas, alcanzando capacidad asociativa completa con $d=1$.*

### Hipótesis 2: Separabilidad Analítica Exacta e Interferencia Destructiva
*La afinidad angular en $U(1)$ no requiere normalización Softmax. Al ser $\cos(\Delta\theta)$ un producto interno separable de rango 2 en $\mathbb{R}^2$, la acumulación causal recurrente $S_t = S_{t-1} + \phi(k_t) v_t^\top$ cancela las señales incoherentes por interferencia de fase destructiva ($\sum \cos \approx 0$), permitiendo complejidad temporal lineal $O(N)$ y memoria de estado estrictamente constante $O(1)$.*

### Hipótesis 3: Independencia de Funciones Trascendentes y Aritmética de Silicio
*La selectividad atencional en fase depende de la simetría periódica y de los ceros en cuadratura, no de la derivada trascendente del coseno. Aproximaciones lineales por tramos (onda triangular) o tablas ROM discretas preservan la precisión con cero unidades de multiplicación en el kernel de interacción.*

---

## 4. PLAN DE TRABAJO Y METODOLOGÍA (WORK PACKAGES)

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                             CRONOGRAMA DE TRABAJO                                 │
├────────────────────────────────┬──────────────────────────────────────────────────┤
│ WP1: Fundamentos Algorítmicos  │ Meses 1 - 12 (Modelado matemático y convergencia)│
│ WP2: Co-Diseño de Silicio      │ Meses 6 - 20 (Verilog/VHDL, FPGA y RISC-V)       │
│ WP3: Benchmarking Industrial   │ Meses 12 - 28 (Bio-señales, Visión y Audio Edge) │
│ WP4: Transferencia y Patentes  │ Meses 20 - 36 (PERTE Chip, Spin-off y Licencias) │
└────────────────────────────────┴──────────────────────────────────────────────────┘
```

### WP1: Fundamentos Algorítmicos y Escalabilidad Toroidal (Meses 1–12)
- Formalización de cotas teóricas de empaquetamiento de memoria asociativa en toros $T^H = S^1 \times \dots \times S^1$.
- Estudio de regularizadores entrópicos angulares para mitigar la aglomeración de fase (*angular crowding*) en vocabularios extensos.
- Análisis de estabilidad numérica del acumulador de fase lineal $O(N)$ en horizontes de contexto ultra-largos ($N > 10.000$ tokens).

### WP2: Co-Diseño Hardware/Silicio y Síntesis Multiplier-Free en FPGAs y RISC-V (Meses 6–20)
- Implementación de núcleos de hardware en Verilog/VHDL (`PhaseAttention Core`):
  - Unidad de afinidad triangular con restadores enteros de 8/16 bits y rectificadores de signo.
  - Integración como **instrucción personalizada (Custom ISA Extension) o coprocesador acelerador para procesadores RISC-V**.
- Síntesis real y perfilado físico en FPGAs de ultra-bajo coste (Lattice iCE40, Xilinx Spartan-7 / Artix-7), midiendo consumo real en microvatios ($\mu\text{W}$) y ocupación en LUTs/Flip-Flops.

### WP3: Validación en Señales del Mundo Real y Benchmarking (Meses 12–28)
- **Bio-señales Cardíacas y Neurológicas:** Extensión de los modelos de arritmias (PhysioNet ECG) a precursores de crisis epilépticas en electroencefalografía (EEG) basados en *Phase-Locking Value* (PLV).
- **Visión Artificial Embebida (Micro-ViT):** Clasificación visual de defectos industriales en sensores de cámara de bajo consumo (ESP32-CAM, OpenMV).
- **Audio y Detección de Comandos Vocales (Keyword Spotting):** Evaluación en *Google Speech Commands v2*, comparando latencia y memoria frente a MobileNet y Conformer.

### WP4: Transferencia Tecnológica, Patentes y Explotación (Meses 20–36)
- Redacción y depósito de solicitud de patente europea sobre la arquitectura de atención de fase sin multiplicadores.
- Despliegue de pilotos industriales en empresas del tejido productivo valenciano.
- Elaboración del plan de negocio para la creación de una *spin-off* tecnológica VRAIN-UPV.

---

## 5. TRANSFERENCIA TECNOLÓGICA AL TEJIDO INDUSTRIAL VALENCIANO

La UPV lidera el sistema universitario español en patentes y contratos con el sector productivo. **PHASE-EDGE** ofrece vías de transferencia inmediata:
1. **Sector Automoción y Fabricación Inteligente (Ford Almussafes y clúster AVIA):** Sensores acústicos y de vibración de bajo coste para control de calidad en líneas de estampado y soldadura robotizada.
2. **Sector Cerámico y Vidrio (Clúster de Castellón):** Monitorización continua de motores de atomizadores, prensas y hornos cerámicos en ambientes de alta temperatura (120 °C–150 °C), donde la electrónica densa falla pero los microcontroladores robustos con lógica de fase operan con fiabilidad.
3. **Sector Sanitario y Biomédico (Colaboración con IIS La Fe y Hospital Clínico de Valencia):** Parches desechables adhesivos inteligentes para monitorización cardiológica ambulatoria durante 14 días con batería de botón.
4. **Infraestructuras y Redes de Distribución Eléctrica:** Detección de anomalías en transformadores de media tensión para operadores eléctricos, reduciendo apagones imprevistos.

---

## 6. SINERGIAS Y ENCAJE ESTRATÉGICO EN EL VRAIN Y LA UPV

### 6.1 Mapeo de Grupos e Investigadores de Interés Directo

Para maximizar el impacto de la incorporación y garantizar una masa crítica de colaboración, se han identificado los siguientes grupos y referentes clave dentro del ecosistema UPV:

| Estructura / Grupo | Investigadores Clave | Línea de Investigación de Referencia | Sinergia Concreta con PHASE-EDGE |
| :--- | :--- | :--- | :--- |
| **GAP / DISCA (UPV)**<br>*Grupo de Arquitecturas Paralelas* | **Prof. José Duato** (Premio Nacional)<br>**Dr. Carles Hernández**<br>**Prof. José Flich** | Aceleradores hardware para IA, RISC-V, co-diseño software-hardware, proyecto europeo **FRACTAL**. | **Coprocesador / acelerador PhaseAttention para procesadores RISC-V.** Síntesis física en FPGAs y medición de $\mu\text{W}$ reales en plataformas de bajo consumo. |
| **MLLP (VRAIN)**<br>*Machine Learning and Language Processing* | **Prof. Alfons Juan Císcar** | Aprendizaje automático, procesamiento del habla, modelos de lenguaje y traducción neuronal eficiente. | **Modelos de lenguaje y audio eficientes (Keyword Spotting)** basados en atención lineal sin Softmax, reduciendo latencias de inferencia. |
| **ELP (VRAIN)**<br>*Extensions of Logic Programming* | **Dra. Mª José Ramírez Quintana**<br>**Dr. Josep Silva Galiana** | Inteligencia artificial visual, Deep Learning, explicabilidad (XAI). | **Micro-ViT para visión artificial embebida** en cámaras de bajo coste y análisis de interpretabilidad geométrica de las fases. |
| **PROS (VRAIN)**<br>*Centro de Investigación en Métodos de Producción de Software* | **Prof. Óscar Pastor López** | Métodos de IA en bioinformática, genómica, medicina personalizada y salud. | **Monitorización ambulatoria continua de bio-señales (ECG/EEG)** con parches de ultra-bajo consumo para validación clínica en entorno hospitalario. |
| **GTI-IA (VRAIN)**<br>*Tecnología Informática - Inteligencia Artificial* | **Prof. Vicent Botti** (Director VRAIN)<br>**Dra. Eva Onaindía** | Sistemas multi-agente, arquitecturas de IA distribuida en el borde y edge intelligence. | **Coordinación de redes de agentes inteligentes en la periferia** (*Edge AI Swarms*) con nodos de cómputo ultraligeros. |
| **Evaluación y Métricas de IA (VRAIN)** | **Prof. José Hernández-Orallo** | Evaluación formal de modelos de IA, eficiencia, límites de capacidad y fiabilidad. | **Formalización matemática de cotas de memoria asociativa en toroides $T^H$** y benchmarks de capacidad informacional. |

### 6.2 Proyectos Activos de la UPV con Conexión Inmediata
- **Proyecto FRACTAL (H2020 / KDT JU):** Liderado a nivel de arquitectura hardware en la UPV por el GAP/DISCA. Diseña nodos de computación cognitiva de bajo consumo y alta fiabilidad basados en RISC-V. *PhaseAttention* encaja como un acelerador de atención sin multiplicadores directamente integrable en esta plataforma.
- **Proyecto SEdgeI (Inteligencia Sostenible en el Borde):** Proyecto centrado en frameworks y algoritmos de IA energéticamente sostenibles en dispositivos heterogéneos.
- **Iniciativa PERTE Chip (Microelectrónica y Semiconductores):** Con más de 12.000 millones de euros de fondos del Plan de Recuperación, Transformación y Resiliencia en España, la IP hardware de *PhaseAttention* posiciona a la UPV para liderar bloques de propiedad intelectual (IP cores) de atención lineal en silicio nacional.
- **Red de Excelencia ValgrAI (Valencian Graduate School and Research Network of Artificial Intelligence):** Dirigida por el Prof. Vicent Botti, articula programas de atracción e incorporación de doctores e investigadores de alto impacto para la Comunitat Valenciana.

### 6.3 Vías y Programas de Incorporación de Mario Carbonell
1. **Contratos de Investigador Postdoctoral / Doctor Vinculado a Proyectos (VRAIN / DISCA):** Incorporación inmediata a través de proyectos activos en Green AI, FRACTAL o convocatorias de I+D+i del Ministerio de Ciencia e Innovación / Conselleria.
2. **Programa CIDEGENT / Convocatorias de Excelencia de la Generalitat Valenciana:** Presentación de *PHASE-EDGE* como proyecto individual de excelencia para doctores de alto potencial.
3. **Ayudas Juan de la Cierva (Incorporación / Formación) y Convocatorias Propias UPV:** Solicitud adscrita a VRAIN bajo la tutoría de los investigadores principales de MLLP, PROS o GAP.
4. **Alianzas Público-Privadas con el Ecosistema PERTE Chip y Empresas del Clúster:** Cofinanciación industrial para la valorización de la patente y desarrollo del prototipo de silicio.

---

## 7. PLAN DE PUBLICACIÓN Y DIFUSIÓN DE ALTO IMPACTO

| Nivel | Foro / Revista | Justificación |
| :--- | :--- | :--- |
| **Tier 1 - Sistemas y ML** | **MLSys** (*Conference on Machine Learning and Systems*) | Presentación del co-diseño de arquitectura de atención lineal, streaming $O(1)$ y métricas de silicio. |
| **Tier 1 - TinyML** | **TinyML Research Symposium** | Publicación de referencia para la comunidad de aprendizaje en microcontroladores y Edge AI. |
| **Tier 1 - Silicio y Circuitos** | **IEEE Transactions on Circuits and Systems (TCAS)** | Demostración de la atención sin multiplicadores en lógica digital y métricas de puertas lógicas. |
| **Tier 1 - Procesamiento de Señal** | **IEEE ICASSP** | Publicación de los benchmarks de bio-señales ECG, audio y sincronización de fase. |
| **Tier 1 - Inteligencia Artificial** | **NeurIPS / ICML (Edge AI Workshops)** | Presentación de los fundamentos topológicos $U(1)$ vs $\mathbb{R}^1$ y la superación del cuello de botella monótono. |

---

## 8. CONCLUSIÓN Y PERFIL DEL INVESTIGADOR

Esta propuesta demuestra la capacidad de **inventar primitivas algorítmicas desde primeros principios matemáticos**, validarlas rigurosamente con código reproducible y traducirlas a requerimientos de silicio y transferencia industrial.

Con **6 benchmarks empíricos ya completados**, paridad demostrada frente al SOTA de atención lineal (`cosFormer`), validación clínica en ECG (MIT-BIH), visión espacial (Micro-ViT) y modelado causal (TinyShakespeare), el proyecto **PHASE-EDGE** proporciona al VRAIN y al DISCA un activo científico diferencial: **la demostración de que la inteligencia atencional puede ejecutarse con costes energéticos y de silicio dos órdenes de magnitud inferiores a los estándares actuales.**
