# ECG Model Benchmark — Fase Experimental de Evaluación de Modelos de IA

Pipeline en Python que entrena y compara **4 arquitecturas de IA** (Random Forest, ResNet18, CLIP Zero-Shot y CLIP Embeddings) sobre **4 datasets distintos de electrocardiogramas**, como fase exploratoria previa a mi Proyecto de Título.

> 🎓 **Nota de contexto:** Este repositorio es una **etapa experimental de benchmarking**, no el proyecto final. El objetivo es evaluar de forma rigurosa qué arquitecturas generalizan mejor ante distintos formatos y calidades de datos de ECG, antes de aplicar los aprendizajes a un dataset clínico real y/o llevar el modelo ganador a un software funcional en el proyecto de título.

## 🎯 Propósito

En la práctica clínica, los ECGs varían según el equipo médico, la configuración de electrodos y la calidad de la captura. Entrenar y medir un modelo sobre un único dataset "limpio" da una falsa sensación de precisión que no sobrevive al mundo real. Antes de comprometerme con una arquitectura para el proyecto de título, quise validar experimentalmente cómo se comportan 4 enfoques distintos de IA frente a 4 fuentes de datos heterogéneas — incluyendo casos donde los datos disponibles no son suficientes, algo que en un entorno real hay que saber detectar, no ignorar.

## 🔜 Próximos pasos (Proyecto de Título)

- Aplicar el modelo con mejor desempeño (ResNet18, según este benchmark) sobre un **dataset clínico real** de mayor volumen y calidad controlada.
- Empaquetar el pipeline de inferencia en un **software/aplicación funcional**, con foco en usabilidad para personal clínico.
- Extender la validación con métricas de explicabilidad (XAI) sobre el dataset definitivo.

## 🧠 Modelos comparados

| Modelo | Enfoque |
|---|---|
| **Random Forest** | ML clásico sobre representación tabular de píxeles |
| **ResNet18** | Transfer learning con redes convolucionales |
| **CLIP Zero-Shot** | Inferencia semántica sin entrenamiento, vía prompts de texto (OpenAI CLIP) |
| **CLIP Embeddings + Clasificador Lineal** | Extracción de características multimodales + clasificador simple |

## 📊 Resultados

### Datasets de imágenes (calidad y tamaño adecuados)

| Dataset | Muestras | Clases | Mejor modelo | Accuracy | F1-Macro |
|---|---|---|---|---|---|
| **ECG Dataset** | 325 | 3 | ResNet18 | 0.862 | 0.851 |
| **NHFB** | 258 | 3 | ResNet18 | 0.904 | 0.870 |

En ambos, el orden de desempeño fue consistente: **ResNet18 > Random Forest > CLIP Embeddings > CLIP Zero-Shot**. CLIP Zero-Shot en particular mostró alta sensibilidad a la redacción de los prompts de texto, cayendo hasta 0.19 de accuracy en NHFB — un recordatorio de que "sin entrenar" no es sinónimo de "listo para producción".

### Datasets tabulares (limitación de datos detectada)

| Dataset | Muestras | Features/muestra | Resultado |
|---|---|---|---|
| **Archive** | 50 (10 test) | 2 | No concluyente |
| **mit-bih-arrhythmia-database** | 50 (10 test) | 2 | No concluyente |

⚠️ **Limitación conocida y documentada:** estos dos datasets, tal como fueron obtenidos, contienen únicamente 2 columnas de features utilizables y 50 registros en total (10 en test). Esto es insuficiente para representar una señal de ECG real y para generar métricas estadísticamente confiables — con tan pocas muestras, los resultados (0.3–0.7 de accuracy) reflejan el tamaño de la muestra más que la capacidad real de los modelos. El pipeline **detecta y reporta esta limitación automáticamente** (vía matrices de confusión degeneradas y advertencias en consola) en vez de enmascararla con métricas artificialmente altas. Quedan documentados como resultado no concluyente, no como fallo silencioso.

## ⚙️ Flujo del pipeline

1. **Detección automática de estructura**: identifica si cada dataset es de imágenes por clase, señal tabular (CSV/TXT) o registros WFDB (PhysioNet).
2. **Carga y preprocesamiento**: adapta cada fuente a un formato común según el modelo (tensores para ResNet, arrays para RF, imágenes para CLIP).
3. **Entrenamiento y evaluación comparativa**: los 4 modelos se evalúan con el mismo split train/test por dataset, para una comparación justa.
4. **Métricas conscientes del desbalance**: prioriza F1-Macro y Balanced Accuracy sobre Accuracy simple, con alertas automáticas cuando hay sesgo hacia clases mayoritarias.
5. **Validación final automática**: el pipeline valida que todas las combinaciones dataset×modelo esperadas se hayan completado sin errores silenciosos antes de dar el resultado como válido.
6. **Reportes y visualizaciones**: exporta tablas comparativas (`.csv`), interpretaciones (`.txt`), matrices de confusión, heatmaps de importancia y curvas de aprendizaje (`.png`).

## 🛠️ Stack técnico

- **Lenguaje**: Python 3
- **Datos**: `numpy`, `pandas`
- **Visualización**: `matplotlib`, `seaborn`, `Pillow`
- **ML clásico**: `scikit-learn`
- **Deep Learning**: `torch`, `torchvision`
- **Foundation Models**: `transformers` (CLIP)
- **Señales clínicas**: `wfdb` (estándar PhysioNet)

## 📁 Estructura del proyecto

```
├── data/
│   ├── raw/          # Datasets originales (excluido de git)
│   └── processed/    # Datos desempaquetados/procesados (temporal)
├── outputs/
│   ├── figures/      # Gráficos generados
│   └── reports/      # Tablas y reportes automáticos
├── src/
│   ├── data/         # Detección, carga y preprocesamiento
│   ├── evaluation/   # Métricas y reportes
│   ├── models/       # Random Forest, ResNet18, CLIP
│   ├── utils/        # Funciones auxiliares
│   ├── visualization/# Gráficos
│   └── main.py       # Orquestador
├── notebooks/        # Exploración de datos
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Instalación y uso

```bash
git clone <url-del-repo>
cd <nombre-del-repo>
pip install -r requirements.txt
python src/main.py
```

## 📁 Datasets

Los datasets no se incluyen en este repositorio por tamaño y por tratarse de datos clínicos.

- **PTB-XL / registros WFDB**: PhysioNet, licencia ODC-BY (requiere atribución).
- **ECG Dataset, NHFB, Archive, mit-bih**: Kaggle/Mendeley, generalmente CC0 / CC BY 4.0.

👉 Descárgalos desde: **[https://data.mendeley.com/datasets/xw9sd3btcs/2, https://zenodo.org/records/13825810, https://www.kaggle.com/datasets/evilspirit05/ecg-analysis, https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data]**

## ⚠️ Nota

Este repositorio tiene **fines experimentales y educativos**: es la fase de benchmarking de modelos previa a mi Proyecto de Título. **No debe usarse como herramienta de diagnóstico médico real** sin validación clínica y regulatoria correspondiente.


