# Proyecto ECG Datasets Kagle

Este proyecto realiza un análisis, preprocesamiento y entrenamiento de modelos de Machine Learning y Deep Learning sobre diferentes conjuntos de datos de señales Electrocardiográficas (ECG).

## Estructura del Proyecto

- `data/`
  - `raw/`: Datasets originales (ignorados en git).
  - `processed/`: Archivos temporales y procesados.
- `src/`: Código fuente modularizado.
  - `data/`: Descubrimiento, carga y preprocesamiento de datos.
  - `models/`: Definición y entrenamiento de modelos (Random Forest, ResNet18, CLIP).
  - `evaluation/`: Generación de métricas y reportes.
  - `visualization/`: Generación de gráficas de análisis.
  - `utils/`: Utilidades generales.
  - `main.py`: Orquestador principal del proyecto.
- `notebooks/`: Exploración de datos en formato interactivo.
- `outputs/`
  - `figures/`: Gráficas generadas (`.png`).
  - `reports/`: Tablas y análisis (`.csv`, `.txt`).
- `tests/`: Pruebas unitarias.

## Requisitos

Instala las dependencias usando:
```bash
pip install -r requirements.txt
```

## Ejecución

Para correr el flujo completo (o el modo configurado en `main.py`), ejecuta desde la raíz del proyecto:
```bash
python src/main.py
```
