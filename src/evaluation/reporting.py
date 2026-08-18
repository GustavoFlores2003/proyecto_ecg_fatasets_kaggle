import os
import copy
import importlib
import shutil
import zipfile
import ast
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision.models as models
    TORCH_DISPONIBLE = True
except ImportError:
    torch = None
    nn = None
    TensorDataset = None
    DataLoader = None
    models = None
    TORCH_DISPONIBLE = False

try:
    import transformers
    from transformers import CLIPModel, CLIPProcessor
    CLIP_DISPONIBLE = TORCH_DISPONIBLE
except ImportError:
    CLIPModel = None
    CLIPProcessor = None
    CLIP_DISPONIBLE = False

try:
    import wfdb
    WFDB_DISPONIBLE = True
except ImportError:
    wfdb = None
    WFDB_DISPONIBLE = False

# Local imports
from src.evaluation.metrics import construir_tabla_comparativa_modelos, detectar_alertas_evaluacion




def interpretar_comparacion_modelos(
    tabla_comparativa: pd.DataFrame,
    alertas: list[str]
) -> str:
    """
    Genera una interpretación simple para presentación académica.
    """
    if tabla_comparativa.empty:
        return "No hay resultados para interpretar."

    mejor_fila = tabla_comparativa.iloc[0]
    mejor_modelo = str(mejor_fila["modelo"])
    f1_macro_mejor = float(mejor_fila["f1_macro"])
    acc_mejor = float(mejor_fila["accuracy"])

    lineas = [
        "INTERPRETACION COMPARATIVA",
        (
            f"- Según F1 macro, el mejor modelo es '{mejor_modelo}' "
            f"(F1 macro={f1_macro_mejor:.4f}, accuracy={acc_mejor:.4f})."
        ),
        "- Para comparación justa en datasets médicos, priorizar F1 macro y balanced accuracy, no solo accuracy.",
        "- La matriz de confusión debe revisarse para identificar qué clases clínicas son más confundidas."
    ]

    if alertas:
        lineas.append("- Alertas de evaluación detectadas:")
        for alerta in alertas:
            lineas.append(f"  * {alerta}")

    return "\n".join(lineas)

def generar_reporte_comparativo_modelos(
    resultados_por_modelo: dict[str, dict],
    conteo_clases: dict[str, int],
    ruta_csv_salida: str = "comparacion_modelos.csv",
    ruta_txt_salida: str = "interpretacion_comparacion_modelos.txt"
) -> dict:
    """
    Construye tabla, detecta alertas y genera interpretación en formato consistente.
    """
    tabla = construir_tabla_comparativa_modelos(
        resultados_por_modelo=resultados_por_modelo,
        ruta_csv_salida=ruta_csv_salida
    )
    alertas = detectar_alertas_evaluacion(
        conteo_clases=conteo_clases,
        resultados_por_modelo=resultados_por_modelo
    )
    interpretacion = interpretar_comparacion_modelos(tabla, alertas)

    with open(ruta_txt_salida, "w", encoding="utf-8") as archivo:
        archivo.write(interpretacion + "\n")
    print(f"Interpretación comparativa guardada en: {ruta_txt_salida}")

    return {
        "tabla": tabla,
        "alertas": alertas,
        "interpretacion": interpretacion,
        "ruta_csv": ruta_csv_salida,
        "ruta_txt": ruta_txt_salida,
    }

def generar_discusion_resultados(
    resultados_modelo: dict,
    referencias_literatura: Optional[list[dict]] = None,
    ruta_salida: str = "discusion_resultados.txt"
) -> str:
    """
    Genera un texto resumido con la discusión de resultados y lo guarda en disco.

    Parámetros
    ----------
    resultados_modelo : dict
        Salida de la función mostrar_resultados.
    referencias_literatura : Optional[list[dict[str, float]]]
        Lista con referencias (nombre y accuracy) para comparar el rendimiento.
    ruta_salida : str
        Archivo de texto donde se guardará la discusión.
    """
    if not resultados_modelo:
        raise ValueError("No se proporcionaron resultados del modelo para generar la discusión.")

    lineas: list[str] = ["RESULTADOS Y DISCUSION"]
    exactitud = resultados_modelo.get("exactitud", 0.0)
    tamano_prueba = resultados_modelo.get("tamano_prueba", 0)
    lineas.append(
        f"- El modelo RandomForest obtuvo {exactitud:.2%} de exactitud sobre {tamano_prueba} muestras de prueba."
    )

    reporte_dict = resultados_modelo.get("reporte_dict", {})
    clases_metricas = [
        (clase, metricas.get("precision", 0.0), metricas.get("recall", 0.0), metricas.get("f1-score", 0.0))
        for clase, metricas in reporte_dict.items()
        if isinstance(metricas, dict) and "f1-score" in metricas and clase not in {"accuracy", "macro avg", "weighted avg"}
    ]

    if clases_metricas:
        clase_mejor = max(clases_metricas, key=lambda elemento: elemento[3])
        clase_peor = min(clases_metricas, key=lambda elemento: elemento[3])
        lineas.append(
            f"- La clase con mejor F1 fue '{clase_mejor[0]}' (F1={clase_mejor[3]:.2f}), mientras que '{clase_peor[0]}'"
            f" mostró la menor F1 (F1={clase_peor[3]:.2f})."
        )

    matriz_confusion = resultados_modelo.get("matriz_confusion")
    nombres_clases = resultados_modelo.get("nombres_clases", [])
    if matriz_confusion is not None and len(nombres_clases) == matriz_confusion.shape[0]:
        errores_por_clase = matriz_confusion.sum(axis=1) - np.diag(matriz_confusion)
        if errores_por_clase.size > 0:
            indice_mayor_error = int(np.argmax(errores_por_clase))
            errores = int(errores_por_clase[indice_mayor_error])
            clase_confusa = nombres_clases[indice_mayor_error]
            lineas.append(
                f"- La mayor confusión ocurre en la clase '{clase_confusa}', con {errores} ejemplos mal clasificados."
            )

    if referencias_literatura:
        lineas.append("\nComparación con literatura (post 2017):")
        for referencia in referencias_literatura:
            nombre = referencia.get("nombre", "Referencia sin nombre")
            accuracy_ref = referencia.get("accuracy")
            if accuracy_ref is None:
                continue
            diferencia = exactitud - accuracy_ref
            lineas.append(
                f"  * {nombre}: {accuracy_ref:.2%} de exactitud (diferencia {diferencia:+.2%} frente al modelo actual)."
            )
    else:
        lineas.append("\nNo se proporcionaron métricas de referencia para comparar los resultados.")

    ventajas = [
        "Pipeline reproducible que abarca carga, preprocesamiento y evaluación.",
        "Generación automática de matriz de confusión utilizable en el informe."
    ]
    limitaciones = ["El modelo base no explora optimización de hiperparámetros ni arquitecturas profundas."]
    if tamano_prueba and tamano_prueba < 500:
        limitaciones.append("El tamaño de la muestra de prueba es reducido; se recomienda recopilar más datos.")

    lineas.append("\nVentajas:")
    for ventaja in ventajas:
        lineas.append(f"  - {ventaja}")

    lineas.append("\nLimitaciones:")
    for limitacion in limitaciones:
        lineas.append(f"  - {limitacion}")

    contenido = "\n".join(lineas)
    with open(ruta_salida, "w", encoding="utf-8") as archivo_salida:
        archivo_salida.write(contenido + "\n")

    print(f"\nDiscusión de resultados guardada en: {ruta_salida}")
    return contenido

def generar_conclusiones_y_trabajo_futuro(
    resultados_modelo: dict,
    referencias_literatura: Optional[list[dict]] = None,
    ruta_salida: str = "conclusiones_trabajo_futuro.txt"
) -> str:
    """
    Crea un borrador de conclusiones y líneas de trabajo futuro basadas en las métricas.
    """
    if not resultados_modelo:
        raise ValueError("No hay resultados disponibles para redactar conclusiones.")

    exactitud = resultados_modelo.get("exactitud", 0.0)
    tamano_total = resultados_modelo.get("tamano_total", 0)

    conclusiones = [
        "CONCLUSIONES",
        f"- Se completó un flujo integral desde la carga de datos hasta la evaluación, empleando {tamano_total} muestras.",
        f"- El RandomForest base logra {exactitud:.2%} de exactitud y ofrece métricas detalladas por clase.",
    ]

    if referencias_literatura:
        mejor_referencia = max(
            (ref for ref in referencias_literatura if ref.get("accuracy") is not None),
            key=lambda ref: ref["accuracy"],
            default=None
        )
        if mejor_referencia:
            diff = exactitud - mejor_referencia["accuracy"]
            conclusiones.append(
                f"- Frente a {mejor_referencia['nombre']}, el modelo actual muestra una diferencia de {diff:+.2%}."
            )

    trabajo_futuro = [
        "TRABAJO FUTURO",
        "- Incrementar el dataset para balancear clases y evaluar la robustez del modelo.",
        "- Explorar arquitecturas profundas (CNN/Transformers) y búsqueda de hiperparámetros.",
        "- Integrar validación cruzada y explicación de características relevantes.",
    ]

    contenido = "\n".join(conclusiones + [""] + trabajo_futuro)
    with open(ruta_salida, "w", encoding="utf-8") as archivo_salida:
        archivo_salida.write(contenido + "\n")

    print(f"Conclusiones y trabajo futuro guardados en: {ruta_salida}")
    return contenido
