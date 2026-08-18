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
    # pyrefly: ignore [missing-import]
    import transformers
    # pyrefly: ignore [missing-import]
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




def graficar_distribucion_clases(
    conteo_clases: dict[str, int],
    ruta_guardado: str = "distribucion_clases.png"
) -> str:
    """
    Genera un gráfico de barras con la cantidad de ejemplos por clase.
    """
    if not conteo_clases:
        print("No hay conteos de clases para graficar la distribución.")
        return ""

    clases_ordenadas = sorted(conteo_clases.items(), key=lambda item: item[1], reverse=True)
    clases = [item[0] for item in clases_ordenadas]
    cantidades = [item[1] for item in clases_ordenadas]

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(clases, cantidades, color="#2874A6")
    ax.set_title("Distribución de ejemplos por clase")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad de muestras")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right")

    for rect in barras:
        altura = rect.get_height()
        ax.annotate(f"{int(altura)}", xy=(rect.get_x() + rect.get_width() / 2, altura),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(ruta_guardado, dpi=150)
    print(f"Distribución de clases guardada en: {ruta_guardado}")
    plt.show(block=False)
    plt.pause(0.3)
    plt.close(fig)
    return ruta_guardado

def graficar_curva_aprendizaje_modelo(
    estimador_base,
    X: np.ndarray,
    y: np.ndarray,
    ruta_guardado: str = "curva_aprendizaje.png"
) -> str:
    """
    Calcula y grafica la curva de aprendizaje del modelo para monitorear sobreajuste.
    """
    if X.shape[0] < 10:
        print("Muy pocas muestras para generar curva de aprendizaje.")
        return ""

    train_sizes, train_scores, valid_scores, *_ = learning_curve(
        estimator=estimador_base,
        X=X,
        y=y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=3,
        scoring="accuracy",
        shuffle=True,
        random_state=42,
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_std = valid_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#1E8449", label="Entrenamiento")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="#1E8449")
    ax.plot(train_sizes, valid_mean, "o-", color="#B03A2E", label="Validación")
    ax.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2, color="#B03A2E")
    ax.set_title("Curva de aprendizaje (accuracy)")
    ax.set_xlabel("Tamaño de entrenamiento")
    ax.set_ylabel("Exactitud")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(ruta_guardado, dpi=150)
    print(f"Curva de aprendizaje guardada en: {ruta_guardado}")
    plt.show(block=False)
    plt.pause(0.3)
    plt.close(fig)
    return ruta_guardado

def graficar_importancia_caracteristicas(
    modelo,
    nombres_columnas: Optional[list[str]],
    ruta_guardado: str = "importancia_caracteristicas.png",
    top_n: int = 15
) -> str:
    """
    Grafica las características más relevantes del modelo.
    """
    if not hasattr(modelo, "feature_importances_"):
        print("El modelo no expone importancias de características.")
        return ""

    importancias = modelo.feature_importances_
    if importancias.ndim == 0 or importancias.size == 0:
        print("No hay importancias de características disponibles.")
        return ""

    if not nombres_columnas:
        nombres_columnas = [f"pixel_{i}" for i in range(importancias.size)]

    top_n = min(top_n, importancias.size)
    indices = np.argsort(importancias)[-top_n:][::-1]
    valores = importancias[indices]
    etiquetas = [nombres_columnas[i] if i < len(nombres_columnas) else f"característica_{i}" for i in indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), valores, color="#7D3C98")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Importancia")
    ax.set_title("Características más importantes (Random Forest)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta_guardado, dpi=150)
    print(f"Importancia de características guardada en: {ruta_guardado}")
    plt.show(block=False)
    plt.pause(0.3)
    plt.close(fig)
    return ruta_guardado

def graficar_importancia_pixeles(
    modelo_rf,
    tamano_imagen: tuple[int, int],
    ruta_guardado_figura: str
) -> None:
    """
    Genera un heatmap mostrando la importancia de los píxeles usada por el modelo (Random Forest).
    """
    importancias = getattr(modelo_rf, "feature_importances_", None)
    if importancias is None:
        print("Aviso: El modelo no tiene feature_importances_. No se graficará el heatmap de píxeles.")
        return

    # Verificar si el tamaño de píxeles coincide con el de las características
    pixeles_esperados = tamano_imagen[0] * tamano_imagen[1]
    if len(importancias) != pixeles_esperados:
        print(f"Aviso: La cantidad de características ({len(importancias)}) no coincide con la grilla de la imagen ({pixeles_esperados}).")
        return

    importancias_2d = importancias.reshape(tamano_imagen[0], tamano_imagen[1])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        importancias_2d,
        cmap="hot",
        cbar_kws={"label": "Importancia (peso) del Píxel", "shrink": 0.8},
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )
    
    ax.set_title("Heatmap de Importancia de Píxeles (Random Forest)", fontsize=14, fontweight="bold", pad=15)
    
    fig.tight_layout()
    fig.savefig(ruta_guardado_figura, dpi=150, bbox_inches="tight")
    print(f"  Heatmap de píxeles guardado en: {ruta_guardado_figura}")
    
    plt.show(block=False)
    plt.pause(0.5)
    plt.close(fig)

def mostrar_resultados(
    modelo,
    X_prueba: np.ndarray,
    y_prueba: np.ndarray,
    nombres_clases: Optional[list[str]] = None,
    ruta_guardado_figura: str = "matriz_confusion.png"
):
    """
    Realiza predicciones y muestra un reporte completo de métricas y matriz de confusión.

    Parámetros
    ----------
    modelo :
        Modelo ya entrenado (por ejemplo, RandomForestClassifier).
    X_prueba : np.ndarray
        Características del conjunto de prueba.
    y_prueba : np.ndarray
        Etiquetas reales del conjunto de prueba.
    nombres_clases : Optional[list[str]]
        Lista con los nombres de las clases (para etiquetas de la matriz).
        Si es None, se usarán los valores únicos de y_prueba.
    ruta_guardado_figura : str
        Ruta donde se guardará la imagen PNG de la matriz de confusión.
    """
    from src.evaluation.metrics import evaluar_resultados_clasificacion
    y_pred = modelo.predict(X_prueba)
    return evaluar_resultados_clasificacion(
        y_prueba=y_prueba,
        y_pred=y_pred,
        nombres_clases=nombres_clases,
        ruta_guardado_figura=ruta_guardado_figura,
        nombre_modelo="random_forest"
    )

def vectores_a_imagenes_pil(
    X: np.ndarray,
    tamano_imagen: tuple[int, int],
    convertir_a_grises: bool
) -> list[Image.Image]:
    """
    Reconstruye imágenes PIL desde vectores para alimentar modelos CLIP.
    """
    alto, ancho = tamano_imagen
    canales = 1 if convertir_a_grises else 3
    pixeles_esperados = alto * ancho * canales

    if X.shape[1] != pixeles_esperados:
        raise ValueError(
            "Las características no coinciden con el tamaño/canales esperados para reconstruir imágenes. "
            f"Esperado={pixeles_esperados}, recibido={X.shape[1]}"
        )

    imagenes_pil: list[Image.Image] = []
    for fila in X:
        if convertir_a_grises:
            arreglo = (fila.reshape(alto, ancho) * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arreglo, mode="L").convert("RGB")
        else:
            arreglo = (fila.reshape(alto, ancho, 3) * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arreglo, mode="RGB")
        imagenes_pil.append(img)
    return imagenes_pil
