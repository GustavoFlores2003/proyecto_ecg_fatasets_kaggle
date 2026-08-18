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




def convertir_senal_a_representacion_visual(
    X_senales: np.ndarray,
    tamano_imagen: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Convierte cada muestra de señal/tabular en imagen 2D reproducible (trazo ECG).
    """
    alto, ancho = tamano_imagen
    X_senales = np.asarray(X_senales, dtype=np.float32)

    X_visual = np.zeros((X_senales.shape[0], alto * ancho), dtype=np.float32)

    for i in range(X_senales.shape[0]):
        fila = X_senales[i]
        if fila.size == 0:
            continue

        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.plot(fila, color="black", linewidth=1.0)
        ax.set_axis_off()
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        fig.canvas.draw()
        buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_rgba = buffer.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
        img_gray = img_rgba[..., 0].astype(np.float32) / 255.0
        img_pil = Image.fromarray((img_gray * 255.0).astype(np.uint8), mode="L")
        if img_pil.size != (ancho, alto):
            img_pil = img_pil.resize((ancho, alto))
        X_visual[i] = (np.asarray(img_pil, dtype=np.float32) / 255.0).flatten()
        plt.close(fig)

    return X_visual

def dividir_indices_robusto(
    y: np.ndarray,
    proporcion_prueba: float = 0.2,
    semilla: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide índices en train/test usando estratificación cuando es viable.
    """
    indices = np.arange(y.shape[0])
    try:
        idx_train, idx_test = train_test_split(
            indices,
            test_size=proporcion_prueba,
            random_state=semilla,
            stratify=y
        )
    except ValueError as exc:
        print(f"Aviso: no fue posible estratificar split ({exc}); se usa split sin estratificar.")
        idx_train, idx_test = train_test_split(
            indices,
            test_size=proporcion_prueba,
            random_state=semilla,
            stratify=None
        )
    return idx_train, idx_test

def dividir_datos(
    X: np.ndarray,
    y: np.ndarray,
    proporcion_prueba: float = 0.2,
    semilla: int = 42
):
    """
    Divide los datos en entrenamiento y prueba.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de características completa.
    y : np.ndarray
        Vector de etiquetas.
    proporcion_prueba : float
        Proporción de datos que se usará para prueba (por ejemplo 0.2 = 20%).
    semilla : int
        Semilla aleatoria para reproducibilidad.

    Retorna
    -------
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba
    """
    # Dividimos en entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=proporcion_prueba,
        random_state=semilla,
        stratify=y  # Mantiene la proporción de clases en train y test
    )

    print(f"Tamaño entrenamiento: {X_entrenamiento.shape[0]} muestras")
    print(f"Tamaño prueba: {X_prueba.shape[0]} muestras")

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

def validar_datos(
    X: np.ndarray,
    y: np.ndarray,
    contexto: str = "dataset"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valida consistencia básica de datos para evitar errores silenciosos.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"{contexto}: X debe ser una matriz 2D. Forma recibida: {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"{contexto}: y debe ser un vector 1D. Forma recibida: {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{contexto}: número de muestras inconsistente entre X ({X.shape[0]}) e y ({y.shape[0]})."
        )
    if X.shape[0] == 0:
        raise ValueError(f"{contexto}: no hay muestras para entrenar/evaluar.")

    try:
        X = X.astype(np.float32)
    except ValueError as exc:
        raise ValueError(
            f"{contexto}: X contiene valores no numéricos que no pueden convertirse a float."
        ) from exc

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError(f"{contexto}: X contiene NaN o valores infinitos.")

    if pd.isna(y).any():
        raise ValueError(f"{contexto}: y contiene etiquetas faltantes (NaN).")

    return X, y
