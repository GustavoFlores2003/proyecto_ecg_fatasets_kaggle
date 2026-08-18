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
from src.evaluation.metrics import evaluar_resultados_clasificacion
from src.utils.helpers import construir_mapeo_clases




def preparar_tensores_resnet(
    X: np.ndarray,
    y: np.ndarray,
    clase_a_indice: dict[str, int],
    tamano_imagen: tuple[int, int],
    convertir_a_grises: bool
):
    """
    Convierte vectores de pixeles al formato tensor [N, 3, H, W] compatible con ResNet.
    """
    alto, ancho = tamano_imagen
    canales_entrada = 1 if convertir_a_grises else 3
    pixeles_esperados = alto * ancho * canales_entrada
    if X.shape[1] != pixeles_esperados:
        raise ValueError(
            "La dimensión de entrada no coincide con el tamaño/canales esperados para ResNet. "
            f"Esperado={pixeles_esperados}, recibido={X.shape[1]}"
        )

    if convertir_a_grises:
        imagenes = X.reshape(-1, alto, ancho, 1)
        # ResNet preentrenada espera 3 canales, por eso repetimos el canal de grises.
        imagenes = np.repeat(imagenes, repeats=3, axis=3)
    else:
        imagenes = X.reshape(-1, alto, ancho, 3)

    tensores_x = torch.from_numpy(imagenes.astype(np.float32)).permute(0, 3, 1, 2)

    # Normalización estándar de ImageNet para usar pesos preentrenados.
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    tensores_x = (tensores_x - mean) / std

    etiquetas_indices = np.array([clase_a_indice[str(etiqueta)] for etiqueta in y], dtype=np.int64)
    tensores_y = torch.from_numpy(etiquetas_indices)
    return tensores_x, tensores_y

def crear_dataloaders_resnet(
    X_entrenamiento: np.ndarray,
    y_entrenamiento: np.ndarray,
    X_validacion: np.ndarray,
    y_validacion: np.ndarray,
    tamano_imagen: tuple[int, int],
    convertir_a_grises: bool,
    tamano_batch: int = 16
):
    """
    Genera DataLoader de entrenamiento/validación y mapeo de clases.
    """
    clase_a_indice, nombres_clases = construir_mapeo_clases(y_entrenamiento)
    x_train_tensor, y_train_tensor = preparar_tensores_resnet(
        X_entrenamiento, y_entrenamiento, clase_a_indice, tamano_imagen, convertir_a_grises
    )
    x_val_tensor, y_val_tensor = preparar_tensores_resnet(
        X_validacion, y_validacion, clase_a_indice, tamano_imagen, convertir_a_grises
    )

    loader_train = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=tamano_batch,
        shuffle=True
    )
    loader_val = DataLoader(
        TensorDataset(x_val_tensor, y_val_tensor),
        batch_size=tamano_batch,
        shuffle=False
    )
    return loader_train, loader_val, clase_a_indice, nombres_clases

def crear_modelo_resnet18(num_clases: int, usar_preentrenado: bool = True):
    """
    Crea una ResNet18 y adapta la última capa al número de clases del proyecto.
    """
    if not TORCH_DISPONIBLE:
        raise ImportError("PyTorch/torchvision no están instalados.")

    pesos = models.ResNet18_Weights.DEFAULT if usar_preentrenado else None
    modelo = models.resnet18(weights=pesos)
    in_features = modelo.fc.in_features
    modelo.fc = nn.Linear(in_features, num_clases)
    return modelo

def entrenar_modelo_resnet(
    modelo,
    loader_entrenamiento,
    loader_validacion,
    epocas: int = 5,
    tasa_aprendizaje: float = 1e-4,
    dispositivo: Optional[str] = None
):
    """
    Entrena una ResNet con validación por época y conserva el mejor modelo.
    """
    if not TORCH_DISPONIBLE:
        raise ImportError("PyTorch/torchvision no están instalados.")

    if dispositivo is None:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    modelo = modelo.to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

    mejor_acc_val = -1.0
    mejor_estado = copy.deepcopy(modelo.state_dict())
    historial = {"train_loss": [], "val_acc": []}

    for epoca in range(epocas):
        modelo.train()
        perdida_total = 0.0

        for x_batch, y_batch in loader_entrenamiento:
            x_batch = x_batch.to(dispositivo)
            y_batch = y_batch.to(dispositivo)

            optimizador.zero_grad()
            logits = modelo(x_batch)
            loss = criterio(logits, y_batch)
            loss.backward()
            optimizador.step()

            perdida_total += loss.item() * x_batch.size(0)

        perdida_promedio = perdida_total / len(loader_entrenamiento.dataset)
        historial["train_loss"].append(perdida_promedio)

        metricas_val = evaluar_modelo_resnet(
            modelo=modelo,
            loader_datos=loader_validacion,
            nombres_clases=None,
            ruta_guardado_figura="matriz_confusion_tmp.png",
            mostrar_figura=False,
            dispositivo=dispositivo
        )
        acc_val = metricas_val["exactitud"]
        historial["val_acc"].append(acc_val)

        if acc_val > mejor_acc_val:
            mejor_acc_val = acc_val
            mejor_estado = copy.deepcopy(modelo.state_dict())

        print(
            f"Época {epoca + 1}/{epocas} - loss_train={perdida_promedio:.4f} - "
            f"acc_val={acc_val:.4f}"
        )

    modelo.load_state_dict(mejor_estado)
    return modelo, historial

def evaluar_modelo_resnet(
    modelo,
    loader_datos,
    nombres_clases: Optional[list[str]],
    ruta_guardado_figura: str,
    mostrar_figura: bool = True,
    dispositivo: Optional[str] = None
):
    """
    Realiza predicción por lotes y reutiliza la evaluación estándar del proyecto.
    """
    if not TORCH_DISPONIBLE:
        raise ImportError("PyTorch/torchvision no están instalados.")

    if dispositivo is None:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    modelo = modelo.to(dispositivo)
    modelo.eval()

    y_reales = []
    y_pred_indices = []

    with torch.no_grad():
        for x_batch, y_batch in loader_datos:
            x_batch = x_batch.to(dispositivo)
            logits = modelo(x_batch)
            pred = torch.argmax(logits, dim=1)

            y_reales.extend(y_batch.cpu().numpy().tolist())
            y_pred_indices.extend(pred.cpu().numpy().tolist())

    if nombres_clases is None:
        nombres_clases = [str(i) for i in sorted(set(y_reales))]

    y_reales_txt = np.array([nombres_clases[idx] for idx in y_reales])
    y_pred_txt = np.array([nombres_clases[idx] for idx in y_pred_indices])

    if mostrar_figura:
        return evaluar_resultados_clasificacion(
            y_prueba=y_reales_txt,
            y_pred=y_pred_txt,
            nombres_clases=nombres_clases,
            ruta_guardado_figura=ruta_guardado_figura,
            nombre_modelo="resnet18"
        )

    exactitud = accuracy_score(y_reales_txt, y_pred_txt)
    return {"exactitud": exactitud}
