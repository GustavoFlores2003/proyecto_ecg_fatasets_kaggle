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
from src.visualization.plots import vectores_a_imagenes_pil
from src.evaluation.metrics import evaluar_resultados_clasificacion




def cargar_modelo_clip_preentrenado(nombre_modelo: str = "openai/clip-vit-base-patch32"):
    """
    Carga CLIP preentrenado para inferencia zero-shot o extracción de embeddings.
    """
    if not CLIP_DISPONIBLE:
        raise ImportError("Se requiere torch + transformers para usar CLIP.")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    procesador = CLIPProcessor.from_pretrained(nombre_modelo)
    modelo = CLIPModel.from_pretrained(nombre_modelo)
    modelo = modelo.to(dispositivo)
    modelo.eval()
    return modelo, procesador, dispositivo

def generar_embeddings_clip_imagenes(
    modelo,
    procesador,
    imagenes: list[Image.Image],
    dispositivo: str,
    tamano_batch: int = 16
) -> np.ndarray:
    import numpy as np
    import torch

    def extraer_tensor(salida_modelo):
        if torch.is_tensor(salida_modelo):
            return salida_modelo
        if hasattr(salida_modelo, "pooler_output") and salida_modelo.pooler_output is not None:
            return salida_modelo.pooler_output
        if hasattr(salida_modelo, "last_hidden_state") and salida_modelo.last_hidden_state is not None:
            return salida_modelo.last_hidden_state[:, 0, :]
        raise TypeError(f"No se pudo extraer embedding. Tipo recibido: {type(salida_modelo)}")

    embeddings = []

    for inicio in range(0, len(imagenes), tamano_batch):
        lote = imagenes[inicio:inicio + tamano_batch]

        with torch.no_grad():
            entradas_img = procesador(
                images=lote,
                return_tensors="pt"
            )
            entradas_img = {k: v.to(dispositivo) for k, v in entradas_img.items()}

            salida_img = modelo.get_image_features(
                pixel_values=entradas_img["pixel_values"]
            )

            emb_img = extraer_tensor(salida_img)
            emb_img = emb_img / emb_img.norm(p=2, dim=-1, keepdim=True)

            embeddings.append(emb_img.cpu().numpy())

    return np.vstack(embeddings)

def predecir_clip_zero_shot(
    modelo,
    procesador,
    imagenes,
    nombres_clases,
    dispositivo,
    tamano_batch=16,
    plantilla_prompt="An ECG image of {}"
):
    import numpy as np
    import torch

    prompts = [plantilla_prompt.format(clase) for clase in nombres_clases]

    def extraer_tensor(salida_modelo):
        if torch.is_tensor(salida_modelo):
            return salida_modelo
        if hasattr(salida_modelo, "pooler_output") and salida_modelo.pooler_output is not None:
            return salida_modelo.pooler_output
        if hasattr(salida_modelo, "last_hidden_state") and salida_modelo.last_hidden_state is not None:
            return salida_modelo.last_hidden_state[:, 0, :]
        raise TypeError(f"No se pudo extraer embedding. Tipo recibido: {type(salida_modelo)}")

    with torch.no_grad():
        entradas_texto = procesador(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        entradas_texto = {k: v.to(dispositivo) for k, v in entradas_texto.items()}

        salida_texto = modelo.get_text_features(
            input_ids=entradas_texto["input_ids"],
            attention_mask=entradas_texto["attention_mask"]
        )

        emb_texto = extraer_tensor(salida_texto)
        emb_texto = emb_texto / emb_texto.norm(p=2, dim=-1, keepdim=True)

    pred_indices = []

    for inicio in range(0, len(imagenes), tamano_batch):
        lote = imagenes[inicio:inicio + tamano_batch]

        with torch.no_grad():
            entradas_img = procesador(
                images=lote,
                return_tensors="pt"
            )
            entradas_img = {k: v.to(dispositivo) for k, v in entradas_img.items()}

            salida_img = modelo.get_image_features(
                pixel_values=entradas_img["pixel_values"]
            )

            emb_img = extraer_tensor(salida_img)
            emb_img = emb_img / emb_img.norm(p=2, dim=-1, keepdim=True)

            logits = emb_img @ emb_texto.T
            pred_batch = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            pred_indices.extend(pred_batch)

    return np.array([nombres_clases[i] for i in pred_indices])

def evaluar_clip_zero_shot(
    modelo,
    procesador,
    X_prueba: np.ndarray,
    y_prueba: np.ndarray,
    nombres_clases: list[str],
    tamano_imagen: tuple[int, int],
    convertir_a_grises: bool,
    dispositivo: str,
    ruta_guardado_figura: str
):
    """
    Ejecuta clasificación zero-shot con CLIP y reporta métricas estándar.
    """
    imagenes_test = vectores_a_imagenes_pil(X_prueba, tamano_imagen, convertir_a_grises)
    y_pred = predecir_clip_zero_shot(
        modelo=modelo,
        procesador=procesador,
        imagenes=imagenes_test,
        nombres_clases=nombres_clases,
        dispositivo=dispositivo,
        tamano_batch=16,
        plantilla_prompt="an ECG image showing {}"
    )
    return evaluar_resultados_clasificacion(
        y_prueba=y_prueba,
        y_pred=y_pred,
        nombres_clases=nombres_clases,
        ruta_guardado_figura=ruta_guardado_figura,
        nombre_modelo="clip_zero_shot"
    )

def entrenar_y_evaluar_clip_embeddings(
    modelo,
    procesador,
    X_entrenamiento: np.ndarray,
    y_entrenamiento: np.ndarray,
    X_prueba: np.ndarray,
    y_prueba: np.ndarray,
    tamano_imagen: tuple[int, int],
    convertir_a_grises: bool,
    dispositivo: str,
    ruta_guardado_figura: str
):
    """
    Extrae embeddings CLIP y entrena un clasificador lineal simple para clasificación supervisada.
    """
    imagenes_train = vectores_a_imagenes_pil(X_entrenamiento, tamano_imagen, convertir_a_grises)
    imagenes_test = vectores_a_imagenes_pil(X_prueba, tamano_imagen, convertir_a_grises)

    emb_train = generar_embeddings_clip_imagenes(
        modelo=modelo,
        procesador=procesador,
        imagenes=imagenes_train,
        dispositivo=dispositivo,
        tamano_batch=16
    )
    emb_test = generar_embeddings_clip_imagenes(
        modelo=modelo,
        procesador=procesador,
        imagenes=imagenes_test,
        dispositivo=dispositivo,
        tamano_batch=16
    )

    clasificador = LogisticRegression(max_iter=2000)
    clasificador.fit(emb_train, y_entrenamiento)
    y_pred = clasificador.predict(emb_test)

    nombres_clases = sorted([str(c) for c in np.unique(y_entrenamiento)])
    return evaluar_resultados_clasificacion(
        y_prueba=y_prueba,
        y_pred=y_pred,
        nombres_clases=nombres_clases,
        ruta_guardado_figura=ruta_guardado_figura,
        nombre_modelo="clip_embeddings"
    )
