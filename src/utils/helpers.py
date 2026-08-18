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




def normalizar_nombre_dataset(nombre: str) -> str:
    """Normaliza un nombre para usarlo en archivos."""
    if not nombre:
        return "dataset"
    nombre_limpio = nombre.strip().replace(" ", "_")
    return nombre_limpio.lower() or "dataset"

def es_nombre_tecnico_o_intermedio(nombre: str) -> bool:
    """
    Filtra artefactos técnicos/intermedios que no deben contarse como dataset final.
    """
    n = str(nombre).strip().lower()
    n_sin_ext = Path(n).stem
    if not n:
        return True
    if n.startswith("temp_dataset_") or n.startswith("_datasets_extraidos"):
        return True
    if n.isdigit() or n_sin_ext.isdigit():
        return True
    return False

def extraer_zip_si_corresponde(ruta_zip: Path, base_dir: Path) -> Path:
    """
    Extrae un zip en carpeta persistente de trabajo y evita re-extraer si ya existe.
    """
    if ruta_zip.suffix.lower() != ".zip":
        return ruta_zip

    # Carpeta temporal persistente para evitar re-extracciones innecesarias.
    nombre_normalizado = normalizar_nombre_dataset(ruta_zip.stem)
    destino = base_dir / f"temp_dataset_{nombre_normalizado}"
    marcador_ok = destino / ".extraccion_ok"

    if destino.exists() and marcador_ok.exists():
        print(f"ZIP detectado: {ruta_zip.name}")
        print(f"Tipo dataset origen: zip")
        print(f"Ruta usada (cache existente): {destino}")
        return destino

    if destino.exists() and not marcador_ok.exists():
        shutil.rmtree(destino, ignore_errors=True)

    destino.mkdir(parents=True, exist_ok=True)
    print(f"ZIP detectado: {ruta_zip.name}")
    print("Tipo dataset origen: zip")
    print(f"Extrayendo en: {destino}")
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        zf.extractall(destino)

    marcador_ok.write_text("ok", encoding="utf-8")
    return destino

def _nombre_dataset_canonico(nombre: str) -> str:
    """
    Normaliza alias de nombres para validación final.
    """
    n = str(nombre).strip().lower()
    if "ecg dataset" in n:
        return "ECG Dataset"
    if "nhfb" in n:
        return "NHFB"
    if n == "archive" or "archive" in n:
        return "archive"
    if "ptb-xl" in n or "ptb_xl" in n or "ptb" in n:
        return "PTB-XL"
    return str(nombre)

def construir_mapeo_clases(y: np.ndarray) -> tuple[dict[str, int], list[str]]:
    """
    Crea mapeo estable etiqueta->índice para entrenamiento en PyTorch.
    """
    clases = sorted([str(c) for c in np.unique(y)])
    clase_a_indice = {clase: idx for idx, clase in enumerate(clases)}
    return clase_a_indice, clases
