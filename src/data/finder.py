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




def detectar_datasets(base_dir: Path) -> list[dict[str, str]]:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Detecta datasets disponibles (directorios y .zip) en la carpeta del script.
    """
    candidatos: list[dict[str, str]] = []
    ignorar = {".venv", "__pycache__"}

    for elemento in sorted(base_dir.iterdir(), key=lambda p: p.name.lower()):
        if elemento.name in ignorar:
            continue
        if es_nombre_tecnico_o_intermedio(elemento.name):
            continue

        if elemento.is_dir():
            candidatos.append({
                "nombre": elemento.name,
                "ruta": str(elemento),
                "tipo": "directorio"
            })
            continue

        if elemento.is_file() and elemento.suffix.lower() in [".zip", ".csv"]:
            # Se incluyen todos los ZIP y CSV para identificación/adaptación posterior.
            candidatos.append({
                "nombre": elemento.stem,
                "ruta": str(elemento),
                "tipo": "archivo"
            })

    return candidatos

def ordenar_datasets_priorizados(datasets: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Prioriza el orden de ejecución para depuración temprana.
    Orden objetivo: PTB-XL, ECG Dataset, NHFB, archive, resto.
    """
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    def prioridad(ds: dict[str, str]) -> tuple[int, str]:
        nombre = str(ds.get("nombre", "")).lower()

        if "ptb-xl" in nombre or "ptb_xl" in nombre or "ptb" in nombre:
            return (0, nombre)
        if "ecg dataset" in nombre:
            return (1, nombre)
        if "nhfb" in nombre:
            return (2, nombre)
        if nombre == "archive" or "archive" in nombre:
            return (3, nombre)
        return (4, nombre)

    return sorted(datasets, key=prioridad)

def identificar_tipo_dataset(ruta_base: Path) -> dict[str, object]:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Identifica tipo de dataset para decidir pipeline de carga/adaptación.
    """
    if not ruta_base.is_file() and es_directorio_dataset_imagenes(ruta_base):
        return {
            "tipo_dataset": "imagenes_por_clase",
            "estructura": "imagenes",
            "detalle": f"Directorio con clases detectado en {ruta_base}",
        }

    try:
        if not ruta_base.is_file() and encontrar_directorio_dataset(ruta_base):
            return {
                "tipo_dataset": "imagenes_por_clase",
                "estructura": "imagenes_anidadas",
                "detalle": f"Estructura de imágenes anidada detectada en {ruta_base}",
            }
    except Exception as exc:
        print(f"[DEBUG] Error al buscar estructura de imágenes anidada en {ruta_base}: {exc}")

    if hay_estructura_registros_ecg(ruta_base):
        return {
            "tipo_dataset": "senal_ecg",
            "estructura": "physionet_records",
            "detalle": "Se detectaron archivos de señal (.hea/.dat/.atr o RECORDS)",
        }

    archivos_tabulares = encontrar_archivos_tabulares(ruta_base)
    if archivos_tabulares:
        tipo_tabular = "senal_ecg" if _es_tabular_senal_ecg(ruta_base, archivos_tabulares) else "csv_tabular"
        return {
            "tipo_dataset": tipo_tabular,
            "estructura": "csv_tsv_txt",
            "detalle": f"{len(archivos_tabulares)} archivos tabulares candidatos",
            "archivos_tabulares": archivos_tabulares,
        }

    return {
        "tipo_dataset": "desconocido",
        "estructura": "no_identificada",
        "detalle": f"No se encontró estructura compatible en {ruta_base}",
    }

def es_directorio_dataset_imagenes(ruta: Path) -> bool:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Verifica si un directorio parece dataset de imágenes por clase.
    """
    if not ruta.exists() or not ruta.is_dir():
        return False

    ruta_train = ruta / "train"
    ruta_test = ruta / "test"
    if ruta_train.exists() and ruta_test.exists():
        clases_train = [p for p in ruta_train.iterdir() if p.is_dir()]
        clases_test = [p for p in ruta_test.iterdir() if p.is_dir()]
        return len(clases_train) >= 2 and len(clases_test) >= 2

    clases_directas = [p for p in ruta.iterdir() if p.is_dir()]
    if len(clases_directas) < 2:
        return False

    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    clases_con_imagenes = 0
    for clase in clases_directas:
        if any(
            archivo.is_file() and archivo.suffix.lower() in extensiones
            for archivo in clase.iterdir()
        ):
            clases_con_imagenes += 1
    return clases_con_imagenes >= 2

def es_zip_dataset_imagenes(ruta_zip: Path) -> bool:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Verifica si un zip contiene estructura compatible de imágenes por clase.
    """
    if not ruta_zip.exists() or ruta_zip.suffix.lower() != ".zip":
        return False

    extensiones = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    try:
        with zipfile.ZipFile(ruta_zip, "r") as zf:
            nombres = [n for n in zf.namelist() if not n.endswith("/")]
            imagenes = [n for n in nombres if n.lower().endswith(extensiones)]
            return len(imagenes) > 0
    except (zipfile.BadZipFile, OSError):
        return False

def es_dataset_ptbxl(nombre_dataset: str) -> bool:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Detecta PTB-XL de forma flexible para excluirlo temporalmente del pipeline.
    """
    nombre = str(nombre_dataset).lower()
    return ("ptb-xl" in nombre) or ("ptb_xl" in nombre) or ("ptb" in nombre)

def es_dataset_mitbih(nombre_dataset: str) -> bool:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Detecta MIT-BIH de forma flexible para excluirlo del pipeline.
    """
    nombre = str(nombre_dataset).lower()
    return ("mit-bih" in nombre) or ("mit_bih" in nombre) or ("arrhythmia-database" in nombre)

def encontrar_directorio_dataset(ruta_base: Path) -> Path:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Encuentra el directorio que contiene el dataset dentro de una ruta base.
    """
    if es_directorio_dataset_imagenes(ruta_base):
        return ruta_base

    # Búsqueda conservadora a 2 niveles para zips con carpeta anidada.
    for nivel_1 in sorted([p for p in ruta_base.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        if es_directorio_dataset_imagenes(nivel_1):
            return nivel_1
        for nivel_2 in sorted([p for p in nivel_1.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            if es_directorio_dataset_imagenes(nivel_2):
                return nivel_2

    raise RuntimeError(f"No se encontró estructura de dataset válida en: {ruta_base}")

def encontrar_archivos_tabulares(ruta_base: Path) -> list[Path]:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Busca archivos tabulares potenciales en una ruta de dataset.
    """
    extensiones = {".csv", ".tsv", ".txt", ".data"}
    archivos = []
    if ruta_base.is_file():
        if ruta_base.suffix.lower() in extensiones:
            return [ruta_base]
        return []

    for archivo in ruta_base.rglob("*"):
        if not archivo.is_file():
            continue
        if archivo.suffix.lower() not in extensiones:
            continue
        nombre = archivo.name.lower()
        if nombre.startswith("readme") or "license" in nombre:
            continue
        archivos.append(archivo)
    return sorted(archivos, key=lambda p: p.name.lower())

def hay_estructura_registros_ecg(ruta_base: Path) -> bool:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Detecta presencia de estructura PhysioNet por archivos .hea/.dat/.atr o RECORDS.
    """
    if any(p.name.upper() == "RECORDS" for p in ruta_base.rglob("RECORDS")):
        return True
    tiene_hea = any(True for _ in ruta_base.rglob("*.hea"))
    tiene_dat = any(True for _ in ruta_base.rglob("*.dat"))
    return tiene_hea and tiene_dat

def _es_tabular_senal_ecg(ruta_base: Path, archivos_tabulares: list[Path]) -> bool:
    from src.utils.helpers import es_nombre_tecnico_o_intermedio
    """
    Heurística para diferenciar señales ECG de tabular genérico.
    """
    texto_ruta = str(ruta_base).lower()
    pistas_nombre = ["ecg", "arrhythm", "ptb", "lead", "rhythm"]
    if any(p in texto_ruta for p in pistas_nombre):
        return True

    for archivo in archivos_tabulares[:5]:
        nombre = archivo.name.lower()
        if any(p in nombre for p in pistas_nombre):
            return True

    return False
