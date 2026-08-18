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




def entrenar_modelo_random_forest(
    X_entrenamiento: np.ndarray,
    y_entrenamiento: np.ndarray,
    n_arboles: int = 100,
    profundidad_maxima: Optional[int] = None,
    semilla: int = 42
):
    """
    Entrena un modelo Random Forest para clasificación.

    Parámetros
    ----------
    X_entrenamiento : np.ndarray
        Características de entrenamiento.
    y_entrenamiento : np.ndarray
        Etiquetas de entrenamiento.
    n_arboles : int
        Cantidad de árboles en el bosque.
    profundidad_maxima : Optional[int]
        Profundidad máxima de cada árbol. Si es None, el árbol crece completo.
    semilla : int
        Semilla aleatoria para reproducibilidad.

    Retorna
    -------
    modelo : RandomForestClassifier
        Modelo entrenado.
    """
    modelo = RandomForestClassifier(
        n_estimators=n_arboles,
        max_depth=profundidad_maxima,
        random_state=semilla
    )

    modelo.fit(X_entrenamiento, y_entrenamiento)

    return modelo
