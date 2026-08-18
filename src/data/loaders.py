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




def cargar_dataset(
    descriptor_dataset: dict[str, str],
    base_dir: Path,
    tamano_imagen: tuple[int, int],
    convertir_a_grises: bool,
    limite_por_clase: Optional[int],
    proporcion_prueba: float = 0.2,
    semilla: int = 42
) -> dict:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Carga un dataset desde directorio o zip y devuelve train/test + metadatos.
    """
    ruta = Path(descriptor_dataset["ruta"])

    if descriptor_dataset["tipo"] == "zip":
        ruta_base = extraer_zip_si_corresponde(ruta_zip=ruta, base_dir=base_dir)
    else:
        ruta_base = ruta

    info_tipo = identificar_tipo_dataset(ruta_base)
    tipo_dataset = str(info_tipo.get("tipo_dataset", "desconocido"))
    print(f"Tipo de estructura encontrada: {tipo_dataset}")
    print(f"Detalle estructura: {info_tipo.get('detalle', 'sin detalle')}")

    tamano_imagen_dataset = tamano_imagen

    if tipo_dataset == "imagenes_por_clase":
        ruta_dataset = encontrar_directorio_dataset(ruta_base)
        X_imagenes, y, nombres_columnas, conteo_por_clase = cargar_datos_ecg_desde_imagenes(
            ruta_directorio=str(ruta_dataset),
            tamano_imagen=tamano_imagen,
            convertir_a_escala_grises=convertir_a_grises,
            limite_por_clase=limite_por_clase
        )
        X_imagenes, y = validar_datos(
            X_imagenes,
            y,
            contexto=f"dataset_{descriptor_dataset['nombre']}_imagenes"
        )

        idx_train, idx_test = dividir_indices_robusto(
            y=y,
            proporcion_prueba=proporcion_prueba,
            semilla=semilla
        )
        X_train_img = X_imagenes[idx_train]
        X_test_img = X_imagenes[idx_test]
        y_entrenamiento = y[idx_train]
        y_prueba = y[idx_test]

        X_por_modelo = {
            "random_forest": (X_train_img, X_test_img),
            "resnet18": (X_train_img, X_test_img),
            "clip_embeddings": (X_train_img, X_test_img),
            "clip_zero_shot": (X_train_img, X_test_img),
        }
        X_base = X_imagenes
        ruta_dataset_usada = ruta_dataset
    elif tipo_dataset in {"senal_ecg", "csv_tabular"}:
        estructura = info_tipo.get("estructura", "")
        if estructura == "physionet_records":
            X_tabular, y, nombres_columnas, conteo_por_clase, fuente_senal = cargar_senales_ecg_desde_registros(
                ruta_base=ruta_base,
                nombre_dataset=descriptor_dataset["nombre"]
            )
            print(f"[DEBUG] Dataset {descriptor_dataset['nombre']}: fuente de señal={fuente_senal}")
            archivo_principal = ruta_base
        elif estructura == "csv_tsv_txt" or tipo_dataset == "csv_tabular":
            X_tabular, y, nombres_columnas, conteo_por_clase, archivo_principal = cargar_senales_desde_tabulares(
                ruta_base=ruta_base
            )
        else:
            raise RuntimeError(f"Estructura no soportada para tabular/señal: {estructura}")
        X_tabular, y = validar_datos(
            X_tabular,
            y,
            contexto=f"dataset_{descriptor_dataset['nombre']}_tabular"
        )

        tamano_imagen_dataset = (224, 224)
        X_visual = convertir_senal_a_representacion_visual(
            X_senales=X_tabular,
            tamano_imagen=tamano_imagen_dataset
        )
        print(
            "Transformación aplicada: señal/tabular -> representación visual 2D "
            f"({tamano_imagen_dataset[0]}x{tamano_imagen_dataset[1]}) para ResNet/CLIP"
        )
        print(
            f"[DEBUG] Dataset {descriptor_dataset['nombre']}: señales convertidas a imagen="
            f"{X_visual.shape[0]}"
        )
        X_visual, _ = validar_datos(
            X_visual,
            y,
            contexto=f"dataset_{descriptor_dataset['nombre']}_visual"
        )

        idx_train, idx_test = dividir_indices_robusto(
            y=y,
            proporcion_prueba=proporcion_prueba,
            semilla=semilla
        )
        y_entrenamiento = y[idx_train]
        y_prueba = y[idx_test]
        print(
            f"[DEBUG] Dataset {descriptor_dataset['nombre']}: tamaño final X={X_tabular.shape}, "
            f"y={y.shape}, split_exitoso=train:{len(idx_train)} test:{len(idx_test)}"
        )

        X_train_tab = X_tabular[idx_train]
        X_test_tab = X_tabular[idx_test]
        X_train_vis = X_visual[idx_train]
        X_test_vis = X_visual[idx_test]

        X_por_modelo = {
            "random_forest": (X_train_tab, X_test_tab),
            "resnet18": (X_train_vis, X_test_vis),
            "clip_embeddings": (X_train_vis, X_test_vis),
            "clip_zero_shot": (X_train_vis, X_test_vis),
        }
        X_base = X_tabular
        ruta_dataset_usada = archivo_principal.parent
        print(f"Archivo tabular principal usado: {archivo_principal}")
    else:
        raise RuntimeError(
            f"Dataset no compatible: {descriptor_dataset['nombre']} | detalle={info_tipo.get('detalle')}"
        )

    return {
        "nombre_dataset": descriptor_dataset["nombre"],
        "ruta_dataset": str(ruta_dataset_usada),
        "tipo_dataset": tipo_dataset,
        "X": X_base,
        "y": y,
        "X_entrenamiento": X_por_modelo["random_forest"][0],
        "X_prueba": X_por_modelo["random_forest"][1],
        "y_entrenamiento": y_entrenamiento,
        "y_prueba": y_prueba,
        "X_por_modelo": X_por_modelo,
        "tamano_imagen_dataset": tamano_imagen_dataset,
        "nombres_columnas": nombres_columnas,
        "conteo_por_clase": conteo_por_clase,
    }

def cargar_datos_ecg(ruta_csv: str, nombre_columna_etiqueta: str = "etiqueta"):
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Carga un archivo CSV con datos de ECG.

    Parámetros
    ----------
    ruta_csv : str
        Ruta del archivo CSV (por ejemplo: 'datos/ecg_dataset.csv').
    nombre_columna_etiqueta : str
        Nombre de la columna que contiene la clase/etiqueta (por ejemplo: 'label', 'clase', etc.).

    Retorna
    -------
    X : np.ndarray
        Matriz de características (todas las columnas menos la etiqueta).
    y : np.ndarray
        Vector de etiquetas/clases.
    nombres_columnas : list
        Lista con los nombres de las columnas de características.
    conteo_por_clase : dict
        Conteo de ejemplos por clase para análisis y gráficos.
    """
    # Leemos el CSV en un DataFrame
    datos = pd.read_csv(ruta_csv)

    # Verificación simple: mostrar las primeras filas
    print("Primeras filas del dataset:")
    print(datos.head())
    print("\nColumnas disponibles:", list(datos.columns))

    # Separamos características (X) y etiqueta (y)
    # Aquí asumimos que la columna de la etiqueta se llama nombre_columna_etiqueta
    X = datos.drop(columns=[nombre_columna_etiqueta])
    y = datos[nombre_columna_etiqueta]
    conteo_por_clase = y.value_counts().to_dict()

    # Convertimos a arreglos de NumPy
    X = X.values
    y = y.values

    nombres_columnas = list(datos.drop(columns=[nombre_columna_etiqueta]).columns)

    return X, y, nombres_columnas, conteo_por_clase

def cargar_senales_ecg_desde_registros(
    ruta_base: Path,
    nombre_dataset: str,
    limite_muestras: int = 2000
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int], str]:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Carga datasets ECG basados en registros (PTB-XL) con etiquetas y señales válidas.
    """
    nombre_norm = normalizar_nombre_dataset(nombre_dataset)

    # Caso PTB-XL: usa metadata CSV y archivos de registros.
    candidatos_ptb = list(ruta_base.rglob("ptbxl_database.csv"))
    if candidatos_ptb:
        ruta_csv = candidatos_ptb[0]
        raiz_ptb = ruta_csv.parent
        df = pd.read_csv(ruta_csv)
        señales, etiquetas = [], []
        crudas = 0

        for _, fila in df.iterrows():
            if len(señales) >= limite_muestras:
                break
            crudas += 1
            etiqueta = _etiqueta_ptb_desde_scp(fila.get("scp_codes", ""))
            if etiqueta == "desconocido":
                continue
            nombre_archivo = str(fila.get("filename_lr", "")).strip()
            if not nombre_archivo:
                continue
            ruta_dat = (raiz_ptb / nombre_archivo).with_suffix(".dat")
            senal = _extraer_senal_desde_archivo_dat(ruta_dat)
            if senal is None:
                continue
            señales.append(senal)
            etiquetas.append(etiqueta)

        if not señales:
            raise RuntimeError("PTB-XL detectado pero no se pudieron construir señales/etiquetas válidas.")

        X = np.vstack(señales).astype(np.float32)
        y = np.array(etiquetas)
        conteo = pd.Series(y).value_counts().to_dict()
        nombres_cols = [f"signal_{i}" for i in range(X.shape[1])]
        print(
            f"[DEBUG] Dataset {nombre_norm}: muestras crudas={crudas}, "
            f"muestras válidas={X.shape[0]}, clases={sorted(list(conteo.keys()))[:10]}"
        )
        return X, y, nombres_cols, conteo, "registros_ptbxl"

    # Si no se encontró PTB-XL, no hay estructura de registros soportada.
    raise RuntimeError(
        "No se encontró estructura de registros PTB-XL compatible en el dataset de señales ECG."
    )

def cargar_imagenes_desde_directorio_clases(
    ruta_clases: Path,
    tamano_imagen: tuple[int, int] = (128, 128),
    convertir_a_escala_grises: bool = True,
    limite_por_clase: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Carga imágenes cuando un directorio contiene directamente subcarpetas por clase.
    """
    if not ruta_clases.exists():
        raise FileNotFoundError(f"No se encontró el directorio: {ruta_clases}")

    extensiones_permitidas = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    datos = []
    etiquetas = []
    conteo_por_clase: dict[str, int] = {}
    imagenes_procesadas = 0

    for clase_dir in sorted(p for p in ruta_clases.iterdir() if p.is_dir()):
        etiqueta = clase_dir.name
        for img_path in sorted(clase_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in extensiones_permitidas:
                continue

            if (
                limite_por_clase is not None
                and conteo_por_clase.get(etiqueta, 0) >= limite_por_clase
            ):
                break

            try:
                with Image.open(img_path) as img:
                    if convertir_a_escala_grises:
                        img = img.convert("L")
                    else:
                        img = img.convert("RGB")

                    if tamano_imagen is not None:
                        img = img.resize(tamano_imagen)

                    arreglo = np.asarray(img, dtype=np.float32) / 255.0
            except Exception as exc:  # pragma: no cover - solo logging
                print(f"Advertencia: no se pudo procesar {img_path}: {exc}")
                continue

            datos.append(arreglo.flatten())
            etiquetas.append(etiqueta)
            conteo_por_clase[etiqueta] = conteo_por_clase.get(etiqueta, 0) + 1
            imagenes_procesadas += 1

            if imagenes_procesadas % 200 == 0:
                print(f"{imagenes_procesadas} imágenes procesadas hasta ahora en {ruta_clases.name}...")

    if not datos:
        raise RuntimeError(
            f"No se pudieron cargar imágenes desde {ruta_clases}."
        )

    X = np.stack(datos, axis=0)
    y = np.array(etiquetas)
    nombres_columnas = [f"pixel_{i}" for i in range(X.shape[1])]
    return X, y, nombres_columnas, conteo_por_clase

def cargar_senales_desde_tabulares(
    ruta_base: Path,
    limite_muestras: Optional[int] = 10000
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int], Path]:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Carga señales/tablas ECG no visuales y prepara matriz numérica + etiquetas.
    """
    archivos = encontrar_archivos_tabulares(ruta_base)
    if not archivos:
        raise RuntimeError("No se encontraron archivos tabulares para cargar señales.")

    mejor: Optional[tuple[pd.DataFrame, str, Path]] = None

    for archivo in archivos:
        try:
            df = _leer_tabular_robusto(archivo)
            etiqueta_col = _detectar_columna_etiqueta(df)
            if etiqueta_col is None:
                continue
            if mejor is None or df.shape[0] > mejor[0].shape[0]:
                mejor = (df, etiqueta_col, archivo)
        except Exception as exc:
            print(f"Aviso: no se pudo usar {archivo.name} como tabular principal: {exc}")

    if mejor is None:
        raise RuntimeError(
            "No se pudo identificar una columna de etiqueta en los archivos tabulares del dataset."
        )

    df, etiqueta_col, archivo_usado = mejor
    y = df[etiqueta_col].astype(str).fillna("desconocido")
    X_df = df.drop(columns=[etiqueta_col]).copy()

    columnas_validas = []
    for col in X_df.columns:
        serie_num = pd.to_numeric(X_df[col], errors="coerce")
        propor_nan = float(serie_num.isna().mean())
        if propor_nan < 0.9:
            columnas_validas.append(col)
            X_df[col] = serie_num

    if not columnas_validas:
        raise RuntimeError("No se encontraron columnas numéricas utilizables para señales/tablas.")

    X_df = X_df[columnas_validas]
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

    if limite_muestras is not None and X_df.shape[0] > limite_muestras:
        idx = np.arange(X_df.shape[0])
        _, idx_keep = train_test_split(
            idx,
            test_size=limite_muestras,
            random_state=42,
            stratify=y if y.nunique() > 1 else None
        )
        X_df = X_df.iloc[idx_keep].reset_index(drop=True)
        y = y.iloc[idx_keep].reset_index(drop=True)

    X = X_df.values.astype(np.float32)
    y_arr = y.values
    conteo = y.value_counts().to_dict()
    nombres_columnas = [str(c) for c in X_df.columns]
    return X, y_arr, nombres_columnas, conteo, archivo_usado

def cargar_datos_ecg_desde_imagenes(
    ruta_directorio: str,
    tamano_imagen: tuple[int, int] = (128, 128),
    convertir_a_escala_grises: bool = True,
    limite_por_clase: Optional[int] = None
):
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Carga un dataset de imágenes (estructura tipo train/test/clase) y lo transforma en una matriz.

    Parámetros
    ----------
    ruta_directorio : str
        Directorio raíz que contiene carpetas con las clases (por ejemplo archive_unzip/ECG_DATA).
        Si dentro del directorio existen subcarpetas llamadas train/test/validation, se leerán automáticamente.
    tamano_imagen : tuple[int, int]
        Resolución a la que se redimensionará cada imagen para crear vectores de igual longitud.
    convertir_a_escala_grises : bool
        Si es True, las imágenes se convertirán a escala de grises para reducir dimensionalidad.
    limite_por_clase : Optional[int]
        Permite limitar la cantidad de imágenes a cargar por clase (útil para pruebas rápidas). None = sin límite.

    Retorna
    -------
    X : np.ndarray
        Matriz con las imágenes vectorizadas y normalizadas en [0, 1].
    y : np.ndarray
        Vector con las etiquetas (nombres de las carpetas/clases).
    nombres_columnas : list
        Lista genérica de nombres de características (pixel_0, pixel_1, ...).
    conteo_por_clase : dict
        Conteo de imágenes cargadas por cada clase.
    """
    ruta = Path(ruta_directorio)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el directorio de imágenes: {ruta_directorio}")

    # Detectamos subcarpetas comunes (train/test/validation). Si no existen, usamos la raíz directamente.
    carpetas_principales = [
        ruta / nombre
        for nombre in ("train", "test", "validation", "val")
        if (ruta / nombre).exists()
    ]
    if not carpetas_principales:
        carpetas_principales = [ruta]

    extensiones_permitidas = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    datos = []
    etiquetas = []
    conteo_por_clase: dict[str, int] = {}
    imagenes_procesadas = 0

    try:
        for carpeta in carpetas_principales:
            for clase_dir in sorted(p for p in carpeta.iterdir() if p.is_dir()):
                etiqueta = clase_dir.name
                for img_path in sorted(clase_dir.iterdir()):
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in extensiones_permitidas:
                        continue
                    if (
                        limite_por_clase is not None
                        and conteo_por_clase.get(etiqueta, 0) >= limite_por_clase
                    ):
                        break

                    try:
                        with Image.open(img_path) as img:
                            if convertir_a_escala_grises:
                                img = img.convert("L")
                            else:
                                img = img.convert("RGB")

                            if tamano_imagen is not None:
                                img = img.resize(tamano_imagen)

                            arreglo = np.asarray(img, dtype=np.float32) / 255.0
                    except Exception as exc:  # pragma: no cover - solo logging
                        print(f"Advertencia: no se pudo procesar {img_path}: {exc}")
                        continue

                    datos.append(arreglo.flatten())
                    etiquetas.append(etiqueta)
                    conteo_por_clase[etiqueta] = conteo_por_clase.get(etiqueta, 0) + 1
                    imagenes_procesadas += 1

                    if imagenes_procesadas % 200 == 0:
                        print(f"{imagenes_procesadas} imágenes procesadas hasta ahora...")
    except KeyboardInterrupt:
        if datos:
            print(
                f"\nCarga interrumpida manualmente tras procesar {imagenes_procesadas} imágenes."
            )
        else:
            print("\nCarga interrumpida antes de procesar datos; reintenta con límite menor.")
            raise

    if not datos:
        raise RuntimeError(
            "No se pudieron cargar imágenes. Verifica la estructura de carpetas y las extensiones."
        )

    X = np.stack(datos, axis=0)
    y = np.array(etiquetas)
    nombres_columnas = [f"pixel_{i}" for i in range(X.shape[1])]

    total_imagenes = X.shape[0]
    clases_encontradas = sorted(conteo_por_clase.keys())
    print(f"Total de imágenes cargadas: {total_imagenes}")
    print(f"Clases detectadas ({len(clases_encontradas)}): {clases_encontradas}")

    return X, y, nombres_columnas, conteo_por_clase

def procesar_imagen_individual(
    ruta_imagen: str,
    tamano_imagen: tuple[int, int],
    convertir_a_escala_grises: bool
) -> np.ndarray:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Procesa una imagen aislada usando el mismo pipeline del dataset para poder predecirla.
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    with Image.open(ruta_imagen) as img:
        if convertir_a_escala_grises:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = img.resize(tamano_imagen)
        arreglo = np.asarray(img, dtype=np.float32) / 255.0

    return arreglo.flatten()

def _extraer_senal_desde_archivo_dat(ruta_dat: Path, longitud_objetivo: int = 2000) -> Optional[np.ndarray]:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Extrae una secuencia 1D reproducible desde archivo .dat (fallback sin dependencias externas).
    """
    if not ruta_dat.exists():
        return None
    try:
        datos = np.fromfile(ruta_dat, dtype=np.uint8)
        if datos.size < 32:
            return None
        datos = datos.astype(np.float32)
        minimo = float(np.min(datos))
        maximo = float(np.max(datos))
        if maximo > minimo:
            datos = (datos - minimo) / (maximo - minimo)
        else:
            datos = np.zeros_like(datos, dtype=np.float32)
        eje_origen = np.linspace(0.0, 1.0, num=datos.shape[0])
        eje_destino = np.linspace(0.0, 1.0, num=longitud_objetivo)
        return np.interp(eje_destino, eje_origen, datos).astype(np.float32)
    except Exception:
        return None

def _etiqueta_ptb_desde_scp(valor_scp: str) -> str:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    """
    Obtiene etiqueta principal desde scp_codes de PTB-XL.
    """
    try:
        parsed = ast.literal_eval(str(valor_scp))
        if isinstance(parsed, dict) and parsed:
            mejor = max(parsed.items(), key=lambda x: float(x[1]))
            return str(mejor[0])
    except Exception:
        pass
    return "desconocido"

def _detectar_columna_etiqueta(df: pd.DataFrame) -> Optional[str]:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    candidatas = [
        "label", "labels", "class", "target", "etiqueta", "diagnosis", "diagnostic",
        "rhythm", "arrhythmia", "condition", "disease", "y"
    ]
    mapa = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidatas:
        if cand in mapa:
            return mapa[cand]

    for columna in df.columns:
        serie = df[columna]
        if pd.api.types.is_numeric_dtype(serie):
            continue
        n_unicos = serie.dropna().nunique()
        if 2 <= n_unicos <= 200:
            return columna

    if len(df.columns) >= 2:
        ultima = df.columns[-1]
        n_unicos = df[ultima].dropna().nunique()
        if 2 <= n_unicos <= 200:
            return ultima
    return None

def _leer_tabular_robusto(ruta_archivo: Path) -> pd.DataFrame:
    from src.utils.helpers import extraer_zip_si_corresponde, normalizar_nombre_dataset
    from src.data.preprocessing import convertir_senal_a_representacion_visual, dividir_indices_robusto, validar_datos
    from src.data.finder import encontrar_archivos_tabulares, encontrar_directorio_dataset, identificar_tipo_dataset
    separadores = [None, ",", ";", "\t", "|"]
    ultimo_error: Optional[Exception] = None
    for sep in separadores:
        try:
            df = pd.read_csv(ruta_archivo, sep=sep, engine="python")
            if df.shape[1] >= 2 and df.shape[0] >= 10:
                return df
        except Exception as exc:
            ultimo_error = exc
            continue
    if ultimo_error is not None:
        raise ultimo_error
    raise RuntimeError(f"No se pudo leer archivo tabular: {ruta_archivo}")
