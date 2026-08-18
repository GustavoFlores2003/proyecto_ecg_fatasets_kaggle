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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Local imports
from src.models.clip_model import cargar_modelo_clip_preentrenado, entrenar_y_evaluar_clip_embeddings, evaluar_clip_zero_shot
from src.data.finder import detectar_datasets, es_dataset_mitbih, es_dataset_ptbxl, ordenar_datasets_priorizados
from src.data.loaders import cargar_dataset, cargar_datos_ecg, cargar_datos_ecg_desde_imagenes, cargar_imagenes_desde_directorio_clases, procesar_imagen_individual
from src.evaluation.reporting import generar_conclusiones_y_trabajo_futuro, generar_discusion_resultados, generar_reporte_comparativo_modelos
from src.visualization.plots import graficar_curva_aprendizaje_modelo, graficar_distribucion_clases, graficar_importancia_caracteristicas, mostrar_resultados
from src.models.random_forest import entrenar_modelo_random_forest
from src.utils.helpers import _nombre_dataset_canonico, normalizar_nombre_dataset
from src.evaluation.metrics import comparar_resultados, evaluar_modelos_en_dataset
from src.models.resnet import crear_dataloaders_resnet, crear_modelo_resnet18, entrenar_modelo_resnet, evaluar_modelo_resnet
from src.data.preprocessing import dividir_datos, validar_datos



def main():
    """
    Función principal del script.

    Aquí conectamos todos los pasos:
    1) Cargar datos,
    2) Preprocesar (dividir),
    3) Entrenar modelo,
    4) Evaluar y generar gráficos.
    """

    # -----------------------
    # 8.1 Parámetros del usuario
    # -----------------------

    # Si True usaremos imágenes (por ejemplo ECG Dataset.rar ya descomprimido).
    # Si False, el flujo usará un CSV tradicional.
    usar_dataset_imagenes = True
    usar_multiples_datasets = True
    tipo_modelo = "comparar_todos"  # Opciones: "random_forest", "resnet18", "clip_zero_shot", "clip_embeddings", "comparar_todos"

    # Carpeta src (donde está este script)
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    # --- Parámetros para el dataset basado en imágenes ---
    ruta_directorio_imagenes = str(project_root / "data" / "raw" / "ECG Dataset")
    tamano_imagen = (128, 128)  # Puedes ajustar el tamaño si quieres capturar más detalle
    convertir_a_grises = True   # Cambia a False si prefieres trabajar con 3 canales (RGB)
    limite_por_clase = None     # Útil para pruebas rápidas: por ejemplo 200 imágenes por clase

    # --- Parámetros para el dataset basado en CSV ---
    ruta_csv = str(project_root / "data" / "raw" / "ecg_dataset.csv")
    nombre_columna_etiqueta = "etiqueta"

    origen_dataset = ruta_directorio_imagenes if usar_dataset_imagenes else ruta_csv
    if usar_dataset_imagenes:
        nombre_dataset = normalizar_nombre_dataset(Path(origen_dataset).name)
    else:
        nombre_dataset = normalizar_nombre_dataset(Path(origen_dataset).stem)
    prefijo_figuras = nombre_dataset

    # Rutas donde se guardarán las figuras y reportes
    ruta_matriz_confusion = str(project_root / "outputs" / "figures" / f"{prefijo_figuras}_matriz_confusion.png")
    ruta_distribucion_clases = str(project_root / "outputs" / "figures" / f"{prefijo_figuras}_distribucion_clases.png")
    ruta_curva_aprendizaje = str(project_root / "outputs" / "figures" / f"{prefijo_figuras}_curva_aprendizaje.png")
    ruta_importancia = str(project_root / "outputs" / "figures" / f"{prefijo_figuras}_importancia_caracteristicas.png")

    ruta_comparacion_csv = str(project_root / "outputs" / "reports" / "comparacion_modelos.csv")
    ruta_archivo_discusion = str(project_root / "outputs" / "reports" / "discusion_resultados.txt")
    ruta_archivo_conclusiones = str(project_root / "outputs" / "reports" / "interpretacion_comparacion_modelos.txt")

    # -----------------------
    # 8.2 Modo multi-dataset (extensión conservadora)
    # -----------------------
    if usar_multiples_datasets and usar_dataset_imagenes:
        modo_debug_ptbxl = True
        datasets_detectados = detectar_datasets(project_root / "data" / "raw")
        datasets_detectados = ordenar_datasets_priorizados(datasets_detectados)

        datasets_filtrados = []
        for ds in datasets_detectados:
            if not es_dataset_ptbxl(ds["nombre"]):
                datasets_filtrados.append(ds)
        datasets_detectados = datasets_filtrados

        tipo_modelo_multi_dataset = "comparar_todos"

        if not datasets_detectados:
            print("No se detectaron datasets automáticamente; se usará el flujo original.")
        else:
            print("=== MODO PRIORIZADO: PTB-XL PRIMERO ===")
            print("\nDatasets detectados automáticamente:")
            for ds in datasets_detectados:
                print(f" - {ds['nombre']} | tipo={ds['tipo']} | ruta={ds['ruta']}")

            filas_comparacion_global = []
            total_datasets = len(datasets_detectados)

            if modo_debug_ptbxl:
                print("[DEBUG] modo_debug_ptbxl=True: PTB-XL se ejecuta primero para validación temprana.")

            for i, descriptor in enumerate(datasets_detectados, start=1):
                try:
                    print(f"[ORDEN] Procesando dataset {i}/{total_datasets}: {descriptor['nombre']}")
                    if descriptor["tipo"] == "zip":
                        print(
                            f"Procesando dataset: {descriptor['nombre']} (zip) | "
                            f"ruta: {descriptor['ruta']}"
                        )
                    else:
                        print(
                            f"\nProcesando dataset detectado: {descriptor['nombre']} "
                            f"({descriptor['tipo']}) | ruta: {descriptor['ruta']}"
                        )
                    datos_dataset = cargar_dataset(
                        descriptor_dataset=descriptor,
                        base_dir=project_root / "data" / "processed",
                        tamano_imagen=tamano_imagen,
                        convertir_a_grises=convertir_a_grises,
                        limite_por_clase=limite_por_clase,
                        proporcion_prueba=0.2,
                        semilla=42
                    )

                    salidas_dataset = evaluar_modelos_en_dataset(
                        datos_dataset=datos_dataset,
                        tipo_modelo=tipo_modelo_multi_dataset,
                        tamano_imagen=tamano_imagen,
                        convertir_a_grises=convertir_a_grises,
                        ruta_archivo_discusion=str(project_root / "outputs" / "reports" / f"{normalizar_nombre_dataset(descriptor['nombre'])}_{Path(ruta_archivo_discusion).name}"),
                        ruta_archivo_conclusiones=str(project_root / "outputs" / "reports" / f"{normalizar_nombre_dataset(descriptor['nombre'])}_{Path(ruta_archivo_conclusiones).name}")
                    )
                    print(
                        f"[DEBUG] Dataset {descriptor['nombre']}: filas agregadas a resultados globales="
                        f"{len(salidas_dataset['filas_resumen'])}"
                    )
                    filas_comparacion_global.extend(salidas_dataset["filas_resumen"])
                except Exception as exc:
                    print(f"ERROR al procesar dataset '{descriptor['nombre']}': {exc}")
                    print("Se continúa con el siguiente dataset disponible.")

            if filas_comparacion_global:
                tabla_global = comparar_resultados(
                    filas_resultados=filas_comparacion_global,
                    ruta_csv_salida=ruta_comparacion_csv
                )
                print("Datasets realmente evaluados:", sorted(tabla_global["dataset"].unique().tolist()))
                print("Filas globales por dataset:")
                print(tabla_global.groupby("dataset").size())
                validar_integracion_final(
                    datasets_detectados=datasets_detectados,
                    filas_resultados=tabla_global.to_dict(orient="records"),
                    ruta_csv_global=ruta_comparacion_csv
                )
                return
            print("No se pudieron obtener resultados válidos en modo multi-dataset.")
            return

    # -----------------------
    # 8.3 Verificar que la fuente de datos exista (flujo original)
    # -----------------------

    if usar_dataset_imagenes:
        if not os.path.exists(ruta_directorio_imagenes):
            print("ERROR: No se encontró el directorio del dataset de imágenes.")
            print(f"Ruta esperada: {ruta_directorio_imagenes}")
            print("Descomprime ECG Dataset.rar en la misma carpeta del script y vuelve a ejecutar.")
            return
    else:
        if not os.path.exists(ruta_csv):
            print(f"ERROR: No se encontró el archivo CSV en la ruta: {ruta_csv}")
            print("Por favor, revisa la ruta y el nombre del archivo.")
            return

    # -----------------------
    # 8.3 Cargar datos
    # -----------------------

    if usar_dataset_imagenes:
        ruta_dataset = Path(ruta_directorio_imagenes)
        ruta_train = ruta_dataset / "train"
        ruta_test = ruta_dataset / "test"

        # Si existe split predefinido train/test, se respeta para evitar leakage.
        if ruta_train.exists() and ruta_test.exists():
            print("Se detectó estructura train/test predefinida. Se usará sin re-dividir.")

            X_entrenamiento, y_entrenamiento, nombres_columnas, conteo_train = cargar_imagenes_desde_directorio_clases(
                ruta_clases=ruta_train,
                tamano_imagen=tamano_imagen,
                convertir_a_escala_grises=convertir_a_grises,
                limite_por_clase=limite_por_clase
            )
            X_prueba, y_prueba, _, conteo_test = cargar_imagenes_desde_directorio_clases(
                ruta_clases=ruta_test,
                tamano_imagen=tamano_imagen,
                convertir_a_escala_grises=convertir_a_grises,
                limite_por_clase=limite_por_clase
            )

            X_entrenamiento, y_entrenamiento = validar_datos(X_entrenamiento, y_entrenamiento, contexto="train")
            X_prueba, y_prueba = validar_datos(X_prueba, y_prueba, contexto="test")

            conteo_por_clase = {}
            for etiqueta in set(conteo_train) | set(conteo_test):
                conteo_por_clase[etiqueta] = conteo_train.get(etiqueta, 0) + conteo_test.get(etiqueta, 0)

            X = np.vstack([X_entrenamiento, X_prueba])
            y = np.concatenate([y_entrenamiento, y_prueba])
        else:
            X, y, nombres_columnas, conteo_por_clase = cargar_datos_ecg_desde_imagenes(
                ruta_directorio=ruta_directorio_imagenes,
                tamano_imagen=tamano_imagen,
                convertir_a_escala_grises=convertir_a_grises,
                limite_por_clase=limite_por_clase
            )
            X, y = validar_datos(X, y, contexto="dataset_imagenes")
            X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(
                X,
                y,
                proporcion_prueba=0.2,
                semilla=42
            )
    else:
        X, y, nombres_columnas, conteo_por_clase = cargar_datos_ecg(
            ruta_csv=ruta_csv,
            nombre_columna_etiqueta=nombre_columna_etiqueta
        )
        X, y = validar_datos(X, y, contexto="dataset_csv")
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(
            X,
            y,
            proporcion_prueba=0.2,
            semilla=42
        )

    print(f"\nTotal de muestras: {X.shape[0]}")
    print(f"Cantidad de características: {X.shape[1]}")

    graficar_distribucion_clases(conteo_por_clase, ruta_guardado=ruta_distribucion_clases)

    escalador = None

    # -----------------------
    # 8.5 Entrenar y evaluar modelo
    # -----------------------

    n_arboles = 100
    profundidad_maxima = None
    semilla_modelo = 42

    if tipo_modelo == "comparar_todos":
        resultados_por_modelo: dict[str, dict] = {}

        print("\n=== Comparación justa: mismo train/test para todos los modelos ===")

        modelo_rf = entrenar_modelo_random_forest(
            X_entrenamiento=X_entrenamiento,
            y_entrenamiento=y_entrenamiento,
            n_arboles=n_arboles,
            profundidad_maxima=profundidad_maxima,
            semilla=semilla_modelo
        )
        resultados_rf = mostrar_resultados(
            modelo=modelo_rf,
            X_prueba=X_prueba,
            y_prueba=y_prueba,
            nombres_clases=None,
            ruta_guardado_figura=f"{prefijo_figuras}_matriz_confusion_random_forest.png"
        )
        resultados_rf["tamano_total"] = X.shape[0]
        resultados_rf["tamano_entrenamiento"] = X_entrenamiento.shape[0]
        resultados_rf["tamano_prueba"] = X_prueba.shape[0]
        resultados_rf["nombres_columnas"] = nombres_columnas
        resultados_rf["conteo_clases"] = conteo_por_clase
        resultados_por_modelo["random_forest"] = resultados_rf

        if TORCH_DISPONIBLE:
            loader_train, loader_val, _, nombres_clases_resnet = crear_dataloaders_resnet(
                X_entrenamiento=X_entrenamiento,
                y_entrenamiento=y_entrenamiento,
                X_validacion=X_prueba,
                y_validacion=y_prueba,
                tamano_imagen=tamano_imagen,
                convertir_a_grises=convertir_a_grises,
                tamano_batch=16
            )
            modelo_resnet = crear_modelo_resnet18(
                num_clases=len(nombres_clases_resnet),
                usar_preentrenado=True
            )
            modelo_resnet, historial_resnet = entrenar_modelo_resnet(
                modelo=modelo_resnet,
                loader_entrenamiento=loader_train,
                loader_validacion=loader_val,
                epocas=5,
                tasa_aprendizaje=1e-4
            )
            resultados_resnet = evaluar_modelo_resnet(
                modelo=modelo_resnet,
                loader_datos=loader_val,
                nombres_clases=nombres_clases_resnet,
                ruta_guardado_figura=f"{prefijo_figuras}_matriz_confusion_resnet18.png",
                mostrar_figura=True
            )
            resultados_resnet["historial_resnet"] = historial_resnet
            resultados_resnet["tamano_total"] = X.shape[0]
            resultados_resnet["tamano_entrenamiento"] = X_entrenamiento.shape[0]
            resultados_resnet["tamano_prueba"] = X_prueba.shape[0]
            resultados_resnet["nombres_columnas"] = nombres_columnas
            resultados_resnet["conteo_clases"] = conteo_por_clase
            resultados_por_modelo["resnet18"] = resultados_resnet
        else:
            print("Aviso: se omite ResNet porque torch/torchvision no están instalados.")

        if CLIP_DISPONIBLE:
            modelo_clip, procesador_clip, dispositivo_clip = cargar_modelo_clip_preentrenado()
            nombres_clases_clip = sorted([str(c) for c in np.unique(y_entrenamiento)])

            resultados_clip_zero = evaluar_clip_zero_shot(
                modelo=modelo_clip,
                procesador=procesador_clip,
                X_prueba=X_prueba,
                y_prueba=y_prueba,
                nombres_clases=nombres_clases_clip,
                tamano_imagen=tamano_imagen,
                convertir_a_grises=convertir_a_grises,
                dispositivo=dispositivo_clip,
                ruta_guardado_figura=f"{prefijo_figuras}_matriz_confusion_clip_zero_shot.png"
            )
            resultados_clip_zero["tamano_total"] = X.shape[0]
            resultados_clip_zero["tamano_entrenamiento"] = X_entrenamiento.shape[0]
            resultados_clip_zero["tamano_prueba"] = X_prueba.shape[0]
            resultados_clip_zero["nombres_columnas"] = nombres_columnas
            resultados_clip_zero["conteo_clases"] = conteo_por_clase
            resultados_por_modelo["clip_zero_shot"] = resultados_clip_zero

            resultados_clip_emb = entrenar_y_evaluar_clip_embeddings(
                modelo=modelo_clip,
                procesador=procesador_clip,
                X_entrenamiento=X_entrenamiento,
                y_entrenamiento=y_entrenamiento,
                X_prueba=X_prueba,
                y_prueba=y_prueba,
                tamano_imagen=tamano_imagen,
                convertir_a_grises=convertir_a_grises,
                dispositivo=dispositivo_clip,
                ruta_guardado_figura=f"{prefijo_figuras}_matriz_confusion_clip_embeddings.png"
            )
            resultados_clip_emb["tamano_total"] = X.shape[0]
            resultados_clip_emb["tamano_entrenamiento"] = X_entrenamiento.shape[0]
            resultados_clip_emb["tamano_prueba"] = X_prueba.shape[0]
            resultados_clip_emb["nombres_columnas"] = nombres_columnas
            resultados_clip_emb["conteo_clases"] = conteo_por_clase
            resultados_por_modelo["clip_embeddings"] = resultados_clip_emb
        else:
            print("Aviso: se omite CLIP porque torch/transformers no están instalados.")

        if not resultados_por_modelo:
            print("ERROR: no hay modelos disponibles para comparar.")
            return

        reporte_comparativo = generar_reporte_comparativo_modelos(
            resultados_por_modelo=resultados_por_modelo,
            conteo_clases=conteo_por_clase,
            ruta_csv_salida=ruta_comparacion_csv,
            ruta_txt_salida=ruta_archivo_conclusiones
        )
        print("\n=== INTERPRETACIÓN COMPARATIVA ===")
        print(reporte_comparativo["interpretacion"])
        return
    elif tipo_modelo == "resnet18":
        if not TORCH_DISPONIBLE:
            print("ERROR: tipo_modelo='resnet18' requiere instalar torch y torchvision.")
            print("Ejemplo: pip install torch torchvision")
            return

        loader_train, loader_val, _, nombres_clases_resnet = crear_dataloaders_resnet(
            X_entrenamiento=X_entrenamiento,
            y_entrenamiento=y_entrenamiento,
            X_validacion=X_prueba,
            y_validacion=y_prueba,
            tamano_imagen=tamano_imagen,
            convertir_a_grises=convertir_a_grises,
            tamano_batch=16
        )

        modelo = crear_modelo_resnet18(num_clases=len(nombres_clases_resnet), usar_preentrenado=True)
        modelo, historial_resnet = entrenar_modelo_resnet(
            modelo=modelo,
            loader_entrenamiento=loader_train,
            loader_validacion=loader_val,
            epocas=5,
            tasa_aprendizaje=1e-4
        )

        resultados = evaluar_modelo_resnet(
            modelo=modelo,
            loader_datos=loader_val,
            nombres_clases=nombres_clases_resnet,
            ruta_guardado_figura=ruta_matriz_confusion,
            mostrar_figura=True
        )
        resultados["historial_resnet"] = historial_resnet
    elif tipo_modelo == "clip_zero_shot":
        if not CLIP_DISPONIBLE:
            print("ERROR: tipo_modelo='clip_zero_shot' requiere torch y transformers.")
            print("Ejemplo: pip install torch transformers")
            return

        modelo_clip, procesador_clip, dispositivo_clip = cargar_modelo_clip_preentrenado()
        nombres_clases_clip = sorted([str(c) for c in np.unique(y_entrenamiento)])
        resultados = evaluar_clip_zero_shot(
            modelo=modelo_clip,
            procesador=procesador_clip,
            X_prueba=X_prueba,
            y_prueba=y_prueba,
            nombres_clases=nombres_clases_clip,
            tamano_imagen=tamano_imagen,
            convertir_a_grises=convertir_a_grises,
            dispositivo=dispositivo_clip,
            ruta_guardado_figura=ruta_matriz_confusion
        )
    elif tipo_modelo == "clip_embeddings":
        if not CLIP_DISPONIBLE:
            print("ERROR: tipo_modelo='clip_embeddings' requiere torch y transformers.")
            print("Ejemplo: pip install torch transformers")
            return

        modelo_clip, procesador_clip, dispositivo_clip = cargar_modelo_clip_preentrenado()
        resultados = entrenar_y_evaluar_clip_embeddings(
            modelo=modelo_clip,
            procesador=procesador_clip,
            X_entrenamiento=X_entrenamiento,
            y_entrenamiento=y_entrenamiento,
            X_prueba=X_prueba,
            y_prueba=y_prueba,
            tamano_imagen=tamano_imagen,
            convertir_a_grises=convertir_a_grises,
            dispositivo=dispositivo_clip,
            ruta_guardado_figura=ruta_matriz_confusion
        )
    else:
        modelo = entrenar_modelo_random_forest(
            X_entrenamiento=X_entrenamiento,
            y_entrenamiento=y_entrenamiento,
            n_arboles=n_arboles,
            profundidad_maxima=profundidad_maxima,
            semilla=semilla_modelo
        )

        resultados = mostrar_resultados(
            modelo=modelo,
            X_prueba=X_prueba,
            y_prueba=y_prueba,
            nombres_clases=None,
            ruta_guardado_figura=ruta_matriz_confusion
        )

    # Información adicional para reportes automáticos
    resultados["tamano_total"] = X.shape[0]
    resultados["tamano_entrenamiento"] = X_entrenamiento.shape[0]
    resultados["tamano_prueba"] = X_prueba.shape[0]
    resultados["nombres_columnas"] = nombres_columnas
    resultados["conteo_clases"] = conteo_por_clase

    resultados_por_modelo = {
        tipo_modelo: resultados
    }
    reporte_comparativo = generar_reporte_comparativo_modelos(
        resultados_por_modelo=resultados_por_modelo,
        conteo_clases=conteo_por_clase,
        ruta_csv_salida=ruta_comparacion_csv,
        ruta_txt_salida=ruta_archivo_conclusiones
    )
    print("\n=== INTERPRETACIÓN COMPARATIVA ===")
    print(reporte_comparativo["interpretacion"])

    if tipo_modelo == "random_forest":
        graficar_curva_aprendizaje_modelo(
            estimador_base=RandomForestClassifier(
                n_estimators=n_arboles,
                max_depth=profundidad_maxima,
                random_state=semilla_modelo
            ),
            X=X_entrenamiento,
            y=y_entrenamiento,
            ruta_guardado=ruta_curva_aprendizaje
        )

        graficar_importancia_caracteristicas(
            modelo=modelo,
            nombres_columnas=nombres_columnas,
            ruta_guardado=ruta_importancia,
            top_n=20
        )

    # Referencias posteriores a 2017 (actualiza las métricas con datos reales del estado del arte)
    referencias_literatura = [
        {"nombre": "Attia et al. (2019) - Red profunda para fibrilación auricular", "accuracy": 0.94},
        {"nombre": "Yildirim (2020) - Modelo híbrido 1D-CNN/GRU", "accuracy": 0.96},
        {"nombre": "Oh et al. (2021) - Ensemble de CNNs ligeras", "accuracy": 0.93},
        {"nombre": "Gao et al. (2022) - Transformer para ECG multiclase", "accuracy": 0.98},
    ]

    generar_discusion_resultados(
        resultados_modelo=resultados,
        referencias_literatura=referencias_literatura,
        ruta_salida=ruta_archivo_discusion
    )

    generar_conclusiones_y_trabajo_futuro(
        resultados_modelo=resultados,
        referencias_literatura=referencias_literatura,
        ruta_salida=ruta_archivo_conclusiones
    )

    if tipo_modelo == "random_forest":
        solicitar_imagen_usuario(
            modelo=modelo,
            escalador=escalador,
            tamano_imagen=tamano_imagen,
            convertir_a_grises=convertir_a_grises
        )

def validar_integracion_final(
    datasets_detectados: list[dict[str, str]],
    filas_resultados: list[dict],
    ruta_csv_global: str
) -> None:
    """
    Valida automáticamente que la integración final dataset x modelo sea consistente.
    """
    print("\n=== VALIDACIÓN FINAL ===")

    modelos_esperados = ["random_forest", "resnet18", "clip_embeddings", "clip_zero_shot"]

    nombres_datasets_detectados = [d.get("nombre", "") for d in datasets_detectados]
    datasets_detectados_canon = sorted({
        _nombre_dataset_canonico(n) for n in nombres_datasets_detectados
    })
    
    datasets_esperados = datasets_detectados_canon

    tabla = pd.DataFrame(filas_resultados)
    if tabla.empty:
        print("Datasets detectados: 0")
        print("Modelos detectados: 0")
        print("Combinaciones esperadas: 0")
        print("Combinaciones obtenidas: 0")
        print("Filas con error: 0")
        print("Estado final: INCOMPLETO")
        print("Advertencia: no se obtuvieron filas de resultados para validar.")
        return

    datasets_en_tabla_canon = sorted({
        _nombre_dataset_canonico(n) for n in tabla["dataset"].dropna().astype(str).tolist()
    })
    modelos_detectados = sorted(set(tabla["modelo"].dropna().astype(str).tolist()))

    combinaciones_esperadas = len(datasets_esperados) * len(modelos_esperados)
    combinaciones_obtenidas = int(len(tabla))

    columnas_metricas = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    invalidas_nan_none = pd.Series([False] * len(tabla))
    invalidas_menos_uno = pd.Series([False] * len(tabla))
    for col in columnas_metricas:
        if col in tabla.columns:
            serie = pd.to_numeric(tabla[col], errors="coerce")
            invalidas_nan_none = invalidas_nan_none | serie.isna()
            invalidas_menos_uno = invalidas_menos_uno | (serie == -1.0)

    filas_con_error = int((invalidas_nan_none | invalidas_menos_uno).sum())

    csv_ok = False
    csv_filas = -1
    try:
        if os.path.exists(ruta_csv_global):
            csv_tabla = pd.read_csv(ruta_csv_global)
            csv_filas = int(len(csv_tabla))
            csv_ok = True
    except Exception as exc:
        print(f"Advertencia: no se pudo leer el CSV global para validación: {exc}")

    print(f"Datasets detectados: {len(datasets_detectados_canon)}")
    print(f"Nombres datasets detectados: {datasets_detectados_canon}")
    print(f"Modelos detectados: {len(modelos_detectados)}")
    print(f"Nombres modelos detectados: {modelos_detectados}")
    print(f"Combinaciones esperadas: {combinaciones_esperadas}")
    print(f"Combinaciones obtenidas: {combinaciones_obtenidas}")
    print(f"Filas con error: {filas_con_error}")

    if csv_ok:
        print(f"Filas en CSV global ({ruta_csv_global}): {csv_filas}")
    else:
        print(f"Advertencia: CSV global no disponible en ruta: {ruta_csv_global}")

    faltantes_datasets = sorted(set(datasets_esperados) - set(datasets_en_tabla_canon))
    sobrantes_datasets = sorted(set(datasets_en_tabla_canon) - set(datasets_esperados))
    faltantes_modelos = sorted(set(modelos_esperados) - set(modelos_detectados))
    sobrantes_modelos = sorted(set(modelos_detectados) - set(modelos_esperados))

    if faltantes_datasets:
        print(f"Advertencia: faltan datasets esperados: {faltantes_datasets}")
    if sobrantes_datasets:
        print(f"Advertencia: datasets no esperados detectados: {sobrantes_datasets}")
    if faltantes_modelos:
        print(f"Advertencia: faltan modelos esperados: {faltantes_modelos}")
    if sobrantes_modelos:
        print(f"Advertencia: modelos no esperados detectados: {sobrantes_modelos}")

    if combinaciones_obtenidas != combinaciones_esperadas:
        print(
            "Advertencia: faltan combinaciones o hay filas extra. "
            f"Esperado={combinaciones_esperadas}, obtenido={combinaciones_obtenidas}"
        )

    if csv_ok and csv_filas != combinaciones_esperadas:
        print(
            "Advertencia: el CSV global no tiene 20 filas válidas esperadas "
            f"(actual={csv_filas}, esperado={combinaciones_esperadas})."
        )

    estado_ok = (
        not faltantes_datasets
        and not faltantes_modelos
        and not sobrantes_modelos
        and combinaciones_obtenidas == combinaciones_esperadas
        and filas_con_error == 0
        and csv_ok
        and csv_filas == combinaciones_esperadas
    )
    print(f"Estado final: {'OK' if estado_ok else 'INCOMPLETO'}")

    print("\n=== GENERANDO HEATMAPS DE VALIDACIÓN ===")
    for ds in datasets_en_tabla_canon:
        # Filtrar datos del dataset actual
        df_ds = tabla[tabla['dataset'].apply(_nombre_dataset_canonico) == ds].copy()
        if df_ds.empty:
            continue
        
        # Preparar datos: índice = modelo, columnas = métricas
        # Conservamos solo las métricas disponibles
        cols_presentes = [c for c in columnas_metricas if c in df_ds.columns]
        df_heat = df_ds.set_index('modelo')[cols_presentes]
        
        # Convertir a numérico por si acaso
        for c in df_heat.columns:
            df_heat[c] = pd.to_numeric(df_heat[c], errors='coerce')
            
        # Descartar modelos o métricas completamente nulas si existieran
        df_heat = df_heat.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        if df_heat.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df_heat,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Score", "shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(f"Rendimiento de Modelos - {ds}", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Modelo", fontsize=12)
        ax.set_xlabel("Métrica", fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
        
        fig.tight_layout()
        
        nombre_png = f"heatmap_validacion_{ds.replace(' ', '_').replace('-', '_').lower()}.png"
        fig.savefig(nombre_png, dpi=150, bbox_inches="tight")
        print(f"  Heatmap guardado: {nombre_png}")
        
        plt.show(block=False)
        plt.pause(0.5)
        plt.close(fig)

def solicitar_imagen_usuario(
    modelo,
    escalador: Optional[object],
    tamano_imagen: tuple[int, int],
    convertir_a_grises: bool
) -> None:
    """
    Pide al usuario una ruta de imagen para validar el modelo con datos externos.
    """
    while True:
        try:
            respuesta = input("\n¿Deseas clasificar una imagen externa para validar el modelo? (s/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("Entrada no disponible; se omite la validación manual.")
            return

        if respuesta not in {"s", "si", "sí"}:
            print("Validación manual finalizada.")
            return

        try:
            ruta_imagen = input("Ingresa la ruta completa de la imagen (PNG/JPG): ").strip().strip('"')
        except (EOFError, KeyboardInterrupt):
            print("No se proporcionó ruta de imagen.")
            return

        if not ruta_imagen:
            print("Ruta vacía, se cancela la validación.")
            return

        try:
            vector_imagen = procesar_imagen_individual(
                ruta_imagen=ruta_imagen,
                tamano_imagen=tamano_imagen,
                convertir_a_escala_grises=convertir_a_grises
            )
        except Exception as exc:
            print(f"No se pudo procesar la imagen: {exc}")
            continue

        vector_imagen = vector_imagen.reshape(1, -1)
        if escalador is not None:
            vector_imagen = escalador.transform(vector_imagen)

        prediccion = modelo.predict(vector_imagen)[0]
        mensaje = f"La predicción para la imagen '{ruta_imagen}' es: {prediccion}"
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(vector_imagen)[0]
            confianza = np.max(probabilidades)
            mensaje += f" (confianza {confianza:.2%})"
        print(mensaje)


if __name__ == '__main__':
    main()
