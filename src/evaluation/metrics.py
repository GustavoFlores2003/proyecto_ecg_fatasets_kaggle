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





def evaluar_modelos_en_dataset(
    datos_dataset: dict,
    tipo_modelo: str,
    tamano_imagen: tuple[int, int],
    convertir_a_grises: bool,
    ruta_archivo_discusion: str,
    ruta_archivo_conclusiones: str
) -> dict:
    """
    Ejecuta los modelos existentes sobre un dataset cargado y devuelve métricas resumidas.
    """
    from src.evaluation.reporting import generar_conclusiones_y_trabajo_futuro, generar_discusion_resultados, generar_reporte_comparativo_modelos
    from src.utils.helpers import normalizar_nombre_dataset, obtener_ruta_figura, obtener_ruta_reporte
    from src.visualization.plots import graficar_curva_aprendizaje_modelo, graficar_distribucion_clases, graficar_importancia_caracteristicas, graficar_importancia_pixeles, mostrar_resultados
    from src.models.clip_model import cargar_modelo_clip_preentrenado, entrenar_y_evaluar_clip_embeddings, evaluar_clip_zero_shot
    from src.models.random_forest import entrenar_modelo_random_forest
    from src.models.resnet import crear_dataloaders_resnet, crear_modelo_resnet18, entrenar_modelo_resnet, evaluar_modelo_resnet, preparar_tensores_resnet
    nombre_dataset = normalizar_nombre_dataset(datos_dataset["nombre_dataset"])
    tipo_dataset = datos_dataset.get("tipo_dataset", "desconocido")
    X = datos_dataset["X"]
    y = datos_dataset["y"]
    X_por_modelo = datos_dataset.get("X_por_modelo", {})
    X_entrenamiento = datos_dataset["X_entrenamiento"]
    X_prueba = datos_dataset["X_prueba"]
    y_entrenamiento = datos_dataset["y_entrenamiento"]
    y_prueba = datos_dataset["y_prueba"]
    nombres_columnas = datos_dataset["nombres_columnas"]
    conteo_por_clase = datos_dataset["conteo_por_clase"]
    tamano_imagen_efectivo = datos_dataset.get("tamano_imagen_dataset", tamano_imagen)

    print(f"\n{'=' * 70}")
    print(f"Procesando dataset: {datos_dataset['nombre_dataset']}")
    print(f"DATASET: {datos_dataset['nombre_dataset']}")
    print(f"Ruta usada: {datos_dataset['ruta_dataset']}")
    print(f"Tipo dataset: {tipo_dataset}")
    print(f"Clases ({len(conteo_por_clase)}): {sorted(list(conteo_por_clase.keys()))}")
    print(f"Total muestras: {X.shape[0]} | Entrenamiento: {X_entrenamiento.shape[0]} | Prueba: {X_prueba.shape[0]}")
    print(f"Características por muestra: {X.shape[1]}")

    graficar_distribucion_clases(
        conteo_por_clase,
        ruta_guardado=str(obtener_ruta_figura(f"{nombre_dataset}_distribucion_clases.png"))
    )

    n_arboles = 100
    profundidad_maxima = None
    semilla_modelo = 42
    modelos_objetivo = ["random_forest", "resnet18", "clip_embeddings", "clip_zero_shot"]
    resultados_por_modelo: dict[str, dict] = {}
    errores_por_modelo: dict[str, str] = {}

    if tipo_modelo == "comparar_todos":
        print("\n=== Comparación justa: mismo train/test para todos los modelos ===")
        for nombre_modelo in modelos_objetivo:
            print(f"Modelo: {nombre_modelo}")
            X_train_modelo, X_test_modelo = X_por_modelo.get(
                nombre_modelo,
                (X_entrenamiento, X_prueba)
            )

            try:
                if nombre_modelo == "random_forest":
                    modelo_rf = entrenar_modelo_random_forest(
                        X_entrenamiento=X_train_modelo,
                        y_entrenamiento=y_entrenamiento,
                        n_arboles=n_arboles,
                        profundidad_maxima=profundidad_maxima,
                        semilla=semilla_modelo
                    )
                    resultados_rf = mostrar_resultados(
                        modelo=modelo_rf,
                        X_prueba=X_test_modelo,
                        y_prueba=y_prueba,
                        nombres_clases=None,
                        ruta_guardado_figura=str(obtener_ruta_figura(f"{nombre_dataset}_matriz_confusion_random_forest.png"))
                    )
                    resultados_por_modelo["random_forest"] = resultados_rf
                    
                    # Generar heatmap de píxeles (importancia)
                    graficar_importancia_pixeles(
                        modelo_rf=modelo_rf,
                        tamano_imagen=tamano_imagen_efectivo,
                        ruta_guardado_figura=str(obtener_ruta_figura(f"{nombre_dataset}_heatmap_pixeles.png"))
                    )

                elif nombre_modelo == "resnet18":
                    if not TORCH_DISPONIBLE:
                        print("Aviso: se omite resnet18 porque torch/torchvision no están instalados.")
                        continue

                    # Mapeo de clases usando train+test para evitar errores por clases ausentes en train.
                    clases_union = np.unique(np.concatenate([y_entrenamiento, y_prueba]))
                    clase_a_indice = {str(clase): idx for idx, clase in enumerate(sorted([str(c) for c in clases_union]))}
                    nombres_clases_resnet = [clase for clase, _ in sorted(clase_a_indice.items(), key=lambda item: item[1])]

                    x_train_tensor, y_train_tensor = preparar_tensores_resnet(
                        X_train_modelo,
                        y_entrenamiento,
                        clase_a_indice,
                        tamano_imagen_efectivo,
                        convertir_a_grises
                    )
                    x_val_tensor, y_val_tensor = preparar_tensores_resnet(
                        X_test_modelo,
                        y_prueba,
                        clase_a_indice,
                        tamano_imagen_efectivo,
                        convertir_a_grises
                    )

                    loader_train = DataLoader(
                        TensorDataset(x_train_tensor, y_train_tensor),
                        batch_size=16,
                        shuffle=True
                    )
                    loader_val = DataLoader(
                        TensorDataset(x_val_tensor, y_val_tensor),
                        batch_size=16,
                        shuffle=False
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
                        ruta_guardado_figura=str(obtener_ruta_figura(f"{nombre_dataset}_matriz_confusion_resnet18.png")),
                        mostrar_figura=True
                    )
                    resultados_resnet["historial_resnet"] = historial_resnet
                    resultados_por_modelo["resnet18"] = resultados_resnet

                elif nombre_modelo == "clip_embeddings":
                    if not CLIP_DISPONIBLE:
                        print("Aviso: se omite clip_embeddings porque torch/transformers no están instalados.")
                        continue

                    modelo_clip, procesador_clip, dispositivo_clip = cargar_modelo_clip_preentrenado()
                    resultados_clip_emb = entrenar_y_evaluar_clip_embeddings(
                        modelo=modelo_clip,
                        procesador=procesador_clip,
                        X_entrenamiento=X_train_modelo,
                        y_entrenamiento=y_entrenamiento,
                        X_prueba=X_test_modelo,
                        y_prueba=y_prueba,
                        tamano_imagen=tamano_imagen_efectivo,
                        convertir_a_grises=convertir_a_grises,
                        dispositivo=dispositivo_clip,
                        ruta_guardado_figura=str(obtener_ruta_figura(f"{nombre_dataset}_matriz_confusion_clip_embeddings.png"))
                    )
                    resultados_por_modelo["clip_embeddings"] = resultados_clip_emb

                elif nombre_modelo == "clip_zero_shot":
                    if not CLIP_DISPONIBLE:
                        print("Aviso: se omite clip_zero_shot porque torch/transformers no están instalados.")
                        continue

                    modelo_clip, procesador_clip, dispositivo_clip = cargar_modelo_clip_preentrenado()
                    nombres_clases_clip = sorted([str(c) for c in np.unique(np.concatenate([y_entrenamiento, y_prueba]))])
                    resultados_clip_zero = evaluar_clip_zero_shot(
                        modelo=modelo_clip,
                        procesador=procesador_clip,
                        X_prueba=X_test_modelo,
                        y_prueba=y_prueba,
                        nombres_clases=nombres_clases_clip,
                        tamano_imagen=tamano_imagen_efectivo,
                        convertir_a_grises=convertir_a_grises,
                        dispositivo=dispositivo_clip,
                        ruta_guardado_figura=str(obtener_ruta_figura(f"{nombre_dataset}_matriz_confusion_clip_zero_shot.png"))
                    )
                    resultados_por_modelo["clip_zero_shot"] = resultados_clip_zero

                else:
                    print(f"Aviso: modelo no reconocido '{nombre_modelo}', se omite.")
                    continue

                resultados_por_modelo[nombre_modelo]["tamano_total"] = X.shape[0]
                resultados_por_modelo[nombre_modelo]["tamano_entrenamiento"] = X_entrenamiento.shape[0]
                resultados_por_modelo[nombre_modelo]["tamano_prueba"] = X_prueba.shape[0]
                resultados_por_modelo[nombre_modelo]["nombres_columnas"] = nombres_columnas
                resultados_por_modelo[nombre_modelo]["conteo_clases"] = conteo_por_clase

            except Exception as exc:
                import traceback
                error_real = traceback.format_exc()
                print(
                    f"Aviso: fallo en modelo '{nombre_modelo}' para dataset "
                    f"'{datos_dataset['nombre_dataset']}': {exc}\n"
                    f"Traceback completo:\n{error_real}"
                )
                errores_por_modelo[nombre_modelo] = str(exc)
    elif tipo_modelo == "resnet18":
        if not TORCH_DISPONIBLE:
            raise RuntimeError("tipo_modelo='resnet18' requiere instalar torch y torchvision.")

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
            ruta_guardado_figura=f"{nombre_dataset}_matriz_confusion.png",
            mostrar_figura=True
        )
        resultados["historial_resnet"] = historial_resnet
        resultados_por_modelo[tipo_modelo] = resultados
    elif tipo_modelo == "clip_zero_shot":
        if not CLIP_DISPONIBLE:
            raise RuntimeError("tipo_modelo='clip_zero_shot' requiere torch y transformers.")

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
            ruta_guardado_figura=f"{nombre_dataset}_matriz_confusion.png"
        )
        resultados_por_modelo[tipo_modelo] = resultados
    elif tipo_modelo == "clip_embeddings":
        if not CLIP_DISPONIBLE:
            raise RuntimeError("tipo_modelo='clip_embeddings' requiere torch y transformers.")

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
            ruta_guardado_figura=f"{nombre_dataset}_matriz_confusion.png"
        )
        resultados_por_modelo[tipo_modelo] = resultados
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
            ruta_guardado_figura=f"{nombre_dataset}_matriz_confusion.png"
        )
        resultados_por_modelo["random_forest"] = resultados

        graficar_curva_aprendizaje_modelo(
            estimador_base=RandomForestClassifier(
                n_estimators=n_arboles,
                max_depth=profundidad_maxima,
                random_state=semilla_modelo
            ),
            X=X_entrenamiento,
            y=y_entrenamiento,
            ruta_guardado=str(obtener_ruta_figura(f"{nombre_dataset}_curva_aprendizaje.png"))
        )
        graficar_importancia_caracteristicas(
            modelo=modelo,
            nombres_columnas=nombres_columnas,
            ruta_guardado=str(obtener_ruta_figura(f"{nombre_dataset}_importancia_caracteristicas.png")),
            top_n=20
        )

    if not resultados_por_modelo:
        raise RuntimeError(f"No hubo resultados para el dataset: {datos_dataset['nombre_dataset']}")

    for resultado in resultados_por_modelo.values():
        resultado["tamano_total"] = X.shape[0]
        resultado["tamano_entrenamiento"] = X_entrenamiento.shape[0]
        resultado["tamano_prueba"] = X_prueba.shape[0]
        resultado["nombres_columnas"] = nombres_columnas
        resultado["conteo_clases"] = conteo_por_clase

    reporte_comparativo = generar_reporte_comparativo_modelos(
        resultados_por_modelo=resultados_por_modelo,
        conteo_clases=conteo_por_clase,
        ruta_csv_salida=str(obtener_ruta_reporte(f"{nombre_dataset}_comparacion_modelos.csv")),
        ruta_txt_salida=str(obtener_ruta_reporte(f"{nombre_dataset}_interpretacion_comparacion_modelos.txt"))
    )
    print("\n=== INTERPRETACIÓN COMPARATIVA DEL DATASET ===")
    print(reporte_comparativo["interpretacion"])

    # Para mantener reportes textuales actuales, se genera sobre el primer modelo disponible.
    primer_resultado = next(iter(resultados_por_modelo.values()))
    generar_discusion_resultados(
        resultados_modelo=primer_resultado,
        referencias_literatura=[
            {"nombre": "Attia et al. (2019) - Red profunda para fibrilación auricular", "accuracy": 0.94},
            {"nombre": "Yildirim (2020) - Modelo híbrido 1D-CNN/GRU", "accuracy": 0.96},
            {"nombre": "Oh et al. (2021) - Ensemble de CNNs ligeras", "accuracy": 0.93},
            {"nombre": "Gao et al. (2022) - Transformer para ECG multiclase", "accuracy": 0.98},
        ],
        ruta_salida=ruta_archivo_discusion
    )
    generar_conclusiones_y_trabajo_futuro(
        resultados_modelo=primer_resultado,
        referencias_literatura=[
            {"nombre": "Attia et al. (2019) - Red profunda para fibrilación auricular", "accuracy": 0.94},
            {"nombre": "Yildirim (2020) - Modelo híbrido 1D-CNN/GRU", "accuracy": 0.96},
            {"nombre": "Oh et al. (2021) - Ensemble de CNNs ligeras", "accuracy": 0.93},
            {"nombre": "Gao et al. (2022) - Transformer para ECG multiclase", "accuracy": 0.98},
        ],
        ruta_salida=ruta_archivo_conclusiones
    )

    filas_resumen = []
    for nombre_modelo in modelos_objetivo:
        resultado = resultados_por_modelo.get(nombre_modelo)
        if resultado is None:
            filas_resumen.append({
                "dataset": datos_dataset["nombre_dataset"],
                "tipo_dataset": tipo_dataset,
                "modelo": nombre_modelo,
                "accuracy": -1.0,
                "balanced_accuracy": -1.0,
                "precision_macro": -1.0,
                "recall_macro": -1.0,
                "f1_macro": -1.0,
                "precision_weighted": -1.0,
                "recall_weighted": -1.0,
                "f1_weighted": -1.0,
                "error": errores_por_modelo.get(nombre_modelo, "modelo_no_ejecutado"),
            })
            print(
                f"Modelo sin resultado en dataset {datos_dataset['nombre_dataset']}: "
                f"{nombre_modelo} | causa={errores_por_modelo.get(nombre_modelo, 'modelo_no_ejecutado')}"
            )
            continue

        resumen = resultado.get("resumen_metricas", {})
        filas_resumen.append({
            "dataset": datos_dataset["nombre_dataset"],
            "tipo_dataset": tipo_dataset,
            "modelo": nombre_modelo,
            "accuracy": float(resumen.get("accuracy", resultado.get("exactitud", 0.0))),
            "balanced_accuracy": float(resumen.get("balanced_accuracy", -1.0)),
            "precision_macro": float(resumen.get("precision_macro", -1.0)),
            "recall_macro": float(resumen.get("recall_macro", -1.0)),
            "f1_macro": float(resumen.get("f1_macro", -1.0)),
            "precision_weighted": float(resumen.get("precision_weighted", -1.0)),
            "recall_weighted": float(resumen.get("recall_weighted", -1.0)),
            "f1_weighted": float(resumen.get("f1_weighted", -1.0)),
            "error": "",
        })

    return {
        "resultados_por_modelo": resultados_por_modelo,
        "filas_resumen": filas_resumen
    }

def evaluar_resultados_clasificacion(
    y_prueba: np.ndarray,
    y_pred: np.ndarray,
    nombres_clases: Optional[list[str]] = None,
    ruta_guardado_figura: str = "matriz_confusion.png",
    nombre_modelo: str = "modelo"
):
    """
    Calcula métricas y guarda la matriz de confusión para cualquier clasificador.
    """
    exactitud = accuracy_score(y_prueba, y_pred)
    print("\n=== MÉTRICAS DE DESEMPEÑO ===")
    print(f"Exactitud (accuracy): {exactitud:.4f}")

    print("\nReporte de clasificación:")
    reporte_texto = classification_report(y_prueba, y_pred)
    reporte_dict = classification_report(y_prueba, y_pred, output_dict=True)
    print(reporte_texto)

    matriz = confusion_matrix(y_prueba, y_pred)

    if nombres_clases is None:
        nombres_clases = [str(c) for c in np.unique(y_prueba)]

    print("\nMatriz de confusión:")
    print(matriz)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(matriz, interpolation="nearest")
    ax.set_title("Matriz de Confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_xticks(np.arange(len(nombres_clases)))
    ax.set_yticks(np.arange(len(nombres_clases)))
    ax.set_xticklabels(nombres_clases, rotation=45, ha="right")
    ax.set_yticklabels(nombres_clases)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, matriz[i, j], ha="center", va="center")

    fig.tight_layout()
    plt.savefig(ruta_guardado_figura)
    print(f"\nMatriz de confusión guardada como: {ruta_guardado_figura}")
    plt.show(block=False)
    plt.pause(0.3)
    plt.close(fig)

    precision_macro = float(reporte_dict.get("macro avg", {}).get("precision", 0.0))
    recall_macro = float(reporte_dict.get("macro avg", {}).get("recall", 0.0))
    f1_macro = float(reporte_dict.get("macro avg", {}).get("f1-score", 0.0))
    precision_weighted = float(reporte_dict.get("weighted avg", {}).get("precision", 0.0))
    recall_weighted = float(reporte_dict.get("weighted avg", {}).get("recall", 0.0))
    f1_weighted = float(reporte_dict.get("weighted avg", {}).get("f1-score", 0.0))

    # Accuracy balanceada calculada como promedio de recall por clase.
    recall_por_clase = []
    for nombre in nombres_clases:
        metricas = reporte_dict.get(nombre)
        if isinstance(metricas, dict) and "recall" in metricas:
            recall_por_clase.append(float(metricas["recall"]))
    accuracy_balanceada = float(np.mean(recall_por_clase)) if recall_por_clase else 0.0

    resumen_metricas = {
        "accuracy": float(exactitud),
        "balanced_accuracy": accuracy_balanceada,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

    resultados = {
        "nombre_modelo": nombre_modelo,
        "exactitud": exactitud,
        "reporte_texto": reporte_texto,
        "reporte_dict": reporte_dict,
        "matriz_confusion": matriz,
        "nombres_clases": nombres_clases,
        "y_prueba": y_prueba,
        "y_pred": y_pred,
        "ruta_figura": ruta_guardado_figura,
        "resumen_metricas": resumen_metricas
    }
    return resultados

def comparar_resultados(
    filas_resultados: list[dict],
    ruta_csv_salida: str = "comparacion_modelos.csv"
) -> pd.DataFrame:
    """
    Compara resultados de modelos entre datasets en una tabla consolidada.
    """
    tabla = pd.DataFrame(filas_resultados)
    if tabla.empty:
        print("No hay resultados para comparar entre datasets.")
        return tabla

    columnas_esperadas = [
        "dataset",
        "tipo_dataset",
        "modelo",
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    for columna in columnas_esperadas:
        if columna not in tabla.columns:
            tabla[columna] = -1.0 if columna not in {"dataset", "tipo_dataset", "modelo"} else ""

    tabla = tabla[columnas_esperadas]
    tabla = tabla.sort_values(by=["dataset", "modelo"], ascending=[True, True]).reset_index(drop=True)
    tabla.to_csv(ruta_csv_salida, index=False, encoding="utf-8")
    print(f"Comparación global entre datasets guardada en: {ruta_csv_salida}")
    print("\n=== COMPARACIÓN GLOBAL DATASET / MODELO ===")
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return tabla

def construir_tabla_comparativa_modelos(
    resultados_por_modelo: dict[str, dict],
    ruta_csv_salida: str = "comparacion_modelos.csv"
) -> pd.DataFrame:
    """
    Crea tabla comparativa con métricas unificadas para cualquier conjunto de modelos.
    """
    filas = []
    for nombre_modelo, resultados in resultados_por_modelo.items():
        resumen = resultados.get("resumen_metricas", {})
        filas.append({
            "modelo": nombre_modelo,
            "accuracy": float(resumen.get("accuracy", resultados.get("exactitud", 0.0))),
            "balanced_accuracy": float(resumen.get("balanced_accuracy", 0.0)),
            "precision_macro": float(resumen.get("precision_macro", 0.0)),
            "recall_macro": float(resumen.get("recall_macro", 0.0)),
            "f1_macro": float(resumen.get("f1_macro", 0.0)),
            "precision_weighted": float(resumen.get("precision_weighted", 0.0)),
            "recall_weighted": float(resumen.get("recall_weighted", 0.0)),
            "f1_weighted": float(resumen.get("f1_weighted", 0.0)),
        })

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla

    tabla = tabla.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    tabla.to_csv(ruta_csv_salida, index=False, encoding="utf-8")
    print(f"Tabla comparativa guardada en: {ruta_csv_salida}")
    print("\n=== TABLA RESUMEN DE MÉTRICAS ===")
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return tabla

def detectar_alertas_evaluacion(
    conteo_clases: dict[str, int],
    resultados_por_modelo: dict[str, dict]
) -> list[str]:
    """
    Detecta riesgos de interpretación: desbalance y métricas potencialmente engañosas.
    """
    alertas: list[str] = []

    if conteo_clases:
        max_clase = max(conteo_clases.values())
        min_clase = min(conteo_clases.values())
        if min_clase > 0:
            ratio = max_clase / min_clase
            if ratio >= 1.5:
                alertas.append(
                    f"Desbalance de clases detectado (ratio mayor/menor = {ratio:.2f}). "
                    "Priorizar F1 macro y matriz de confusión sobre accuracy global."
                )

    for nombre_modelo, resultados in resultados_por_modelo.items():
        resumen = resultados.get("resumen_metricas", {})
        acc = float(resumen.get("accuracy", resultados.get("exactitud", 0.0)))
        f1_macro = float(resumen.get("f1_macro", 0.0))
        balanced_acc = float(resumen.get("balanced_accuracy", 0.0))

        if acc - f1_macro > 0.10:
            alertas.append(
                f"{nombre_modelo}: accuracy notablemente mayor que F1 macro ({acc:.3f} vs {f1_macro:.3f}); "
                "posible sesgo hacia clases mayoritarias."
            )

        if acc - balanced_acc > 0.10:
            alertas.append(
                f"{nombre_modelo}: gap entre accuracy y balanced accuracy ({acc:.3f} vs {balanced_acc:.3f}); "
                "revisar desempeño en clases minoritarias."
            )

        if "clip_zero_shot" in nombre_modelo.lower():
            alertas.append(
                f"{nombre_modelo}: CLIP zero-shot depende de la redacción de prompts; "
                "variaciones de texto pueden cambiar el resultado sin cambiar las imágenes."
            )

    return alertas
