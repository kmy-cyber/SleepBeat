import pandas as pd
import numpy as np
from pyPPG import PPG, Fiducials, Biomarkers
import pyPPG.preproc as PP

def procesar_y_extraer_caracteristicas(signal_obj, etiquetas, fs, duracion_epoca_s=30):
    """
    Procesa la señal PPG, extrae características y las alinea con las etiquetas de sueño.

    Justificación del Proceso:
    1. Filtrado: Se aplica un filtro paso banda (0.5-12 Hz) para eliminar ruido
       y deriva de la línea base, un paso estándar y crítico para mejorar la
       calidad de la señal antes de la detección de picos.
    2. Detección de Fiduciales: Se utiliza el detector de picos de pyppg para
       identificar los puntos clave de cada latido. Una detección precisa es
       fundamental para el cálculo de la HRV.
    3. Cálculo de Biomarcadores: Se extrae un conjunto rico de 74 biomarcadores
       latido a latido, capturando información del dominio del tiempo, frecuencia
       y morfología, como se discutió en la investigación.
    4. Agregación por Épocas: Los biomarcadores se agregan (media y desviación
       estándar) en ventanas de 30 segundos para que coincidan con las etiquetas
       de sueño estándar de la AASM. Esto crea el conjunto de datos final para el
       modelo de Machine Learning.
    """
    print("\n--- Fase 2: Preprocesamiento y Extracción de Características ---")

    # 1. Preprocesamiento de la señal
    print("Filtrando la señal PPG...")
    preproc = PP.Preprocess(fL=0.5, fH=12, order=4, ftype='cheby2')
    s_filt, _, _, _ = preproc.get_signals(s=signal_obj)

    # 2. Detección de puntos fiduciales (picos, valles, etc.)
    print("Detectando puntos fiduciales...")
    s_filt_obj = PPG(s_filt.get_signal(), fs=fs)
    fp_extractor = Fiducials.FpCollection(s=s_filt_obj)
    try:
        fiducials = fp_extractor.get_fiducials(s=s_filt_obj)
        print(f"Se detectaron {len(fiducials.get_onsets())} latidos.")
    except Exception as e:
        print(f"Error en la detección de puntos fiduciales: {e}")
        print("La calidad de la señal puede ser demasiado baja. Abortando.")
        return None, None

    # 3. Cálculo de biomarcadores latido a latido
    print("Calculando biomarcadores...")
    bm_extractor = Biomarkers.BmCollection(s=s_filt_obj, fp=Fiducials(fp=fiducials))
    _, bm_vals, _ = bm_extractor.get_biomarkers()

    # 4. Agregación de características por épocas de 30 segundos
    print("Agregando características por épocas...")
    muestras_por_epoca = duracion_epoca_s * fs
    num_epocas = len(etiquetas)
    
    picos_sistolicos_indices = fiducials.get_s_peaks()
    tiempos_latidos = picos_sistolicos_indices / fs
    # Asegurarse de que las longitudes coincidan antes de asignar
    bm_vals['tiempo'] = tiempos_latidos[:len(bm_vals)]

    lista_caracteristicas_epoca = []
    for i in range(num_epocas):
        tiempo_inicio_epoca = i * duracion_epoca_s
        tiempo_fin_epoca = (i + 1) * duracion_epoca_s
        
        bm_epoca = bm_vals[
            (bm_vals['tiempo'] >= tiempo_inicio_epoca) & 
            (bm_vals['tiempo'] < tiempo_fin_epoca)
        ]
        
        if not bm_epoca.empty:
            stats_epoca = bm_epoca.drop(columns=['tiempo']).agg(['mean', 'std']).unstack()
            stats_epoca.index = [f'{col}_{stat}' for col, stat in stats_epoca.index]
            lista_caracteristicas_epoca.append(stats_epoca)
        else:
            column_names = [f'{col}_{stat}' for col in bm_vals.drop(columns=['tiempo']).columns for stat in ['mean', 'std']]
            lista_caracteristicas_epoca.append(pd.Series(index=column_names, dtype=float))

    X_features = pd.DataFrame(lista_caracteristicas_epoca)
    
    print(f"Rellenando {X_features.isnull().sum().sum()} valores NaN (usando ffill y bfill).")
    X_features.fillna(method='ffill', inplace=True)
    X_features.fillna(method='bfill', inplace=True)
    
    if X_features.isnull().sum().sum() > 0:
        print("Aún quedan valores NaN. Rellenando con 0.")
        X_features.fillna(0, inplace=True)

    print("Extracción de características completada.")
    print(f"Dimensiones del conjunto de características: {X_features.shape}")

    return X_features, etiquetas