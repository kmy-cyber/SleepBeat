import numpy as np
import pandas as pd
import os

def generar_datos_ejemplo(duracion_minutos=10, fs=100):
    """
    Genera un conjunto de datos sintético de una señal PPG y etiquetas de sueño.

    Crea dos archivos CSV en la carpeta 'data/':
    - ppg_signal.csv: Una señal PPG simulada.
    - sleep_labels.csv: Etiquetas de etapas de sueño para épocas de 30 segundos.

    Args:
        duracion_minutos (int): La duración total de la señal en minutos.
        fs (int): La frecuencia de muestreo de la señal PPG en Hz.
    """
    print("Generando datos de ejemplo sintéticos...")

    # --- 1. Configuración de Parámetros ---
    num_muestras = duracion_minutos * 60 * fs
    tiempo = np.arange(num_muestras) / fs
    duracion_epoca_s = 30
    num_epocas = int(num_muestras / (duracion_epoca_s * fs))

    # --- 2. Crear Directorio de Datos ---
    directorio_datos = 'data'
    if not os.path.exists(directorio_datos):
        os.makedirs(directorio_datos)

    # --- 3. Generar Señal PPG Sintética ---
    # Simula una frecuencia cardíaca que varía ligeramente (ej. entre 55 y 70 bpm)
    hr_bpm = 60 + 5 * np.sin(2 * np.pi * 0.005 * tiempo)
    hr_hz = hr_bpm / 60
    
    # Crear la señal base como una sinusoide que representa los latidos
    ppg_signal = np.sin(2 * np.pi * np.cumsum(hr_hz) / fs)
    
    # Añadir algo de ruido para que sea más realista
    ruido = 0.2 * np.random.normal(size=num_muestras)
    ppg_signal += ruido
    
    # Añadir una deriva de línea base lenta (simulando la respiración)
    deriva = 0.5 * np.sin(2 * np.pi * 0.2 * tiempo)
    ppg_signal += deriva

    # Guardar la señal PPG
    ppg_df = pd.DataFrame({'ppg': ppg_signal})
    ruta_ppg = os.path.join(directorio_datos, 'ppg_signal.csv')
    ppg_df.to_csv(ruta_ppg, index=False)
    print(f"Señal PPG de ejemplo guardada en '{ruta_ppg}' ({num_muestras} muestras).")

    # --- 4. Generar Etiquetas de Sueño Sintéticas ---
    # Se genera una etiqueta por cada época de 30 segundos.
    # Mapeo de etiquetas: 0: Vigilia, 1: Ligero (N1/N2), 2: Profundo (N3), 3: REM
    etiquetas_lista = []
    # Simula un ciclo de sueño simple para las épocas generadas
    # Ejemplo para 20 épocas (10 minutos)
    # [0,0] Vigilia -> [1,1,1,1,1,1] Ligero -> [2,2,2,2,2,2] Profundo -> [3,3,3,3,1,1] REM y vuelta a ligero
    for i in range(num_epocas):
        if i < 2:
            etiquetas_lista.append(0)  # Vigilia
        elif i < 8:
            etiquetas_lista.append(1)  # Sueño Ligero
        elif i < 14:
            etiquetas_lista.append(2)  # Sueño Profundo
        else:
            etiquetas_lista.append(3)  # Sueño REM
    
    # --- LA CORRECCIÓN ESTÁ AQUÍ ---
    # El error original era llamar a np.array() sin argumentos.
    # La forma correcta es pasar la lista de etiquetas que hemos creado.
    etiquetas = np.array(etiquetas_lista)

    # Guardar las etiquetas
    etiquetas_df = pd.DataFrame({'sleep_stage': etiquetas})
    ruta_etiquetas = os.path.join(directorio_datos, 'sleep_labels.csv')
    etiquetas_df.to_csv(ruta_etiquetas, index=False)
    print(f"Etiquetas de sueño de ejemplo guardadas en '{ruta_etiquetas}' ({len(etiquetas)} épocas).")
    print("¡Datos de ejemplo generados con éxito!")


if __name__ == '__main__':
    # Este bloque permite ejecutar el script directamente desde la terminal.
    generar_datos_ejemplo()