import os
import pandas as pd
# Asegúrate de que la estructura de carpetas permita esta importación
# Si ejecutas desde la raíz, Python debería encontrar la carpeta 'src'
from src.load_data import cargar_signal_ppg, cargar_etiquetas_sueno
from src.feature_extraction import procesar_y_extraer_caracteristicas
from src.model_training import entrenar_y_evaluar_modelo

# --- Configuración Global ---
FS = 100
DURACION_EPOCA_S = 30
RUTA_DATOS_PPG = 'data/ppg_signal.csv'
RUTA_ETIQUETAS_SUENO = 'data/sleep_labels.csv'
RUTA_SALIDA_CARACTERISTICAS = 'data/features.csv'
RUTA_SALIDA_ETIQUETAS_PROC = 'data/labels_processed.csv'

def main():
    """
    Función principal que orquesta el pipeline de análisis de sueño.
    """
    print("Iniciando el pipeline de análisis de sueño basado en PPG...")

    # --- Fase 1: Carga de Datos ---
    print("\n--- Fase 1: Cargando Datos ---")
    if not os.path.exists(RUTA_DATOS_PPG) or not os.path.exists(RUTA_ETIQUETAS_SUENO):
        print("Error: No se encontraron los archivos de datos.")
        print("Por favor, ejecute primero 'src/0_generar_datos_ejemplo.py' para crear los datos sintéticos.")
        return

    signal_obj = cargar_signal_ppg(RUTA_DATOS_PPG, fs=FS)
    etiquetas = cargar_etiquetas_sueno(RUTA_ETIQUETAS_SUENO)
    print(f"Señal PPG cargada: {len(signal_obj.get_signal())} muestras.")
    print(f"Etiquetas de sueño cargadas: {len(etiquetas)} épocas.")

    # --- Fase 2: Preprocesamiento y Extracción de Características ---
    X_features, y_labels = procesar_y_extraer_caracteristicas(
        signal_obj, etiquetas, fs=FS, duracion_epoca_s=DURACION_EPOCA_S
    )

    if X_features is None:
        print("El pipeline se detuvo debido a un error en la extracción de características.")
        return

    # Guardar las características y etiquetas procesadas para uso futuro
    X_features.to_csv(RUTA_SALIDA_CARACTERISTICAS, index=False)
    pd.DataFrame(y_labels, columns=['sleep_stage']).to_csv(RUTA_SALIDA_ETIQUETAS_PROC, index=False)
    print(f"\nCaracterísticas procesadas guardadas en '{RUTA_SALIDA_CARACTERISTICAS}'")
    print(f"Etiquetas correspondientes guardadas en '{RUTA_SALIDA_ETIQUETAS_PROC}'")

    # --- Fase 3: Entrenamiento y Evaluación del Modelo ---
    entrenar_y_evaluar_modelo(
        ruta_caracteristicas=RUTA_SALIDA_CARACTERISTICAS,
        ruta_etiquetas=RUTA_SALIDA_ETIQUETAS_PROC
    )

if __name__ == '__main__':
    main()
