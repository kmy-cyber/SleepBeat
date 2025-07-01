import pandas as pd
from pyPPG.datahandling import load_data as pyppg_load_data

def cargar_signal_ppg(ruta_archivo, fs):
    """
    Carga una señal PPG desde un archivo CSV y le asigna la frecuencia de muestreo.

    Justificación:
    El error original ocurría porque 'fs' se pasaba directamente a pyppg_load_data.
    La corrección consiste en cargar primero el objeto de la señal y luego
    asignar la frecuencia de muestreo a su atributo .fs, que es el
    procedimiento correcto según la documentación de la biblioteca.
    """
    # 1. Cargar los datos usando la función de pyppg.
    print(f"Cargando señal desde: {ruta_archivo}")
    signal_obj = pyppg_load_data(data_path=ruta_archivo)

    # 2. Asignar la frecuencia de muestreo al atributo 'fs' del objeto.
    signal_obj.fs = fs

    return signal_obj

def cargar_etiquetas_sueno(ruta_archivo):
    """
    Carga las etiquetas de sueño desde un archivo CSV.
    """
    print(f"Cargando etiquetas desde: {ruta_archivo}")
    etiquetas_df = pd.read_csv(ruta_archivo)
    # .values para obtener un array de numpy
    return etiquetas_df['sleep_stage'].values