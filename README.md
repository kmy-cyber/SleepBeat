# Prototipo de Tesis: Análisis de Señales PPG para Estimación del Sueño

Este proyecto es un prototipo de software desarrollado como parte de la tesis "Análisis de señales PPG para extraer y estimar componentes del sueño". El sistema toma una señal de fotopletismografía (PPG), la procesa, extrae características fisiológicas y utiliza un modelo de Machine Learning para clasificar las etapas del sueño.

## Estructura de Carpetas

- **/data/**: Contiene los datos de entrada. Ejecuta `0_generar_datos_ejemplo.py` para crear archivos CSV de ejemplo.
- **/src/**: Contiene el código fuente modularizado por fases.
  - `0_generar_datos_ejemplo.py`: Script para crear datos de PPG y etiquetas de sueño sintéticos.
  - `1_carga_datos.py`: Funciones para cargar los datos.
  - `2_preprocesamiento.py`: Funciones para filtrar y limpiar la señal PPG.
  - `3_extraccion_caracteristicas.py`: Funciones para extraer biomarcadores usando `pyPPG` y agregarlos por épocas.
  - `4_entrenamiento_modelo.py`: Funciones para entrenar y evaluar el clasificador de etapas de sueño.
  - `utils.py`: Funciones de utilidad, como la visualización de resultados.
- **/resultados/**: Carpeta donde se guardarán los gráficos (matriz de confusión, importancia de características) y el modelo entrenado.
- `main.py`: El script principal que orquesta todo el flujo de trabajo.
- `requirements.txt`: Las dependencias de Python necesarias para el proyecto.
- `README.md`: Este archivo.

## Instrucciones de Configuración y Ejecución

1.  **Crear un entorno virtual (recomendado):**bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

2.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generar datos de ejemplo:**
    Antes de ejecutar el pipeline principal, crea los archivos de datos sintéticos.
    ```bash
    python src/0_generar_datos_ejemplo.py
    ```
    Esto creará `ppg_signal.csv` y `sleep_labels.csv` en la carpeta `data/`.

4.  **Ejecutar el pipeline completo:**
    ```bash
    python main.py
    ```

## Salida Esperada

Al ejecutar `main.py`, el programa realizará los siguientes pasos:
1.  Cargará la señal PPG y las etiquetas de sueño.
2.  Preprocesará la señal PPG.
3.  Extraerá biomarcadores y los agregará en épocas de 30 segundos.
4.  Entrenará un modelo `RandomForestClassifier`.
5.  Imprimirá en la consola el reporte de clasificación y el coeficiente Kappa de Cohen.
6.  Guardará en la carpeta `/resultados/`:
    - Una imagen de la matriz de confusión (`matriz_confusion.png`).
    - Una imagen de la importancia de las características (`importancia_caracteristicas.png`).
    - El modelo entrenado (`modelo_sueño.joblib`).

