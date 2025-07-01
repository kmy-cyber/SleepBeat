import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def entrenar_y_evaluar_modelo(ruta_caracteristicas, ruta_etiquetas):
    """
    Carga las características, entrena un modelo de Random Forest y evalúa su rendimiento.

    Justificación del Modelo:
    Se elige un clasificador de Bosque Aleatorio (Random Forest) por varias razones:
    1. Rendimiento: Es un modelo robusto que funciona bien con datos tabulares y
       un gran número de características.
    2. Manejo de Desequilibrio: El parámetro `class_weight='balanced'` ayuda a
       mitigar el problema de que algunas etapas del sueño (como N1) son raras.
    3. Interpretabilidad: Permite calcular la importancia de cada característica,
       lo que es crucial para la tesis al permitir analizar qué biomarcadores
       fueron más útiles para la clasificación.
    """
    print("\n--- Fase 3: Entrenamiento y Evaluación del Modelo ---")

    # Crear el directorio de resultados si no existe
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. Cargar datos procesados
    print("Cargando características y etiquetas procesadas...")
    X = pd.read_csv(ruta_caracteristicas)
    y = pd.read_csv(ruta_etiquetas).values.ravel()

    # 2. División en conjuntos de entrenamiento y prueba
    print("Dividiendo los datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} épocas")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} épocas")

    # 3. Entrenamiento del modelo Random Forest
    print("Entrenando el modelo de Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    # 4. Evaluación del modelo
    print("Evaluando el rendimiento del modelo en el conjunto de prueba...")
    y_pred = rf_model.predict(X_test)

    target_names = ['Vigilia (0)', 'Ligero (1)', 'Profundo (2)', 'REM (3)']
    
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print("\n--- Resultados de la Evaluación ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa Score (κ): {kappa:.4f} (Un valor > 0.6 es bueno)")
    print("\nInforme de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 5. Visualización de la Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    ruta_cm = 'results/confusion_matrix.png'
    plt.savefig(ruta_cm)
    plt.show()
    print(f"Matriz de confusión guardada en '{ruta_cm}'")

    # 6. Análisis de Importancia de Características
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(20)
    
    plt.figure(figsize=(10, 10))
    sns.barplot(x=top_features, y=top_features.index)
    plt.title('Top 20 Características Más Importantes')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    ruta_fi = 'results/feature_importance.png'
    plt.savefig(ruta_fi)
    plt.show()
    print(f"Gráfico de importancia de características guardado en '{ruta_fi}'")