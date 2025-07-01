import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_matriz_confusion(cm, clases, ruta_salida, titulo='Matriz de Confusión'):
    """
    Crea y guarda un gráfico de la matriz de confusión.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title(titulo)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"Matriz de confusión guardada en '{ruta_salida}'")

def plot_importancia_caracteristicas(importancias, ruta_salida, top_n=20):
    """
    Crea y guarda un gráfico de las características más importantes.
    """
    plt.figure(figsize=(10, 8))
    importancias.nlargest(top_n).plot(kind='barh')
    plt.title(f'Top {top_n} Características Más Importantes')
    plt.xlabel('Importancia')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"Gráfico de importancia de características guardado en '{ruta_salida}'")
