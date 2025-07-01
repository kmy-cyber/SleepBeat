import pyPPG.preproc as PP

def preprocesar_signal(signal_obj):
    """
    Aplica un filtro paso banda a la señal PPG cruda.
    Justificación: El preprocesamiento es crucial para eliminar ruido de baja frecuencia
    (deriva de la línea base) y de alta frecuencia. pyPPG utiliza un filtro Chebyshev II
    de 4º orden (0.5-12 Hz), una elección robusta y validada en la literatura para
    preservar la morfología de la onda de pulso. [1]
    """
    # Parámetros de filtrado estándar recomendados por la investigación
    preproc = PP.Preprocess(fL=0.5, fH=12, order=4)
    
    # La función get_signals devuelve la señal PPG filtrada y sus derivadas
    signal_obj.ppg, signal_obj.vpg, signal_obj.apg, signal_obj.jpg = preproc.get_signals(s=signal_obj)
    
    return signal_obj
