import numpy as np

def sigmoid(x, der=False):
    """
    Calcola la funzione sigmoide o la sua derivata.

    Args:
        x (float): Il valore di input.
        der (bool, optional): Se è False, calcola la funzione sigmoide, altrimenti calcola la derivata. Default è False.

    Returns:
        float: Se der=False, restituisce il valore della funzione sigmoide di x, altrimenti restituisce la sua derivata.
    """
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)  # Derivata della funzione sigmoide
