import numpy as np

def identity(x, der = False):
    """
    Calcola la funzione identità o anche la sua derivata.

    Args:
        x (float): Il valore di input.
        der (bool, optional): Se è False, calcola la funzione identità, altrimenti calcola la derivata. Default è False.

    Returns:
        float: Se der=False, restituisce il valore di input x, altrimenti restituisce la sua derivata.
    """
    if not der:
        return x
    else:
        return 1  # La derivata della funzione identità è sempre 1

def sigmoid(x, der = False):
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

def relu(x, der = 0):
    """
    Calcola la funzione ReLU o la sua derivata.

    Args:
        x (float): Il valore di input.
        der (bool, optional): Se è False, calcola la funzione ReLU, altrimenti calcola la derivata. Default è False.

    Returns:
        float: Se der=False, restituisce il valore della funzione ReLU di x, altrimenti restituisce la sua derivata.
    """
    if not der:
        return np.maximum(0, x)
    else:
        return np.where(x > 0, 1, 0)  # Derivata della funzione ReLU

def tanh(x, der = False):
    """
    Calcola la tangente iperbolica o anche la sua derivata.

    Args:
        x (float): Il valore di input.
        der (bool, optional): Se è False, calcola la tangente iperbolica, altrimenti calcola la derivata. Default è False.

    Returns:
        float: Se der=False, restituisce il valore della tangente iperbolica di x, altrimenti restituisce la sua derivata.
    """
    y = np.tanh(x)
    if not der:
        return y
    else:
        return 1 - y**2  # Derivata della tangente iperbolica
