import numpy as np

def get_mnist_training(dataset, training_size):
    """
    Funzione per creare input e target per il training set a partire dal MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.
        training_size (int): Numero di esempi da usare per il training.

    Returns:
        tuple: Dati di input di training e relative etichette in formato one-hot.
    """
    # Estrai i dati di training dal dataset, usando il numero di esempi specificato
    data_train = dataset[:training_size].T  # Trasponiamo per ottenere immagini come colonne

    # Estrai le etichette di training
    train_Y = data_train[0]  # Prima riga, che contiene le etichette

    # Converti le etichette in formato one-hot
    train_Y = get_mnist_labels(train_Y)

    # Estrai i dati di input di training e normalizza dividendo per 255
    train_X = data_train[1:]  # Restanti righe contengono le immagini
    train_X = train_X / 255.  # Normalizzazione

    return train_X, train_Y

def get_mnist_labels(labels):
    """
    Funzione per convertire le etichette in formato one-hot.

    Args:
        labels (numpy.ndarray): Le etichette da convertire.

    Returns:
        numpy.ndarray: Etichette convertite in formato one-hot.
    """
    labels = np.array(labels, dtype=int)
    num_labels = labels.shape[0]
    num_classes = 10  # Numero di classi (0-9)

    one_hot_labels = np.zeros((num_classes, num_labels), dtype=int)

    for idx, label in enumerate(labels):
        one_hot_labels[label, idx] = 1  # Assegna 1 alla posizione corrispondente

    return one_hot_labels