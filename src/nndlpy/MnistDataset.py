import numpy as np

def get_mnist_training(dataset):
    """
    Funzione per creare input e target per il training set a partire dal MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.

    Returns:
        tuple: Dati di input di training e relative etichette in formato one-hot.
    """
    # Estrai i dati di training dal dataset
    data_train = dataset[:len(dataset)].T  # Trasponiamo per ottenere immagini come colonne

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

def get_mnist_test(dataset, test_size):
    """
    Funzione per creare input e target per il test set a partire dal MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.
        test_size (int): Numero di esempi nel test set.

    Returns:
        tuple: Dati di input di test e relative etichette in formato one-hot.
    """
    # Estrai i dati di test dal dataset
    data_test = dataset[:test_size].T  # Trasponiamo per ottenere immagini come colonne

    # Estrai le etichette di test
    test_Y = data_test[0]  # Prima riga contiene le etichette

    # Converti le etichette in formato one-hot
    test_Y = get_mnist_labels(test_Y)

    # Estrai i dati di input di test e normalizza dividendo per 255
    test_X = data_test[1:]  # Restanti righe contengono le immagini
    test_X = test_X / 255.  # Normalizzazione

    return test_X, test_Y

def get_mnist_validation(dataset):
    """
    Funzione per creare input e target per il validation set a partire dal MNIST.

    Args:
        dataset (numpy.ndarray): Il dataset MNIST completo.

    Returns:
        tuple: Dati di input di validazione e relative etichette in formato one-hot.
    """
    # Estrai i dati di validazione dal dataset, che è la parte rimanente dopo training e test
    data_val = dataset[:len(dataset)].T  # Trasponiamo per ottenere immagini come colonne

    # Estrai le etichette di validazione
    validation_Y = data_val[0]  # Prima riga contiene le etichette

    # Converti le etichette in formato one-hot
    validation_Y = get_mnist_labels(validation_Y)

    # Estrai i dati di input di validazione e normalizza dividendo per 255
    validation_X = data_val[1:]  # Restanti righe contengono le immagini
    validation_X = validation_X / 255.  # Normalizzazione

    return validation_X, validation_Y