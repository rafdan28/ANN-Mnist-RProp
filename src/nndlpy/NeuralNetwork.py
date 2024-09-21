from copy import deepcopy

from matplotlib import pyplot as plt
from nndlpy import LossFunctions as LossFunctions

import numpy as np
import time


class NeuralNetwork:
    MEAN, STD_DEV = 0, 0.1

    def __init__(self, hidden_activations, output_activation, loss_function,
                 input_size, hidden_layer_sizes, output_size):
        """
        Costruttore per l'inizializzazione della rete neurale.

        Args:
            hidden_activations (list): Funzioni di attivazione per gli strati nascosti.
            output_activation (function): Funzione di attivazione per lo strato di output.
            loss_function (function): Funzione di perdita (errore).
            input_size (int): Numero di unità nello strato di input.
            hidden_layer_sizes (list): Dimensioni degli strati nascosti.
            output_size (int): Numero di unità nello strato di output.
        """
        self.weights = []
        self.hidden_layers = hidden_layer_sizes
        self.loss_function = loss_function

        if len(hidden_activations) != len(hidden_layer_sizes):
            raise ValueError("Discrepanza tra numero di layer nascosti e funzioni di attivazione.")

        self.activation_functions = hidden_activations + [output_activation]
        self.num_hidden_layers = len(hidden_layer_sizes)
        self._initialize_weights(input_size, output_size)

    def _initialize_weights(self, input_size, output_size):
        """
        Inizializza i pesi e i bias per ogni strato.

        Args:
            input_size (int): Numero di unità nello strato di input.
            output_size (int): Numero di unità nello strato di output.
        """
        hidden_sizes = self.hidden_layers

        # Pesi per lo strato di input
        self._set_weights(0, hidden_sizes[0], input_size)

        # Pesi per gli strati nascosti
        for i in range(1, self.num_hidden_layers):
            self._set_weights(i, hidden_sizes[i], hidden_sizes[i - 1])

        # Pesi per lo strato di output
        self._set_weights(self.num_hidden_layers, output_size, hidden_sizes[-1])

    def _set_weights(self, layer_idx, num_neurons, num_inputs):
        """
        Assegna i pesi per uno specifico layer.

        Args:
            layer_idx (int): Indice del layer.
            num_neurons (int): Numero di neuroni nel layer.
            num_inputs (int): Numero di input del layer.
        """
        self.weights.append(np.random.normal(self.MEAN, self.STD_DEV, (num_neurons, num_inputs + 1)))

    def get_network(self):
        """
        Restituisce la descrizione della struttura della rete.
        """
        hidden_layer_count = self.num_hidden_layers
        input_dim = self.weights[0].shape[1] - 1
        output_dim = self.weights[-1].shape[0]

        neurons_per_hidden_layer = [self.weights[i].shape[0] for i in range(hidden_layer_count)]
        activations = [fn.__name__ for fn in self.activation_functions]

        print(f"Numero di layer nascosti: {hidden_layer_count}")
        print(f"Dimensione dell'input: {input_dim}")
        print(f"Dimensione dell'output: {output_dim}")
        print(f"Neuroni nei layer nascosti: {neurons_per_hidden_layer}")
        print(f"Funzioni di attivazione: {', '.join(activations)}")
        print(f"Funzione di perdita: {self.loss_function.__name__}")

    def clone_network(self):
        """
        Crea una copia esatta e indipendente della rete neurale.

        Returns:
            NeuralNetwork: Una nuova istanza che rappresenta una copia della rete neurale originale.
        """
        return deepcopy(self)

    def _forward_propagation(self, input_data):
        """
        Esegue la propagazione in avanti attraverso tutti i layer della rete neurale.

        Args:
            input_data (ndarray): Input iniziale fornito alla rete.

        Returns:
            ndarray: Output finale della rete neurale dopo aver attraversato tutti i layer.
        """
        # Inizializza con i dati di input
        i = input_data

        # Propaga attraverso ogni layer (nascosti + output)
        for layer_idx in range(len(self.weights)):
            # Aggiungi il bias (x_0 = 1) all'input del layer corrente
            bias = np.ones((1, i.shape[1]))
            i = np.vstack((bias, i))

            # Calcola l'output del layer corrente: i = W * i
            output = np.matmul(self.weights[layer_idx], i)

            # Applica la funzione di attivazione del layer corrente
            i = self.activation_functions[layer_idx](output)

        return i

    def _gradient_descent(self, learning_rate, weights_der):
        """
        Applica la discesa del gradiente per ottimizzare i pesi e i bias della rete neurale.

        Args:
            learning_rate (float): Tasso di apprendimento che determina l'entità degli aggiornamenti.
            weights_der (list): Lista contenente i gradienti dei pesi per ogni layer.

        Returns:
            NeuralNetwork: Istanza aggiornata della rete neurale con i nuovi pesi.
        """
        for layer_idx in range(len(self.weights)):
            # Aggiornamento dei pesi utilizzando il gradiente
            self.weights[layer_idx] -= learning_rate * weights_der[layer_idx]

        return self

    def _back_propagation(self, input_activations, layer_outputs, target, error_function):
        """
        Calcola i gradienti per i pesi e i bias attraverso il processo di back-propagation.

        Args:
            input_activations (list of numpy.ndarray): Attivazioni di input per ciascun layer.
            layer_outputs (list of numpy.ndarray): Output generati da ciascun layer.
            target (numpy.ndarray): Valore target da raggiungere per l'output finale.
            error_function (function): Funzione di errore usata per derivare il gradiente.

        Returns:
            list: Lista contenente i gradienti dei pesi per ogni layer.
        """
        # Estrazione dei pesi e numero di layer
        global delta
        weights = self.weights
        num_layers = len(weights)

        # Inizializzazione della lista per i gradienti dei pesi
        weight_gradients = []

        # Calcolo dei gradienti partendo dall'output verso l'input
        for layer_idx in range(num_layers - 1, -1, -1):
            # Calcolo del delta per il layer corrente
            if layer_idx == num_layers - 1:
                # Per l'ultimo layer, calcola il delta basato sull'errore
                output_error = error_function(layer_outputs[-1], target, der=True)
                delta = input_activations[-1] * output_error
            else:
                # Per i layer nascosti, usa il delta del layer successivo
                delta = input_activations[layer_idx] * np.matmul(weights[layer_idx + 1][:, 1:].T, delta)

            # Calcolo del gradiente dei pesi per il layer corrente
            weight_gradient = np.matmul(delta, layer_outputs[layer_idx].T)

            # Calcolo del gradiente del bias
            bias_gradient = np.sum(delta, axis=1, keepdims=True)
            weight_gradient = np.hstack((bias_gradient, weight_gradient))

            # Aggiungi il gradiente calcolato nella lista
            weight_gradients.insert(0, weight_gradient)

        return weight_gradients

    def _clone_network_params(self, destination_net):
        """
        Trasferisce i parametri (pesi, funzioni di attivazione) da una rete sorgente a una rete di destinazione.

        Args:
            self (NeuralNetwork): Rete sorgente da cui copiare i parametri.
            destination_net (NeuralNetwork): Rete di destinazione in cui copiare i parametri.
        """
        # Copia dei pesi per ogni strato
        for idx, weights in enumerate(self.weights):
            destination_net.weights[idx] = np.copy(weights)

        # Copia delle funzioni di attivazione
        destination_net.activation_functions = list(self.activation_functions)

    def activations_derivatives_calc(self, input_data):
        """
        Calcola gli output per ogni layer e le derivate delle funzioni di attivazione
        necessarie per la backpropagation.

        Args:
            input_data (numpy.ndarray): Dati di input forniti alla rete neurale.

        Returns:
            tuple: Due liste contenenti gli output di ciascun layer e le relative derivate.
        """

        # Recupera i pesi e le funzioni di attivazione
        weights = self.weights
        activation_functions = self.activation_functions
        num_layers = len(weights)

        # Inizializza le liste per gli output dei layer e le derivate
        layer_outputs = [input_data]
        activation_derivatives = []

        for layer in range(num_layers):
            # Inserisce il bias e calcola l'output lineare
            input_with_bias = np.vstack((np.ones((1, layer_outputs[layer].shape[1])), layer_outputs[layer]))
            linear_output = np.dot(weights[layer], input_with_bias)

            # Calcola l'output finale e la derivata della funzione di attivazione
            layer_output = activation_functions[layer](linear_output)
            derivative_activation = activation_functions[layer](linear_output, der=True)

            # Aggiorna le liste degli output e delle derivate
            layer_outputs.append(layer_output)
            activation_derivatives.append(derivative_activation)

        return layer_outputs, activation_derivatives

    def update_weights_rprop(self, gradients, weight_updates, previous_gradients, previous_weight_updates,
                             current_error,
                             previous_error, positive_eta=1.2, negative_eta=0.5, max_delta=50, min_delta=0.00001,
                             rprop_method='STANDARD'):
        """
        Aggiorna i pesi della rete neurale utilizzando l'algoritmo Rprop. Supporta diverse varianti
        come descritto nell'articolo "Empirical evaluation of the improved Rprop learning algorithms".

        Args:
            gradients (list): Gradienti attuali dei pesi per ogni strato.
            weight_updates (list): Aggiornamenti dei pesi per ogni strato.
            previous_gradients (list): Gradienti dei pesi dalla iterazione precedente.
            previous_weight_updates (list): Aggiornamenti dei pesi dalla iterazione precedente.
            current_error (float): Errore per l'epoca corrente.
            previous_error (float): Errore per l'epoca precedente.
            positive_eta (float): Fattore di incremento per il delta in caso di derivata positiva (default: 1.2).
            negative_eta (float): Fattore di decremento per il delta in caso di derivata negativa (default: 0.5).
            max_delta (float): Limite superiore per il delta (default: 50).
            min_delta (float): Limite inferiore per il delta (default: 0.00001).
            rprop_method (str): Metodo Rprop da utilizzare (default: 'STANDARD').

        Returns:
            list: Aggiornamenti dei pesi calcolati per ogni strato.
        """

        # Utilizzo di deepcopy per evitare modifiche non intenzionali ai dati
        updated_weight_diff = deepcopy(previous_weight_updates)

        def adjust_delta(gradient_product, current_update, eta_plus, eta_minus, max_d, min_d):
            if gradient_product > 0:
                return min(current_update * eta_plus, max_d), True
            elif gradient_product < 0:
                return max(current_update * eta_minus, min_d), False
            return current_update, None

        def update_weight(grad_value, delta, method_type):
            if method_type == 'STANDARD' or (method_type == 'IRPROP' and grad_value != 0):
                return -np.sign(grad_value) * delta
            return 0

        for layer_idx, layer_weights in enumerate(self.weights):
            for row_idx, row in enumerate(layer_weights):
                for col_idx, _ in enumerate(row):
                    gradient_product = previous_gradients[layer_idx][row_idx][col_idx] * gradients[layer_idx][row_idx][
                        col_idx]
                    delta, sign_change = adjust_delta(gradient_product, weight_updates[layer_idx][row_idx][col_idx],
                                                      positive_eta, negative_eta, max_delta, min_delta)

                    weight_updates[layer_idx][row_idx][col_idx] = delta
                    grad_value = gradients[layer_idx][row_idx][col_idx]

                    updated_value = update_weight(grad_value, delta, rprop_method)
                    if sign_change is False and (rprop_method == 'RPROP_PLUS' or current_error > previous_error):
                        updated_value = -previous_weight_updates[layer_idx][row_idx][col_idx]

                    updated_weight_diff[layer_idx][row_idx][col_idx] = updated_value
                    self.weights[layer_idx][row_idx][col_idx] += updated_value

                    if sign_change is not None:
                        previous_gradients[layer_idx][row_idx][col_idx] = grad_value
                        previous_weight_updates[layer_idx][row_idx][col_idx] = updated_value

        return updated_weight_diff

    def train_model(self, training_data, training_labels, validation_data, validation_labels, num_epochs=35,
                    learning_rate=0.00001, rprop_method='STANDARD'):
        """
        Gestisce il processo di apprendimento della rete neurale.

        Args:
            training_data (numpy.ndarray): Dati di input per l'addestramento.
            training_labels (numpy.ndarray): Target desiderati per i dati di input di addestramento.
            validation_data (numpy.ndarray): Dati di input per la validazione.
            validation_labels (numpy.ndarray): Target desiderati per i dati di input di validazione.
            num_epochs (int, optional): Numero massimo di epoche per l'addestramento (default: 35).
            learning_rate (float, optional): Tasso di apprendimento per l'ottimizzazione (default: 0.00001).
            rprop_method (str): Tipo di metodo Rprop da utilizzare (default: 'STANDARD').

        Returns:
            tuple: Una tupla contenente:
                - training_errors (list): Lista degli errori di addestramento per ogni epoca.
                - validation_errors (list): Lista degli errori di validazione per ogni epoca.
                - training_accuracies (list): Lista delle accuratezze di addestramento per ogni epoca.
                - validation_accuracies (list): Lista delle accuratezze di validazione per ogni epoca.
        """

        def log_epoch_info(epoch_num, max_epochs, train_acc, train_err, val_acc, val_err, method):
            print(f'\nEpoch: {epoch_num}/{max_epochs}   Rprop used: {method}\n'
                  f'    Training Accuracy: {np.round(train_acc, 5)},       Training Loss: {np.round(train_err, 5)};\n'
                  f'    Validation Accuracy: {np.round(val_acc, 5)},     Validation Loss: {np.round(val_err, 5)}\n')

        training_errors, validation_errors = [], []
        training_accuracies, validation_accuracies = [], []

        best_model = self.clone_network()
        previous_validation_error = lowest_validation_error = float('inf')

        weights_update, previous_gradients, weight_differences = None, None, None
        start_time = time.time()

        for epoch in range(num_epochs + 1):
            # Propagazione in avanti
            training_output = self._forward_propagation(training_data)
            validation_output = self._forward_propagation(validation_data)

            # Funzione di errore
            current_training_error = self.loss_function(training_output, training_labels)
            current_validation_error = self.loss_function(validation_output, validation_labels)

            training_errors.append(current_training_error)
            validation_errors.append(current_validation_error)

            training_accuracies.append(calculate_accuracy(training_output, training_labels))
            validation_accuracies.append(calculate_accuracy(validation_output, validation_labels))

            # Stampa delle informazioni per le epoche
            log_epoch_info(epoch, num_epochs, training_accuracies[-1], current_training_error,
                           validation_accuracies[-1], current_validation_error, rprop_method)

            if epoch == num_epochs:
                break

            # Calcolo dei gradienti
            layer_outputs, activation_derivatives = self.activations_derivatives_calc(training_data)
            gradients = self._back_propagation(activation_derivatives, layer_outputs, training_labels,
                                               self.loss_function)

            if epoch == 0:  # Prima epoca
                self._gradient_descent(learning_rate, gradients)
                weights_update = [[[0.1 for _ in row] for row in layer] for layer in gradients]
                weight_differences = deepcopy(weights_update)
                previous_gradients = deepcopy(gradients)
            else:
                weight_differences = self.update_weights_rprop(gradients, weights_update, previous_gradients,
                                                               weight_differences, current_validation_error,
                                                               previous_validation_error, rprop_method=rprop_method)

            previous_validation_error = current_validation_error

            # Salvataggio del miglior modello
            if current_validation_error < lowest_validation_error:
                lowest_validation_error = current_validation_error
                best_model = self.clone_network()

        elapsed_time = time.time() - start_time
        print("Tempo impiegato per l'addestramento: ", round(elapsed_time, 5), "secondi.")

        best_model._clone_network_params(self)

        return training_errors, validation_errors, training_accuracies, validation_accuracies, elapsed_time


def plot_metrics(results):
    """
    Funzione per tracciare le metriche di addestramento e validazione.

    Args:
        results (list): Lista contenente i risultati delle reti addestrate.
    """
    train_errors, validation_errors, train_accuracies, validation_accuracies, _ = zip(*results)

    epochs_range = range(len(train_errors[0]))

    # Creazione della figura
    plt.figure(figsize=(14, 10))

    # Grafico degli errori di addestramento e validazione
    plt.subplot(2, 2, 1)
    for errors in train_errors:
        plt.plot(epochs_range, errors, label='Errore di addestramento')
    for errors in validation_errors:
        plt.plot(epochs_range, errors, label='Errore di validazione', linestyle='--')
    plt.title('Errori di Addestramento e Validazione')
    plt.xlabel('Epoche')
    plt.ylabel('Errore')
    plt.legend()
    plt.grid()

    # Grafico delle accuratezze di addestramento e validazione
    plt.subplot(2, 2, 2)
    for accuracies in train_accuracies:
        plt.plot(epochs_range, accuracies, label='Accuratezza di addestramento')
    for accuracies in validation_accuracies:
        plt.plot(epochs_range, accuracies, label='Accuratezza di validazione', linestyle='--')
    plt.title('Accuratezze di Addestramento e Validazione')
    plt.xlabel('Epoche')
    plt.ylabel('Accuratezza')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def calculate_accuracy(predictions, true_labels):
    """
    Calcola l'accuratezza della rete neurale confrontando le previsioni con i valori reali.

    Args:
        predictions (numpy.ndarray): Array contenente le previsioni della rete.
        true_labels (numpy.ndarray): Array contenente i valori reali.

    Returns:
        float: Percentuale di previsioni corrette rispetto ai valori reali.
    """
    num_samples = true_labels.shape[1]

    # Calcola le probabilità con la funzione softmax sulle previsioni
    probability_predictions = LossFunctions.softmax(predictions)

    # Ottiene le classi predette trovando l'indice del valore massimo lungo l'asse delle colonne
    predicted_classes = np.argmax(probability_predictions, axis=0)

    # Ottiene le classi target trovando l'indice del valore massimo negli obiettivi reali
    true_classes = np.argmax(true_labels, axis=0)

    # Conta le previsioni corrette confrontando le classi predette con quelle reali
    correct_predictions_count = np.sum(predicted_classes == true_classes)
    accuracy_ratio = correct_predictions_count / num_samples

    return accuracy_ratio


def metrics_mae_rmse_accuracy(metrics_list, epochs, number_of_runs):
    """
    Calcola l'Errore Assoluto Medio (MAE), l'Errore Quadratico Medio (RMSE) e l'accuratezza delle metriche
    per ogni epoca attraverso diverse esecuzioni di addestramento.

    Args: metrics_list (list): Una lista di liste contenente le metriche ottenute da diverse esecuzioni di
    addestramento. Ogni sottolista corrisponde a una singola esecuzione e contiene le metriche calcolate per ogni
    epoca. epochs (int): Il numero totale di epoche. number_of_runs (int): Il numero totale di esecuzioni di
    addestramento.

    Returns:
        Tuple: Una tupla contenente:
                - mae_list: Lista dei MAE per ogni epoca.
                - rmse_list: Lista dei RMSE per ogni epoca.
                - accuracy_list: Lista delle accuratezze per ogni epoca.
    """
    mae_list = []
    rmse_list = []
    accuracy_list = []

    for epoch in range(epochs + 1):
        mae = 0
        rmse = 0
        accuracy = 0

        for run in range(number_of_runs):
            # Calcola MAE
            mae += np.mean(np.abs(metrics_list[run][0][epoch] - metrics_list[run][1][epoch])) / number_of_runs

            # Calcola RMSE
            rmse += np.sqrt(
                np.mean((metrics_list[run][0][epoch] - metrics_list[run][1][epoch]) ** 2)) / number_of_runs

            # Calcola accuratezza (supponendo che le etichette siano nel formato appropriato)
            accuracy += np.mean(
                metrics_list[run][2][epoch]) / number_of_runs  # Assuming accuracy is stored in index 2

        mae_list.append(round(mae, 5))
        rmse_list.append(round(rmse, 5))
        accuracy_list.append(round(accuracy, 5))

    # Stampa dei risultati finali
    print("MAE per ogni epoca:")
    for epoch, value in enumerate(mae_list):
        print(f"Epoca {epoch}: {value}")

    print("\nRMSE per ogni epoca:")
    for epoch, value in enumerate(rmse_list):
        print(f"Epoca {epoch}: {value}")

    print("\nAccuratezza per ogni epoca:")
    for epoch, value in enumerate(accuracy_list):
        print(f"Epoca {epoch}: {value}")

    return mae_list, rmse_list, accuracy_list
