from copy import deepcopy
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

    def _evaluate_model(self, output, labels, metric='accuracy'):
        """
        Valuta le prestazioni della rete neurale confrontando le previsioni con i target desiderati.

        Args:
            output (numpy.ndarray): Array contenente le previsioni della rete.
            labels (numpy.ndarray): Array contenente i target desiderati.
            metric (str): La metrica da calcolare ('accuracy', 'precision', 'recall').

        Returns:
            float: Valore della metrica calcolata.
        """
        num_samples = labels.shape[1]

        # Applica la funzione softmax alle previsioni della rete
        softmax_predictions = LossFunctions.softmax(output)

        # Trova le classi predette
        predicted_classes = np.argmax(softmax_predictions, axis=0)
        target_classes = np.argmax(labels, axis=0)

        if metric == 'accuracy':
            correct_predictions = np.sum(predicted_classes == target_classes)
            return correct_predictions / num_samples

        elif metric == 'precision':
            true_positives = np.sum((predicted_classes == target_classes) & (predicted_classes == 1))
            predicted_positives = np.sum(predicted_classes == 1)
            return true_positives / predicted_positives if predicted_positives > 0 else 0.0

        elif metric == 'recall':
            true_positives = np.sum((predicted_classes == target_classes) & (predicted_classes == 1))
            actual_positives = np.sum(target_classes == 1)
            return true_positives / actual_positives if actual_positives > 0 else 0.0

        else:
            raise ValueError("Metric non valida. Usa 'accuracy', 'precision' o 'recall'.")

    def _compute_activations_and_derivatives(self, input_data):
        """
        Calcola gli output dei neuroni e le derivate delle funzioni di attivazione per la retropropagazione.

        Args:
            input_data (numpy.ndarray): I dati di input.

        Returns:
            tuple: Una tupla contenente gli output dei neuroni di ogni layer e le derivate delle funzioni di attivazione.
        """
        # Inizializza le liste per gli output e le derivate
        layer_outputs = [input_data]
        activation_derivatives = []

        for layer_idx, (weights, activation_function) in enumerate(zip(self.weights, self.activation_functions)):
            # Aggiungi il bias all'input del layer corrente
            input_with_bias = np.vstack((np.ones((1, layer_outputs[layer_idx].shape[1])), layer_outputs[layer_idx]))

            # Calcola l'output del layer
            linear_output = np.dot(weights, input_with_bias)
            layer_output = activation_function(linear_output)

            # Calcola la derivata dell'attivazione
            derivative_activation = activation_function(linear_output, der=True)

            # Memorizza gli output e le derivate
            layer_outputs.append(layer_output)
            activation_derivatives.append(derivative_activation)

        return layer_outputs, activation_derivatives

    @staticmethod
    def standard_rprop(weights_der, weights_delta, layer_weights_difference, layer_idx, row_idx, col_idx):
        return -np.sign(weights_der[layer_idx][row_idx][col_idx]) * weights_delta[layer_idx][row_idx][col_idx]

    @staticmethod
    def rprop_plus(weights_der, weights_delta, layer_weights_difference_prev, train_error, train_error_prev,
                   layer_idx, row_idx, col_idx):
        return -layer_weights_difference_prev[layer_idx][row_idx][col_idx]

    @staticmethod
    def irprop(weights_der, weights_delta, layer_weights_difference, layer_idx, row_idx, col_idx):
        return -np.sign(weights_der[layer_idx][row_idx][col_idx]) * weights_delta[layer_idx][row_idx][col_idx]

    def rprop_update(self, weights_der, weights_delta, layer_weights_difference_prev,
                     train_error, train_error_prev, eta_pos=1.2, eta_neg=0.5,
                     delta_max=50, delta_min=0.00001, rprop_type='STANDARD'):

        layer_weights_difference = layer_weights_difference_prev

        for layer in range(len(self.weights)):
            for row_idx in range(len(weights_der[layer])):
                for col_idx in range(len(weights_der[layer][row_idx])):
                    if rprop_type == 'STANDARD':
                        layer_weights_difference[layer][row_idx][col_idx] = self.standard_rprop(
                            weights_der, weights_delta, layer_weights_difference, layer, row_idx, col_idx)
                    elif rprop_type == 'RPROP_PLUS':
                        layer_weights_difference[layer][row_idx][col_idx] = self.rprop_plus(
                            weights_der, weights_delta, layer_weights_difference_prev, train_error, train_error_prev,
                            layer, row_idx, col_idx)
                    elif rprop_type == 'IRPROP':
                        layer_weights_difference[layer][row_idx][col_idx] = self.irprop(
                            weights_der, weights_delta, layer_weights_difference, layer, row_idx, col_idx)

                    # Aggiorna i pesi qui
                    self.weights[layer][row_idx][col_idx] += layer_weights_difference[layer][row_idx][col_idx]

        return layer_weights_difference

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

    def train(self, train_X, train_Y, validation_X, validation_Y, epochs=35,
              learning_rate=0.00001, rprop_type='STANDARD'):
        """
        Processo di apprendimento per la rete neurale.

        Args:
            train_X (numpy.ndarray): Dati di input per il training.
            train_Y (numpy.ndarray): Target desiderati per i dati di input di training.
            validation_X (numpy.ndarray): Dati di input per la validazione.
            validation_Y (numpy.ndarray): Target desiderati per i dati di input di validazione.
            epochs (int, optional): Numero massimo di epoche per il training (default: 35).
            learning_rate (float, optional): Tasso di apprendimento per il gradiente discendente (default: 0.00001).
            rprop_type (str): Tipo di Rprop da utilizzare (default: 'STANDARD').

        Returns:
            tuple: Una tupla contenente:
                - train_errors (list): Lista degli errori di training per ogni epoca.
                - validation_errors (list): Lista degli errori di validazione per ogni epoca.
                - train_accuracies (list): Lista delle accuratezze di training per ogni epoca.
                - validation_accuracies (list): Lista delle accuratezze di validazione per ogni epoca.
        """
        train_errors = []
        validation_errors = []
        train_accuracies = []
        validation_accuracies = []
        error_function = self.loss_function  # Uso della funzione di perdita definita nella classe

        # Inizializzazione delle variabili per Rprop
        weights_delta, prev_weights_der, weight_diff = None, None, None

        prev_validation_error = float('inf')
        min_validation_error = float('inf')

        # Duplica la rete
        best_network = self.clone_network()

        start_time = time.time()

        for epoch in range(epochs):

            # Propagazione in avanti sul training set
            train_output = self._forward_propagation(train_X)
            train_error = error_function(train_output, train_Y)
            train_errors.append(train_error)

            # Propagazione in avanti sul validation set
            validation_output = self._forward_propagation(validation_X)
            validation_error = error_function(validation_output, validation_Y)
            validation_errors.append(validation_error)

            train_accuracy = self._evaluate_model(train_output, train_Y)
            validation_accuracy = self._evaluate_model(validation_output, validation_Y)
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)

            print(f'Epoca: {epoch + 1}/{epochs}  Rprop utilizzata: {rprop_type}\n'
                  f'    Accuratezza Training: {np.round(train_accuracy, 5)},  Perdita Training: {np.round(train_error, 5)};\n'
                  f'    Accuratezza Validation: {np.round(validation_accuracy, 5)},  Perdita Validation: {np.round(validation_error, 5)}\n')

            # Calcolo dei gradienti e backpropagation
            layer_outputs, layer_derivatives = self._compute_activations_and_derivatives(train_X)
            gradients = self._back_propagation(layer_derivatives, layer_outputs, train_Y, error_function)

            if epoch == 0:  # Prima epoca
                self._gradient_descent(learning_rate, gradients)  # Aggiornamento dei pesi

                # Inizializzazione per Rprop
                weights_delta = [[[0.1 for _ in row] for row in sub_list] for sub_list in gradients]
                weight_diff = [[[0. for _ in row] for row in sub_list] for sub_list in gradients]

                prev_weights_der = deepcopy(gradients)
            else:
                # Aggiornamento usando Rprop
                weight_diff = self.rprop_update(gradients, weights_delta, prev_weights_der, weight_diff,
                                                validation_error, prev_validation_error, rprop_type)

            prev_validation_error = validation_error  # Aggiorna l'errore di validazione

            # Salva la rete migliore se l'errore di validazione è migliorato
            if validation_error < min_validation_error:
                min_validation_error = validation_error
                best_network = self.clone_network()  # Salva la migliore rete

        end_time = time.time()  # Ferma il timer

        print("L'addestramento ha impiegato", round(end_time - start_time, 5), "secondi.")

        # Copia i parametri della rete migliore nella rete corrente
        best_network._clone_network_params(self)

        return train_errors, validation_errors, train_accuracies, validation_accuracies

def metrics_mae_rmse_accuracy(metrics_list, epochs, number_of_runs):
    """
    Calcola l'Errore Assoluto Medio (MAE), l'Errore Quadratico Medio (RMSE) e l'accuratezza delle metriche
    per ogni epoca attraverso diverse esecuzioni di addestramento.

    Args:
        metrics_list (list): Una lista di liste contenente le metriche ottenute da diverse esecuzioni di addestramento.
                             Ogni sottolista corrisponde a una singola esecuzione e contiene le metriche calcolate per ogni epoca.
        epochs (int): Il numero totale di epoche.
        number_of_runs (int): Il numero totale di esecuzioni di addestramento.

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


