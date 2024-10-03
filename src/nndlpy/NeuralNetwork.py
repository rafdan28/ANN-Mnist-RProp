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

        self.activation_functions = hidden_activations
        self.activation_functions.append(output_activation)
        self.num_hidden_layers = len(hidden_layer_sizes)
        self._initialize_weights(input_size, output_size)

    def _initialize_weights(self, input_size, output_size):
        """
        Inizializza i pesi e i bias per ogni strato.

        Args:
            input_size (int): Numero di unità nello strato di input.
            output_size (int): Numero di unità nello strato di output.
        """

        def set_weights(num_neurons, num_inputs):
            """
            Assegna i pesi per uno specifico layer.

            Args:
                num_neurons (int): Numero di neuroni nel layer.
                num_inputs (int): Numero di input del layer.
            """
            self.weights.append(np.random.normal(self.MEAN, self.STD_DEV, (num_neurons, num_inputs + 1)))

        hidden_sizes = self.hidden_layers

        # Pesi per lo strato di input
        set_weights(hidden_sizes[0], input_size)

        # Pesi per gli strati nascosti
        for i in range(1, self.num_hidden_layers):
            set_weights(hidden_sizes[i], hidden_sizes[i - 1])

        # Pesi per lo strato di output
        set_weights(output_size, hidden_sizes[-1])

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

    def _update_weights_rprop(self, gradients, deltas, prev_gradients, prev_weight_changes, current_error,
                              previous_error, increase_factor=1.2, decrease_factor=0.5, max_delta=50, min_delta=0.00001,
                              rprop_method='STANDARD'):
        """
        Funzione Rprop per l'aggiornamento dei pesi in reti neurali multistrato. Implementa la versione standard e le varianti
        descritte nell'articolo "Empirical evaluation of the improved Rprop learning algorithms". Le varianti sono gestite
        tramite il parametro rprop_variant.

        Args:
            gradients (list): Lista dei gradienti dei pesi per ciascun strato.
            deltas (list): Lista dei delta dei pesi per ciascun strato.
            prev_gradients (list): Lista dei gradienti dei pesi della iterazione precedente.
            prev_weight_changes (list): Lista delle variazioni di peso della iterazione precedente.
            current_error (float): Errore dell'epoca corrente.
            previous_error (float): Errore dell'epoca precedente.
            increase_factor (float): Fattore di aggiornamento per derivata positiva (default: 1.2).
            decrease_factor (float): Fattore di aggiornamento per derivata negativa (default: 0.5).
            max_delta (float): Limite superiore per il delta (default: 50).
            min_delta (float): Limite inferiore per il delta (default: 0.00001).
            rprop_method (String): Tipo di Rprop da utilizzare (default: STANDARD).

        Returns:
            NeuralNetwork: Rete neurale aggiornata utilizzando il metodo Rprop.
        """

        # Inizializzazione delle variazioni di peso per l'attuale strato
        weight_changes = prev_weight_changes

        for layer_idx in range(len(self.weights)):
            layer_weights = self.weights[layer_idx]

            for row in range(len(gradients[layer_idx])):
                for col in range(len(gradients[layer_idx][row])):
                    gradient_product = prev_gradients[layer_idx][row][col] * gradients[layer_idx][row][col]

                    if gradient_product > 0:
                        # Calcolo della nuova dimensione del delta per i pesi
                        deltas[layer_idx][row][col] = min(deltas[layer_idx][row][col] * increase_factor, max_delta)

                        # Aggiornamento della variazione di peso
                        weight_changes[layer_idx][row][col] = -(
                                np.sign(gradients[layer_idx][row][col])
                                * deltas[layer_idx][row][col])

                    elif gradient_product < 0:
                        # Calcolo della nuova dimensione del delta per i pesi
                        deltas[layer_idx][row][col] = max(deltas[layer_idx][row][col] * decrease_factor, min_delta)

                        if rprop_method == 'STANDARD' or rprop_method == 'IRPROP':
                            # Aggiornamento della variazione di peso
                            weight_changes[layer_idx][row][col] = -(
                                    np.sign(gradients[layer_idx][row][col]) *
                                    deltas[layer_idx][row][col])
                        else:
                            if rprop_method == 'RPROP_PLUS' or current_error > previous_error:
                                # Aggiornamento della variazione di peso
                                weight_changes[layer_idx][row][col] = -prev_weight_changes[layer_idx][row][col]
                            else:
                                # Azzeramento della variazione di peso
                                weight_changes[layer_idx][row][col] = 0

                        if rprop_method != 'STANDARD':
                            # Azzeramento del gradiente del peso
                            gradients[layer_idx][row][col] = 0

                    else:
                        # Aggiornamento della variazione di peso
                        weight_changes[layer_idx][row][col] = -(
                                np.sign(gradients[layer_idx][row][col])
                                * deltas[layer_idx][row][col])

                    # Aggiornamento del peso
                    layer_weights[row][col] += weight_changes[layer_idx][row][col]

                    # Aggiornamento del gradiente del peso precedente
                    prev_gradients[layer_idx][row][col] = gradients[layer_idx][row][col]

                    prev_weight_changes[layer_idx][row][col] = weight_changes[layer_idx][row][col]

        return weight_changes

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

        def activations_derivatives_calc(input_data):
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

        def gradient_descent(weights_der):
            """
            Applica la discesa del gradiente per ottimizzare i pesi e i bias della rete neurale.

            Args:
                weights_der (list): Lista contenente i gradienti dei pesi per ogni layer.

            Returns:
                NeuralNetwork: Istanza aggiornata della rete neurale con i nuovi pesi.
            """
            for layer_idx in range(len(self.weights)):
                # Aggiornamento dei pesi utilizzando il gradiente
                self.weights[layer_idx] -= learning_rate * weights_der[layer_idx]

            return self

        def log_epoch_info(epoch_num, max_epochs, train_acc, train_err, val_acc, val_err, method):
            print(f'\nEpoch: {epoch_num}/{max_epochs}   Rprop used: {method}\n'
                  f'    Accuratezza sul Training Set: {np.round(train_acc, 5)},       Errore sul Training Set: {np.round(train_err, 5)};\n'
                  f'    Accuratezza sul Validation Set: {np.round(val_acc, 5)},     Errore sul Validation Set: {np.round(val_err, 5)}\n')

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

            training_accuracies.append(_calculate_accuracy(training_output, training_labels))
            validation_accuracies.append(_calculate_accuracy(validation_output, validation_labels))

            # Stampa delle informazioni per le epoche
            log_epoch_info(epoch, num_epochs, training_accuracies[-1], current_training_error,
                           validation_accuracies[-1], current_validation_error, rprop_method)

            if epoch == num_epochs:
                break

            # Calcolo dei gradienti
            layer_outputs, activation_derivatives = activations_derivatives_calc(training_data)
            gradients = self._back_propagation(activation_derivatives, layer_outputs, training_labels,
                                               self.loss_function)

            if epoch == 0:  # Prima epoca
                gradient_descent(gradients)
                weights_update = [[[0.1 for _ in row] for row in layer] for layer in gradients]
                weight_differences = deepcopy(weights_update)
                previous_gradients = deepcopy(gradients)
            else:
                weight_differences = self._update_weights_rprop(gradients, weights_update, previous_gradients,
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

    def print_accuracies(self, title, test_X, test_Y, train_X, train_Y):
        """
        Stampa il titolo specificato, seguito dall'accuratezza della rete neurale sui set di test e di addestramento.

        Argomenti:
            title (str): Il titolo da stampare prima di visualizzare le accuratezze.
            test_X (numpy.ndarray): Il set di input di test.
            test_Y (numpy.ndarray): Le etichette di test corrispondenti.
            train_X (numpy.ndarray): Il set di input di addestramento.
            train_Y (numpy.ndarray): Le etichette di addestramento corrispondenti.

        Returns:
            net_accuracy_test (float): L'accuratezza della rete neurale sul set di test.
        """

        def network_accuracy(X, Y):
            """
            Calcola l'accuratezza della rete neurale su un insieme di dati di input e target specificato.

            Args:
                X (numpy.ndarray): Dati di input su cui valutare la rete.
                Y (numpy.ndarray): Target desiderati per i dati di input.

            Returns:
                float: Percentuale di predizioni corrette rispetto ai target desiderati.
            """
            output = self._forward_propagation(X)
            return _calculate_accuracy(output, Y)

        print(title)
        net_accuracy_test = network_accuracy(test_X, test_Y)
        print(f'Accuratezza sul Test Set: {np.round(net_accuracy_test, 5)}')
        net_accuracy_training = network_accuracy(train_X, train_Y)
        print(f'Accuratezza sul Training Set: {np.round(net_accuracy_training, 5)}')
        return net_accuracy_test


def _calculate_accuracy(predictions, true_labels):
    """
    Calcola l'accuratezza della rete confrontando le previsioni con le etichette reali.

    Args:
        predictions (numpy.ndarray): Array con le previsioni della rete.
        true_labels (numpy.ndarray): Array con le etichette reali.

    Returns:
        float: Accuratezza come percentuale di previsioni corrette.
    """

    def apply_softmax(preds):
        return LossFunctions.softmax(preds)

    def get_predicted_classes(probs):
        return np.argmax(probs, axis=0)

    def get_true_classes(labels):
        return np.argmax(labels, axis=0)

    def calculate_correct_predictions(preds, true_cls):
        return np.sum(preds == true_cls)

    num_samples = true_labels.shape[1]

    # Calcola le probabilità utilizzando la funzione softmax
    probability_predictions = apply_softmax(predictions)

    # Ottiene le classi predette e le classi reali
    predicted_classes = get_predicted_classes(probability_predictions)
    true_classes = get_true_classes(true_labels)

    # Calcola il numero di previsioni corrette
    correct_predictions_count = calculate_correct_predictions(predicted_classes, true_classes)

    # Calcola l'accuratezza come rapporto tra le previsioni corrette e il numero totale di campioni
    accuracy_ratio = correct_predictions_count / num_samples

    return accuracy_ratio


def compute_metrics_statistics(metrics_data, num_epochs, num_runs):
    """
    Calcola la media e la varianza normalizzata delle metriche per ogni epoca su più esecuzioni.

    Args:
        metrics_data (list): Lista di liste contenente le metriche ottenute da diverse esecuzioni.
                             Ogni sottolista corrisponde a una singola esecuzione e contiene le metriche per ogni epoca.
        num_epochs (int): Numero totale di epoche.
        num_runs (int): Numero totale di esecuzioni.

    Returns:
        - avg_metrics: Lista delle medie delle metriche per ogni epoca.
        - norm_variances: Lista delle varianze normalizzate delle metriche per ogni epoca.
        - final_avg_metrics: Lista delle medie delle metriche all'ultima epoca.
        - final_norm_variances: Lista delle varianze normalizzate delle metriche all'ultima epoca.
    """
    num_metrics = len(metrics_data[0])
    mean_metrics = [[] for _ in range(num_metrics)]
    norm_variances = [[] for _ in range(num_metrics)]

    for metric_idx in range(num_metrics - 1):
        for epoch in range(num_epochs + 1):
            epoch_mean, epoch_variance = 0, 0
            for run_idx in range(num_runs):
                # Calcola la media della metrica per questa epoca su tutte le run
                epoch_mean += metrics_data[run_idx][metric_idx][epoch] / num_runs
            mean_metrics[metric_idx].append(epoch_mean)

            for run_idx in range(num_runs):
                # Calcola la varianza della metrica per questa epoca su tutte le run
                epoch_variance += pow(metrics_data[run_idx][metric_idx][epoch] - epoch_mean, 2) / num_runs
            # Calcola la varianza normalizzata rispetto alla media
            normalized_variance = epoch_variance / epoch_mean if epoch_mean != 0 else 0
            norm_variances[metric_idx].append(normalized_variance)

    # Calcolo della media e della varianza per i tempi di esecuzione
    execution_time_mean, execution_time_variance = 0, 0
    for run_idx in range(num_runs):
        execution_time_mean += metrics_data[run_idx][-1] / num_runs
    mean_metrics[-1] = execution_time_mean

    for run_idx in range(num_runs):
        execution_time_variance += pow(metrics_data[run_idx][-1] - execution_time_mean, 2) / num_runs
    norm_variances[-1] = execution_time_variance

    # Estrai le medie finali (dell'ultima epoca) di ogni metrica
    final_last_mean_metrics = [round(mean[-1], 5) if isinstance(mean, list) else round(mean, 5) for mean in mean_metrics]

    # Estrai le varianze finali (dell'ultima epoca) di ogni metrica
    final_last_norm_variances = [
        round(var[-1], 5) if isinstance(var, list) else round(var, 5) for var in norm_variances]

    return mean_metrics, norm_variances, final_last_mean_metrics, final_last_norm_variances

