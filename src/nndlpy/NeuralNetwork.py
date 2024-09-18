import numpy as np

class NeuralNetwork:
    def __init__(self, size_input_layer, size_output_layer, activation_function_output_layer, activation_functions_hidden_layer, neurons_hidden_layer, error_function):
        # Costruzione ed inizializzazione della rete neurale
        #
        # Parametri:
        #   (int) size_input_layer: Dimensione dell'input layer
        #   (int) size_output_layer: Dimensione dell'output layer
        #   (activation function) activation_function_output_layer: Funzione di attivazione per l'output layer
        #   (list) activation_functions_hidden_layer: Lista per le funzioni di attivazione degli hidden layer
        #   (list) neurons_hidden_layer: Lista per il numero di neuroni di ogni hidden layer
        #   (error function) error_function: Funzione di errore
        #
        # Solleva:
        #   ValueError: Se il numero di funzioni di attivazione non corrisponde al numero di hidden layer specificati.

        self.MU = 0  # Media per la distribuzione normale
        self.SIGMA = 0.1  # Deviazione standard per la distribuzione normale

        self.layers_weights = []
        self.neurons_hidden_layers = neurons_hidden_layer

        if len(activation_functions_hidden_layer) != len(neurons_hidden_layer):
            raise ValueError("Il numero di funzioni di attivazione deve essere uguale al numero di neuroni per ogni hidden layer")

        self.activation_functions_hidden_layer = activation_functions_hidden_layer
        self.activation_function_output_layer = activation_function_output_layer
        self.error_function = error_function
        self.size_hidden_layer = len(neurons_hidden_layer)

        self.initialize_weights_bias(size_input_layer, size_output_layer)

    def initialize_weights_bias(self, size_input_layer, size_output_layer):
        # Inizializzazione dei pesi e del bias per tutti gli strati della rete
        #
        # Parametri:
        #  (int) size_input_layer: Dimensione dell'input layer
        #  (int) size_output_layer: Dimensione dell'output layer

        # Inizializzazione dei pesi e del bias per l'input layer
        self.layers_weights.insert(0, np.random.normal(self.MU, self.SIGMA, size=(self.neurons_hidden_layers[0], size_input_layer + 1)))

        # Inizializzazione dei pesi e dei bias per gli hidden layer
        for l in range(1, self.size_hidden_layer):
            self.layers_weights.insert(l, np.random.normal(self.MU, self.SIGMA, size=(self.neurons_hidden_layers[l], self.neurons_hidden_layers[l - 1] + 1)))

        # Inizializzazione dei pesi e del bias per l'output layer
        self.layers_weights.insert(self.size_hidden_layer, np.random.normal(self.MU, self.SIGMA, size=(size_output_layer, self.neurons_hidden_layers[-1] + 1)))

    def forward_propagation(self, input_data):
        # Implementazione della propagazione in avanti mediante pesi e bias.
        #
        # Parametri:
        #  (ndarray) input_data: Dati forniti in input.
        #
        # Ritorna:
        #    Uscita della rete neurale dopo la propagazione in avanti.

        num_layers = len(self.layers_weights)
        current_activations = input_data

        for l in range(num_layers):
            # Aggiunge il bias all'input del layer corrente
            bias_row = np.ones((1, current_activations.shape[1]))
            input_with_bias = np.insert(current_activations, 0, bias_row, axis=0)

            # Calcola l'output del layer corrente
            weights = self.layers_weights[l]
            z_values = np.dot(weights, input_with_bias)

            # Determina la funzione di attivazione da utilizzare per questo layer
            if l < self.size_hidden_layer:
                activation_function = self.activation_functions_hidden_layer[l]
            else:
                activation_function = self.activation_function_output_layer

            # Applica la funzione di attivazione per ottenere le nuove attivazioni
            current_activations = activation_function(z_values)

        return current_activations

    def compute_gradients(self, input_data, target):
        # Calcola i gradienti dei pesi utilizzando la backpropagation..
        #
        # Parametri:
        #   (ndarray) input_data: Dati di input.
        #   (ndarray) target: Valori target desiderati.
        #
        # Ritorna:
        #    Gradienti dei pesi per ciascuno strato della rete neurale.

        # Esegui la propagazione in avanti
        activations = self.forward_propagation(input_data)

        # Calcola l'errore
        error = self.error_function(target, activations)

        # Calcola i gradienti utilizzando la backpropagation
        gradients = []
        delta = error * self.error_function.derivative(target, activations)  # Derivata dell'errore rispetto all'output

        for layer_idx in reversed(range(len(self.layers_weights))):
            # Aggiorna i gradienti per i pesi del layer corrente
            activations_with_bias = np.insert(self.activations[layer_idx], 0, np.ones((1, input_data.shape[1])), axis=0)
            weights_gradient = np.dot(delta, activations_with_bias.T) / input_data.shape[1]
            gradients.insert(0, weights_gradient)

            if layer_idx > 0:
                # Propaga l'errore al layer precedente
                delta = np.dot(self.layers_weights[layer_idx][:, 1:].T, delta) * self.activation_functions_hidden_layer[
                    layer_idx - 1].derivative(self.activations[layer_idx - 1])

        return gradients

    def gradient_descent(self, learning_rate, input_data, target):
        # Aggiornamento dei pesi e dei bias utilizzando la discesa del gradiente.
        #
        # Parametri:
        #   (float) learning_rate: Tasso di apprendimento per il controllo della dimensione
        #                           dei passi dell'aggiornamento.
        #   (list) weights: Lista dei gradienti dei pesi per ciascun layer.
        #
        # Ritorna:
        #   Rete neurale aggiornata.

        # Calcola i gradienti dei pesi
        gradients = self.compute_gradients(input_data, target)

        # Aggiorna i pesi utilizzando il gradiente discendente
        for layer_idx in range(len(self.layers_weights)):
            self.layers_weights[layer_idx] -= learning_rate * gradients[layer_idx]

        return self

    def back_propagation(self, input_activations, layer_outputs, target, error_function):
        # Implementazione della back-propagation per calcolare i gradienti dei pesi e dei bias.
        #
        # Parametri:
        #   (list of numpy.ndarray) input_activations: Attivazioni di input di ciascun layer.
        #   (list of numpy.ndarray) layer_outputs: Output di ciascun layer.
        #   (numpy.ndarray) target: Target desiderato per l'output della rete.
        #   (error_function) error_function : Funzione di errore utilizzata per calcolare il gradiente.
        #
        # Ritorna:
        #   Tupla contenente i gradienti dei pesi e dei bias per ciascun layer.

        num_layers = len(self.layers_weights)
        weight_gradients = []
        bias_gradients = []

        # A partire dall'output layer, sono calcolati i gradienti dei pesi e dei bias per ciascun layer
        for l in range(num_layers - 1, -1, -1):
            if l == num_layers - 1:
                # Calcolo del delta per l'output layer
                output_error_derivative = error_function.derivative(layer_outputs[-1], target)
                delta = [input_activations[-1] * output_error_derivative]
            else:
                # Calcolo del delta per gli hidden layer
                activation_derivative = self.activation_functions_hidden_layer[l].derivative(layer_outputs[l])
                error_derivative = np.dot(self.layers_weights[l + 1][:, 1:].T, delta[0]) * activation_derivative
                delta = [input_activations[l] * error_derivative]

            # Calcolo del gradiente dei pesi per il layer corrente
            weight_gradient = np.dot(delta[0], layer_outputs[l].T)

            # Calcolo del gradiente del bias per il layer corrente
            bias_gradient = np.sum(delta[0], axis=1, keepdims=True)

            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, bias_gradient)

        return weight_gradients, bias_gradients

    # def gradient_descent(self, learning_rate, weights):
    #     # Aggiornamento dei pesi e dei bias utilizzando la discesa del gradiente.
    #     #
    #     # Parametri:
    #     #   (float) learning_rate: Tasso di apprendimento per il controllo della dimensione
    #     #                           dei passi dell'aggiornamento.
    #     #   (list) weights: Lista dei gradienti dei pesi per ciascuno strato.
    #     #
    #     # Ritorna:
    #     #   Rete neurale aggiornata.
    #
    #     for l in range(len(self.layers_weights)):
    #         # Aggiornamento dei pesi utilizzando il gradiente discendente
    #         self.layers_weights[l] -= learning_rate * weights[l]
    #
    #     return self