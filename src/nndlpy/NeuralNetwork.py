import numpy as np

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
        # self.hidden_activation_functions = hidden_activations
        # self.hidden_activation_functions.append(output_activation)
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
        # activations = [fn.__name__ for fn in self.hidden_activation_functions]

        print(f"Numero di layer nascosti: {hidden_layer_count}")
        print(f"Dimensione dell'input: {input_dim}")
        print(f"Dimensione dell'output: {output_dim}")
        print(f"Neuroni nei layer nascosti: {neurons_per_hidden_layer}")
        print(f"Funzioni di attivazione: {', '.join(activations)}")
        print(f"Funzione di perdita: {self.loss_function.__name__}")