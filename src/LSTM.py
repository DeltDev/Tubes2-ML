import numpy as np

class EmbeddingLayer:
    def __init__(self):
        self.weights = None

    def load_weights(self, weights):
        self.weights = weights
    
    def forward(self, input_ids):
        return self.weights[input_ids]
        
class LSTMCell:
    def __init__(self):
        self.Wf = None  # Forget gate weights
        self.Wi = None  # Input gate weights
        self.Wc = None  # Cell state weights
        self.Wo = None  # Output gate weights

        self.Uf = None  # Forget gate recurrent weights
        self.Ui = None  # Input gate recurrent weights
        self.Uc = None  # Cell state recurrent weights
        self.Uo = None  # Output gate recurrent weights

        self.bf = None  # Forget gate bias
        self.bi = None  # Input gate bias
        self.bc = None  # Cell state bias
        self.bo = None  # Output gate bias

    def load_weights(self, kernel, recurrent_kernel, bias):
        input_size, hidden_size = kernel.shape
        hidden_size = hidden_size // 4  # LSTM has 4 gates
        
        self.Wf = kernel[:, :hidden_size]
        self.Wi = kernel[:, hidden_size:hidden_size*2]
        self.Wc = kernel[:, hidden_size*2:hidden_size*3]
        self.Wo = kernel[:, hidden_size*3:]

        self.Uf = recurrent_kernel[:, :hidden_size]
        self.Ui = recurrent_kernel[:, hidden_size:hidden_size*2]
        self.Uc = recurrent_kernel[:, hidden_size*2:hidden_size*3]
        self.Uo = recurrent_kernel[:, hidden_size*3:]

        self.bf = bias[:hidden_size]
        self.bi = bias[hidden_size:hidden_size*2]
        self.bc = bias[hidden_size*2:hidden_size*3]
        self.bo = bias[hidden_size*3:]
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x_t, h_prev, c_prev):
        # Forget gate
        f = self.sigmoid(np.dot(x_t, self.Wf) + np.dot(h_prev, self.Uf) + self.bf)
        
        # Input gate
        i = self.sigmoid(np.dot(x_t, self.Wi) + np.dot(h_prev, self.Ui) + self.bi)
        
        # Cell state
        c_hat = self.tanh(np.dot(x_t, self.Wc) + np.dot(h_prev, self.Uc) + self.bc)
        c = f * c_prev + i * c_hat
        
        # Output gate
        o = self.sigmoid(np.dot(x_t, self.Wo) + np.dot(h_prev, self.Uo) + self.bo)
        
        # Hidden state
        h = o * self.tanh(c)
        
        return h, c

class LSTMLayer:
    def __init__(self, hidden_size, return_sequences=False, bidirectional=True):
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.forward_cell = LSTMCell()
        self.backward_cell = LSTMCell() if bidirectional else None
    
    def load_weights(self, forward_weights, backward_weights=None):
        self.forward_cell.load_weights(
            forward_weights[0], forward_weights[1], forward_weights[2]
        )

        if self.bidirectional and backward_weights:
            self.backward_cell.load_weights(
                backward_weights[0], backward_weights[1], backward_weights[2]
            )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, input_size = x.shape

        h_forward = np.zeros((batch_size, self.hidden_size))
        c_forward = np.zeros((batch_size, self.hidden_size))
        forward_output = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_forward, c_forward = self.forward_cell.forward(x_t, h_forward, c_forward)
            forward_output.append(h_forward)
        
        if self.bidirectional:
            h_backward = np.zeros((batch_size, self.hidden_size))
            c_backward = np.zeros((batch_size, self.hidden_size))
            backward_output = []

            for t in reversed(range(seq_len)):
                x_t = x[:, t, :]
                h_backward, c_backward = self.backward_cell.forward(x_t, h_backward, c_backward)
                backward_output.append(h_backward)

            backward_output.reverse()
        
        if self.return_sequences:
            if self.bidirectional:
                forward_stack = np.stack(forward_output, axis=1)  # (batch_size, seq_len, hidden_size)
                backward_stack = np.stack(backward_output, axis=1)
                return np.concatenate([forward_stack, backward_stack], axis=-1)
            else:
                return np.stack(forward_output, axis=1)  # (batch_size, seq_len, hidden_size)
        else:
            if self.bidirectional:
                return np.concatenate([h_forward, h_backward], axis=-1)
            else:
                return h_forward

class DropoutLayer:
    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, x, training=False):
        if training or self.rate == 0:
            return x
        mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
        return x * mask / (1 - self.rate)  # Scale to maintain expected value

class DenseLayer:
    def __init__(self, activation=None):
        self.activation = activation
        self.weights = None
        self.bias = None

    def load_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        output = np.dot(x, self.weights) + self.bias
        if self.activation == 'relu':
            return self.relu(output)
        elif self.activation == 'softmax':
            return self.softmax(output)
        else:
            return output  # No activation function applied

class LSTMModel:
    def __init__(self, num_units=64, num_layers=1, bidirectional=False):
        self.num_units = num_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_layer = EmbeddingLayer()
        self.lstm_layers = [
            LSTMLayer(
                hidden_size=num_units,
                return_sequences=(i < num_layers - 1), 
                bidirectional=bidirectional) 
            for i in range(num_layers)
        ]
        self.dropout_layer = DropoutLayer(rate=0.5)
        self.dense_layer = DenseLayer(
            activation='softmax'
        )

    def load_keras_weights(self, keras_model):
        weights = keras_model.get_weights()

        self.embedding_layer.load_weights(weights[0])

        if self.bidirectional:
            for i in range(len(self.lstm_layers)):
                forward_weights = weights[1 + i * 3: 1 + (i + 1) * 3]
                backward_weights = weights[1 + (len(self.lstm_layers) + i) * 3: 1 + (len(self.lstm_layers) + i + 1) * 3]
                self.lstm_layers[i].load_weights(forward_weights, backward_weights)
        else:
            for i in range(len(self.lstm_layers)):
                layer_weights = weights[1 + i * 3: 1 + (i + 1) * 3]
                self.lstm_layers[i].load_weights(layer_weights)

        self.dense_layer.load_weights(weights[-2], weights[-1])

    def predict(self, input_ids, training=False):
        x = self.embedding_layer.forward(input_ids)
        
        for lstm_layer in self.lstm_layers:
            x = lstm_layer.forward(x)
            if training:
                x = self.dropout_layer.forward(x, training)

        output = self.dense_layer.forward(x)
        return output