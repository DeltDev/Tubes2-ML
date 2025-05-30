import numpy as np
import importlib

class SimpleRNN:
    def __init__(self, num_layers=1, bidirectional=False):
        self.embedding_weights = None
        self.rnn_weights = []  # List of dicts, one per layer
        self.dense_weights = None
        self.dense_bias = None
        self.vocab_size = None
        self.embedding_dim = None
        self.rnn_units = None
        self.num_classes = None
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def load_keras_weights(self, keras_model):
        weights = keras_model.get_weights()
        layer_idx = 0

        # Embedding
        self.embedding_weights = weights[layer_idx]
        layer_idx += 1

        # RNN layers
        self.rnn_weights = []
        for layer in range(self.num_layers):
            layer_weights = {}
            for direction in ['forward', 'backward'] if self.bidirectional else ['forward']:
                layer_weights[direction] = {
                    'input_weights': weights[layer_idx],
                    'recurrent_weights': weights[layer_idx + 1],
                    'bias': weights[layer_idx + 2]
                }
                layer_idx += 3
            self.rnn_weights.append(layer_weights)

        # Dense layer
        self.dense_weights = weights[layer_idx]
        self.dense_bias = weights[layer_idx + 1]

        self.vocab_size, self.embedding_dim = self.embedding_weights.shape
        self.rnn_units = self.rnn_weights[0]['forward']['input_weights'].shape[1]
        self.num_classes = self.dense_weights.shape[1]

        print(f"Loaded weights - Vocab: {self.vocab_size}, Embedding: {self.embedding_dim}, "
              f"RNN units: {self.rnn_units}, Classes: {self.num_classes}, "
              f"Layers: {self.num_layers}, Bidirectional: {self.bidirectional}")

    def embedding_forward(self, input_ids):
        return self.embedding_weights[input_ids]

    def single_rnn_pass(self, x, W_in, W_rec, bias, reverse=False):
        batch_size, seq_len, _ = x.shape
        h = np.zeros((batch_size, self.rnn_units))
        output_seq = []
        time_steps = range(seq_len) if not reverse else reversed(range(seq_len))
        for t in time_steps:
            x_t = x[:, t, :]
            h = np.tanh(np.dot(x_t, W_in) + np.dot(h, W_rec) + bias)
            output_seq.append(h)
        if reverse:
            output_seq = output_seq[::-1]
        return np.stack(output_seq, axis=1), h  # Return full sequence and last hidden

    def rnn_forward(self, embedded_input):
        x = embedded_input
        for layer in range(self.num_layers):
            layer_weights = self.rnn_weights[layer]
            if self.bidirectional:
                fw_out, fw_last = self.single_rnn_pass(
                    x,
                    layer_weights['forward']['input_weights'],
                    layer_weights['forward']['recurrent_weights'],
                    layer_weights['forward']['bias'],
                    reverse=False
                )
                bw_out, bw_last = self.single_rnn_pass(
                    x,
                    layer_weights['backward']['input_weights'],
                    layer_weights['backward']['recurrent_weights'],
                    layer_weights['backward']['bias'],
                    reverse=True
                )
                # Concatenate outputs
                x = np.concatenate([fw_out, bw_out], axis=-1)
                last_hidden = np.concatenate([fw_last, bw_last], axis=-1)
            else:
                x, last_hidden = self.single_rnn_pass(
                    x,
                    layer_weights['forward']['input_weights'],
                    layer_weights['forward']['recurrent_weights'],
                    layer_weights['forward']['bias'],
                    reverse=False
                )
        return last_hidden

    def dense_forward(self, rnn_output):
        logits = np.dot(rnn_output, self.dense_weights) + self.dense_bias
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, input_ids):
        embedded = self.embedding_forward(input_ids)
        rnn_output = self.rnn_forward(embedded)
        return self.dense_forward(rnn_output)

    def predict(self, input_ids):
        return np.argmax(self.forward(input_ids), axis=1)
