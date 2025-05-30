import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from scipy.signal import correlate2d
from sklearn.metrics import f1_score
from tensorflow.keras.models import load_model
class Conv2DLayer:
    def __init__(self, weights, bias, stride=1, padding='same'):
        self.weights = weights
        self.bias = bias
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        batch, h, w, c = x.shape
        kh, kw, in_c, out_c = self.weights.shape
        pad_h = kh // 2 if self.padding == 'same' else 0
        pad_w = kw // 2 if self.padding == 'same' else 0
        x_padded = np.pad(x, ((0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')
        out_h = (h + 2 * pad_h - kh) // self.stride + 1
        out_w = (w + 2 * pad_w - kw) // self.stride + 1
        out = np.zeros((batch, out_h, out_w, out_c))
        for i in range(out_h):
            for j in range(out_w):
                region = x_padded[:, i*self.stride:i*self.stride+kh, j*self.stride:j*self.stride+kw, :]
                for k in range(out_c):
                    out[:, i, j, k] = np.sum(region * self.weights[..., k], axis=(1,2,3)) + self.bias[k]
        return np.maximum(out, 0)

class PoolingLayer:
    def __init__(self, size=2, stride=2, mode='max'):
        self.size = size
        self.stride = stride
        self.mode = mode

    def forward(self, x):
        batch, h, w, c = x.shape
        out_h = (h - self.size) // self.stride + 1
        out_w = (w - self.size) // self.stride + 1
        out = np.zeros((batch, out_h, out_w, c))
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, i*self.stride:i*self.stride+self.size, j*self.stride:j*self.stride+self.size, :]
                if self.mode == 'max':
                    out[:, i, j, :] = np.max(region, axis=(1,2))
                else:
                    out[:, i, j, :] = np.mean(region, axis=(1,2))
        return out

class FlattenLayer:
    def forward(self, x):
        return x.reshape((x.shape[0], -1))

class DenseLayer:
    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, x):
        out = np.dot(x, self.weights) + self.bias
        if self.activation == 'relu':
            return np.maximum(0, out)
        elif self.activation == 'softmax':
            exp = np.exp(out - np.max(out, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)
        return out

# === FORWARD PROP CNN CLASS ===

class CNNScratch:
    def __init__(self, keras_model_path, pooling='max'):
        self.layers = []
        self.load_weights_from_h5(keras_model_path, pooling)

    def load_weights_from_h5(self, h5_path, pooling):
      with h5py.File(h5_path, 'r') as f:
        weights_group = f['model_weights']
        for lname in weights_group:
            g = weights_group[lname]
            if 'kernel:0' in g and 'bias:0' in g:
                w = np.array(g['kernel:0'])
                b = np.array(g['bias:0'])

                if len(w.shape) == 4:
                    self.layers.append(Conv2DLayer(w, b))
                elif len(w.shape) == 2:
                    activation = 'softmax' if w.shape[1] == 10 else 'relu'
                    self.layers.append(DenseLayer(w, b, activation=activation))
            else:
                if 'pool' in lname.lower():
                    self.layers.append(PoolingLayer(mode=pooling))
                elif 'flatten' in lname.lower():
                    self.layers.append(FlattenLayer())
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x, axis=1)

def evaluate_scratch_model(weight_file, pooling='max'):
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.flatten()
    model = CNNScratch(weight_file, pooling=pooling)
    y_pred = model.predict(x_test[:1000])
    return f1_score(y_test[:1000], y_pred, average='macro')