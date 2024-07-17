from cmath import cosh
import numpy as np
from activations import *
from FCLayer import *
from layer import *
from lost_functions import *

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, cost_function, cost_function_prime):
        self.loss = cost_function
        self.loss_prime = cost_function_prime
    
    def predict(self, input_data):
        results = []
        for i in range(len(input_data)):
            output = np.expand_dims(input_data[i], axis = -1)
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(output)
        
        return results

    def fit(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            err = 0

            for j in range(len(x_train)):
                output = np.expand_dims(x_train[j], axis = -1)
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                err += self.loss(y_train[j], np.squeeze(output, -1))

                error = np.expand_dims(self.loss_prime(y_train[j], np.squeeze(output, -1)), -1)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= len(x_train)
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

                


