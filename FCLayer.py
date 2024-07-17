from layer import Layer
from activations import *

import numpy as np

class FCLayer(Layer):
    def __init__(self, input_size, output_size, activation = "sigmoid"):
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.bias = np.random.rand(output_size, 1) - 0.5
        self.activation = activation


    def forward_propagation(self, input_data):
        self.input = input_data
        self.weighted_sum = np.dot(self.weights, self.input)+self.bias
        self.output = []
        if self.activation == "sigmoid":
            for i in self.weighted_sum:
                self.output.append(sigmoid(i))
            self.output = np.array(self.output)
        elif self.activation == "tanh":
            for i in self.weighted_sum:
                self.output.append(tanh(i))
            self.output = np.array(self.output)
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        self.activation_gradient = []
        if self.activation == "sigmoid":
            for i in self.weighted_sum:
                self.activation_gradient.append(sigmoid_prime(i))
            self.activation_gradient = np.array(self.activation_gradient)
        if self.activation == "tanh":
            for i in self.weighted_sum:
                self.activation_gradient.append(tanh_prime(i))
            self.activation_gradient = np.array(self.activation_gradient)

        error_gradient = self.activation_gradient * output_gradient    
        weights_gradient = (np.dot(self.input, error_gradient.T)).T
        bias_gradient = error_gradient
        input_gradient = np.dot(self.weights.T, error_gradient)

        '''Gradient desent'''
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

    



'''ll = FCLayer(3, 2)

print(ll.weights)
print(ll.bias)
print(ll.forward_propagation([
    [3],
    [1],
    [2]
]))

print(ll.weighted_sum)
err = np.array([[0.1],[0.4]])
print(ll.backward_propagation(err, 0.1))
'''
