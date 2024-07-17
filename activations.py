import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1-np.tanh(z)**2
