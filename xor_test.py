import numpy as np

from network import *
from FCLayer import *
from activations import *
from lost_functions import *

# training data
x_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = np.array([
    [0],
    [1],
    [1],
    [0]
])


# network
net = Network()
net.add(FCLayer(2, 3, activation="tanh"))
net.add(FCLayer(3, 1, activation="tanh"))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
