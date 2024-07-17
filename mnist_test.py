import numpy as np

from network import *
from FCLayer import *
from activations import *
from lost_functions import *

from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = np_utils.to_categorical(y_train)


x_test = x_test.reshape(x_test.shape[0], 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

net = Network()
net.add(FCLayer(28*28, 100))
net.add(FCLayer(100, 50))
net.add(FCLayer(50, 10))

net.use(mse, mse_prime)
net.fit(x_train[:1000], y_train[:1000], epochs=30, learning_rate=1)

test_no = 100
out = np.squeeze(net.predict(x_test[:test_no]))
out_true = y_test[:test_no]
correct = 0
for i, j in zip(out, out_true):
    predicted_number = np.argmax(i)
    predicted_probability = i[predicted_number]
    true_value = np.argmax(j)
    if predicted_number == true_value:
        correct += 1
        print("Predicted value : %s || Confidence : %.2f || True value : %s || True" % (predicted_number, predicted_probability, true_value))
    else:
        print("Predicted value : %s || Confidence : %.2f || True value : %s ||" % (predicted_number, predicted_probability, true_value))

print("||====================||")
print("||  Accuracy : %.2f  ||" % ((correct/test_no)*100))
print("||====================||")
