import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split



def ReLU(z):
    return z * (z > 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dictionary_to_vector(parameters):
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys


def vector_to_dictionary(theta):
    parameters = {}
    parameters["W1"] = theta[: 640].reshape((10, 64))
    parameters["b1"] = theta[640: 650].reshape((10, 1))
    parameters["W2"] = theta[650: 700].reshape((5, 10))
    parameters["b2"] = theta[700: 705].reshape((5, 1))
    parameters["W3"] = theta[705: 710].reshape((1, 5))
    parameters["b3"] = theta[710: 711].reshape((1, 1))

    return parameters


def gradients_to_vector(gradients):
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def load_dataset():
    digits = load_digits()
    data = pd.DataFrame(digits.data)
    target = digits['target'].reshape(-1, 1)
    target = (target == 5) + 0
    return data, target



def train_dev_split(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.15, random_state=42)
    X_train = X_train.values.reshape(64, -1)
    X_test = X_test.values.reshape(64, -1)
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)
    return  X_train, y_train, X_test, y_test


def plot_loss(losses_train, losses_test):
    plt.plot(losses_train, label="loss_train")
    plt.plot(losses_test, label="loss_test")
    plt.xlabel('num_iter')
    plt.ylabel('Log loss')
    plt.legend()
    plt.savefig("./images/simple_DNN_loss_plot.png")



