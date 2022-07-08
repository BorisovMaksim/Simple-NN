import numpy as np
import copy
from utils import ReLU, sigmoid, gradients_to_vector, dictionary_to_vector, vector_to_dictionary

np.seterr(over='ignore')



class NeuralNetwork:
    def __init__(self):
        self.layers_dims = [64, 10, 5, 1]

    def initialize_parameters(self):
        W1 = np.random.randn(self.layers_dims[1], self.layers_dims[0]) * 0.01
        W2 = np.random.randn(self.layers_dims[2], self.layers_dims[1]) * 0.01
        W3 = np.random.randn(self.layers_dims[3], self.layers_dims[2]) * 0.01

        b1 = np.zeros((self.layers_dims[1], 1))
        b2 = np.zeros((self.layers_dims[2], 1))
        b3 = np.zeros((self.layers_dims[3], 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}

        return parameters

    def forward_pass(self, X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        Z1 = np.matmul(W1, X) + b1
        A1 = ReLU(Z1)
        Z2 = np.matmul(W2, A1) + b2
        A2 = ReLU(Z2)
        Z3 = np.matmul(W3, A2) + b3
        A3 = sigmoid(Z3)

        cache = {"Z1": Z1, "A1": A1,
                 "Z2": Z2, "A2": A2,
                 "Z3": Z3, "A3": A3}
        return A3, cache

    def backward_pass(self, cache, parameters, X, Y):
        Z3 = cache['Z3']
        A3 = cache['A3']
        Z2 = cache['Z2']
        A2 = cache['A2']
        Z1 = cache['Z1']
        A1 = cache['A1']

        W1 = parameters['W1']
        W2 = parameters['W2']
        W3 = parameters['W3']
        b1 = parameters['b1']
        b2 = parameters['b2']
        b3 = parameters['b3']

        m = X.shape[1]

        dZ3 = A3 - Y
        dW3 = 1 / m * np.matmul(dZ3, A2.T)
        db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

        dZ2 = np.heaviside(Z2, 0) * (np.matmul(W3.T, dZ3))
        dW2 = 1 / m * np.matmul(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.heaviside(Z1, 0) * np.matmul(W2.T, dZ2)
        dW1 = 1 / m * np.matmul(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        assert dW3.shape == W3.shape
        assert dW2.shape == W2.shape
        assert dW1.shape == W1.shape
        assert db3.shape == b3.shape
        assert db2.shape == b2.shape
        assert db1.shape == b1.shape

        gradients = {
            "dW3": dW3, "db3": db3,
            "dW2": dW2, "db2": db2,
            "dW1": dW1, "db1": db1
        }
        return gradients

    def update_parameters(self, parameters, gradients, learning_rate):
        W1 = copy.deepcopy(parameters['W1'])
        b1 = copy.deepcopy(parameters['b1'])
        W2 = copy.deepcopy(parameters['W2'])
        b2 = copy.deepcopy(parameters['b2'])
        W3 = copy.deepcopy(parameters['W3'])
        b3 = copy.deepcopy(parameters['b3'])

        dW1 = gradients['dW1']
        db1 = gradients['db1']
        dW2 = gradients['dW2']
        db2 = gradients['db2']
        dW3 = gradients['dW3']
        db3 = gradients['db3']

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W3 = W3 - learning_rate * dW3
        b3 = b3 - learning_rate * db3

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        return parameters

    def compute_loss(self, Y, A3):
        m = Y.shape[1]
        return -1 / m * (np.matmul(Y, np.log(A3).T) + np.matmul((1 - Y), np.log(1 - A3).T))[0][0]

    def train(self, num_iterations, X_train, y_train, X_test, y_test, print_loss=True, learning_rate=0.01):
        parameters = self.initialize_parameters()
        losses_train = []
        losses_test = []
        for i in range(num_iterations):
            A3, cache = self.forward_pass(X_train, parameters)
            gradients = self.backward_pass(cache, parameters, X_train, y_train)
            parameters = self.update_parameters(parameters, gradients, learning_rate=learning_rate)
            loss_train = self.compute_loss(y_train, A3)
            loss_test = self.compute_loss(y_test, self.forward_pass(X_test, parameters)[0])
            losses_train.append(loss_train)
            losses_test.append(loss_test)
            if print_loss and i % 500 == 0:
                print(f"loss_train after {i + 1} iterations = {round(loss_train, 4)}")
        return parameters, losses_train, losses_test

    def predict(self, parameters, x):
        return (self.forward_pass(x, parameters)[0] < 0.5) + 0

    def gradient_check(self, parameters, gradients, X, Y, epsilon=1e-7, print_msg=True):
        parameters_values, _ = dictionary_to_vector(parameters)
        grad = gradients_to_vector(gradients)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        for i in range(num_parameters):
            theta_plus = np.copy(parameters_values)
            theta_plus[i] = theta_plus[i] + epsilon
            out, cache = self.forward_pass(X, vector_to_dictionary(theta_plus))
            J_plus[i] = self.compute_loss(Y, out)

            theta_minus = np.copy(parameters_values)
            theta_minus[i] = theta_minus[i] - epsilon
            out, cache = self.forward_pass(X, vector_to_dictionary(theta_minus))
            J_minus[i] = self.compute_loss(Y, out)

            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator

        if print_msg:
            if difference > 2e-7:
                print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                    difference) + "\033[0m")
            else:
                print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                    difference) + "\033[0m")

        return difference