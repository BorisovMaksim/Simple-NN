from NN import NeuralNetwork
from utils import load_dataset, train_dev_split, plot_loss






def main():
    data, target = load_dataset()
    X_train, y_train, X_test, y_test = train_dev_split(data, target)
    NN = NeuralNetwork()
    parameters, losses_train, losses_test = NN.train(num_iterations=2000, X_train=X_train, y_train=y_train, X_test=X_test,
                                         y_test=y_test, print_loss=False, learning_rate=0.01)

    out, cache = NN.forward_pass(X_train, parameters)
    gradients = NN.backward_pass(cache, parameters, X_train, y_train)
    NN.gradient_check(parameters, gradients, X_train, y_train, 1e-7, True)

    plot_loss(losses_train, losses_test)

    return 1


if __name__ == '__main__':
    main()
