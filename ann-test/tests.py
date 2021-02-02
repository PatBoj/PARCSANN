import importlib
import numpy as np
import matplotlib.pyplot as plt
ann = importlib.import_module("ann-test")

def testActivationFunctions():
    ann.n = ann.NeuralNetwork(5)
    x = np.arange(-10, 10, 0.01)

    # sigmoid
    y = ann.n.sigmoid(x)
    plt.title("sigmoid")
    plt.plot(x, y)
    plt.show()

    # derivative sigmoid
    y = ann.n.sigmoidDerivative(x)
    plt.title("derivative sigmoid")
    plt.plot(x, y)
    plt.show()

    # ReLU
    y = ann.n.ReLU(x)
    plt.title("ReLU")
    plt.plot(x, y)
    plt.show()

    # derivative ReLU
    y = ann.n.ReLUDerivative(x)
    plt.title("derivative ReLU")
    plt.plot(x, y)
    plt.show()

    # tanh
    y = ann.n.tanh(x)
    plt.title("tanh")
    plt.plot(x, y)
    plt.show()

    # derivative tanh
    y = ann.n.tanhDerivative(x)
    plt.title("derivative tanh")
    plt.plot(x, y)
    plt.show()

testActivationFunctions()