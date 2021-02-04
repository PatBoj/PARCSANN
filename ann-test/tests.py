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

def testCostFunctions():
    ann.n = ann.NeuralNetwork(5)
    output = np.arange(-10, 10, 0.01)
    desireOutput = np.full(2000, 0)

    # mean squared error
    plt.title("mse")
    plt.plot(output, ann.n.costMSE(desireOutput, output))
    plt.show()

    # mean squared error derivative
    plt.title("mse derivative")
    plt.plot(output, ann.n.costMSEDerivative(desireOutput, output))
    plt.show()

    # single point mean squared error
    print("MSE single point test:")
    print("\tIt's working :)\n") if ann.n.costMSE(0, 1) == 1 else print("\tIt's not working :(\n")

    # single point mean squared error derivative
    print("MES derivative single point test:")
    print("\tIt's working :)\n") if ann.n.costMSEDerivative(0, 1) == -2 else print("\tIt's not working :(\n")

def testing():
    weights = []
    for i in range(5):
        weights.append(np.random.rand(5, 5) * 2 - 1)
    for i in range(5):
        print(weights[i])

testing()