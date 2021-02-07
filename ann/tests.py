import importlib
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
ann = importlib.import_module("ann")
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

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

def feedforwardTest():
    n = ann.NeuralNetwork([3, 5, 10, 2], "ReLU")
    n.setWeightOnes()

    x = np.array([[1], [1], [1]])

    print("NEW TEST THIS IS")
    
    z, a = n.feedforward(x)
    for i in range(len(a)):
        print(a[i], "\n")

    for i in range(len(z)):
        print(z[i], "\n")

def dataSetTest():
    np.random.seed(1)

    """
    AREA	
    PERIMETER	
    MAJORAXIS	
    MINORAXIS	
    ECCENTRICITY	
    CONVEX_AREA	
    EXTENT	
    CLASS (Cammeo = 10, Osmancik = 01)
    """

    data = genfromtxt("../patterns/data.csv", delimiter=';')
    np.random.shuffle(data)

    #inputs = data[:, :-2].T
    #outputs = data[:, [-2, -1]].T

    nTesting = 1000
    testingInputs = data[:nTesting, :-1].T
    testingOutputs = data[:nTesting, [-1]].T
    inputs = data[nTesting:, :-1].T
    outputs = data[nTesting:, [-1]].T

    """
    print("DATA: ", data.shape, "\n", data, "\n")
    print("INPUT: ", inputs.shape, "\n", inputs, "\n")
    print("OUTPUT: ", outputs.shape, "\n", outputs, "\n")
    print("MODIFY INPUT: ", inputs[:, 0:10].shape, "\n", inputs[:, 0:10], "\n")
    print("MODIFY INPUT SEQUEL: ", inputs[:, 10:20].shape, "\n", inputs[:, 10:20], "\n")
    print("MODIFY INPUT OUTPUT: ", outputs[:, 0:10].shape, "\n", outputs[:, 0:10], "\n")
    """

    # 3810 observations, factors: 1, 2, 3, 5, 6, 10, 15, 30, 127, 254, 381, 635, 
    #                             762, 1270, 1905, 3810,

    #n = ann.NeuralNetwork([len(inputs), 100, 100, 100, len(outputs)])
    #n.train(inputs, outputs, batchSize = 381, epochs = 100, eta = 0.1)
    """
    n = ann.NeuralNetwork([len(inputs), 100, 100, 100, len(outputs)])
    n.train(inputs, outputs, batchSize = 381, epochs = 1000, eta = .2)
    t = np.array([6, 1, 7, 2, 312, 3463, 123, 157, 457])
    z, a = n.feedforward(inputs[:, t])
    
    print("\nNetwork gives: \n", a[-1].T, ", \nbut it should give: \n", outputs[:, t], sep="")
    """
    n = ann.NeuralNetwork([len(inputs), 10, 12, 10, len(outputs)])
    error = n.train(inputs, outputs, batchSize = 10, epochs = 100, eta = 0.001, testInputs = testingInputs, testOutputs = testingOutputs)
    np.savetxt("error.csv", error, delimiter = ",", fmt = "%.10f")
    #_, a = n.feedforward(testingInputs)
    #print(np.round(a[-1]))
    #print(testingOutputs)
    #print((1 - np.sum(np.abs(np.round(a[-1]) - testingOutputs)) / nTesting) * 100, "%", sep="")

dataSetTest()