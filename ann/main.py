import importlib
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
ann = importlib.import_module("ann")
pso = importlib.import_module("firefly")
import os
import pyswarms as ps

np.random.seed(1)

# set working directory as the one where this file is
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load and suffle data
data = genfromtxt("../patterns/data_7000_rho_in_rho_out.csv", delimiter=';', skip_header=1)
np.random.shuffle(data)

nTesting = 1000
testingInputs = data[:nTesting, :-1].T
testingOutputs = data[:nTesting, [-1]].T
inputs = data[nTesting:, :-1].T
outputs = data[nTesting:, [-1]].T

def toInt(number):
    return int(round(number, 0))

def neuralNetworkError(layers):
    #printLayer = np.zeros(shape = np.shape(layers))
    #for i in range(np.shape(printLayer)[0]):
    #    for j in range(np.shape(printLayer)[1]):
    #        printLayer[i, j] = int(round(layers[i, j], 0))
    #print(printLayer)

    error = np.ones(np.shape(layers)[0])
    for i in range(np.shape(layers)[0]):
        n = ann.NeuralNetwork([len(inputs), int(round(layers[i,0], 0)), int(round(layers[i,1], 0)), len(outputs)])
        n.train(inputs, outputs, batchSize = 1, epochs = 100, eta = 0.5, testInputs = testingInputs, testOutputs = testingOutputs)
        error[i] = n.error(testingInputs, testingOutputs)
    return error

def annError(parameters):
    batchSize = parameters[:, 0]
    epochs = parameters[:, 1]
    eta = parameters[:, 2]
    layers = parameters[:, 3:]

    error = np.ones(np.shape(parameters)[0])

    for i in range(np.shape(layers)[0]):
        layer = np.zeros(len(layers[i]))
        for j in range(len(layer)):
            layer[j] = toInt(layers[i, j])
        layer = np.concatenate((len(inputs), np.trim_zeros(layer), len(outputs)), axis=None)
        layer = layer.astype(int)

        n = ann.NeuralNetwork(layer)
        n.train(inputs, outputs, batchSize = toInt(batchSize[i]), epochs = toInt(epochs[i]), eta = eta[i], testInputs = testingInputs, testOutputs = testingOutputs)
        error[i] = n.error(testingInputs, testingOutputs)
    return error

nParameteres = 10

print(neuralNetworkError(np.array([[100, 100], [10, 10]])))


min_bound = np.zeros(nParameteres)
min_bound[0] = 1
min_bound[1] = 1
max_bound = min_bound + 50
max_bound[2] = 10
bounds = (min_bound, max_bound)


min_bound = np.ones(nParameteres)
max_bound = min_bound + 99
#min_bound = np.append(min_bound, 1)
#max_bound = np.append(max_bound, 2)
bounds = (min_bound, max_bound)


'''
firefly = pso.FireFlies(gamma = 1, beta = 1, alpha = 0.1, dimensions = nParameteres, particles = 20, 
    iterations = 15, function = annError, boundary = bounds)
'''

'''
# Create bounds
min_bound = np.ones(3)
max_bound = min_bound + 99
bounds = (min_bound, max_bound)

options = {'c1': 1, 'c2': 1, 'w': 1}
optimizer = ps.single.GlobalBestPSO(n_particles = 10, dimensions = 3, options = options, bounds = bounds)
cost, pos = optimizer.optimize(neuralNetworkError, iters = 10)
'''
'''
n = ann.NeuralNetwork([len(inputs), 0, 1, 0, len(outputs)])
n.train(inputs, outputs, batchSize = 10, epochs = 100, eta = 0.001, testInputs = testingInputs, testOutputs = testingOutputs)
print(n.error(testingInputs, testingOutputs))
'''

# Saving all errors to file (not average)
#
'''
n = ann.NeuralNetwork([len(inputs), 14, 7, len(outputs)])
error = n.train(inputs, outputs, batchSize = 1, epochs = 100, eta = 0.5, testInputs = testingInputs, testOutputs = testingOutputs)
np.savetxt("error.csv", error, delimiter = ",", fmt = "%.10f")
_, a = n.feedforward(testingInputs)

np.savetxt("prediction.csv", np.stack((a[-1].flatten(), testingOutputs.flatten()), axis=-1), delimiter = ",", fmt = "%.10f")
'''