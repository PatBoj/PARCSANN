import numpy as np

class NeuralNetwork:
    """
    A class to represent an multi-layer perceptron 
    (the simplest artificial neural network)

    ...

    Attributes
    ----------
    nLayers : int
        number of hidden layers
    
    """
    def __init__(self, n):
        # Seed the random number generator
        np.random.seed(1)

        # Set synaptic weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((n, 1)) - 1

help(NeuralNetwork)