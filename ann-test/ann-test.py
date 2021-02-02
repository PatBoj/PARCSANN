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
        self.synaptic_weights = 2 * np.random.random_sample((n, 1)) - 1

    def sigmoid(self, x):
        """
        Sigmoid function s(x) = 1 / (1 + e^(-x))

        Parameters
        ----------
        x : int
            Argument of the function
        
        Returns
        -------
        int
            Value of the sigmoid function in given point
        """

        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        """
        Derivative of a sigmoid function s'(x) = s(x) [1 - s(x)]

        Parameters
        ----------
        x : int
            Argument of the function
        
        Returns
        -------
        int
            Derivative of a sigmoid funtion in a given point
        """

        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def ReLU(self, x):
        """
        Rectified Linear Unit (ReLU) function
        ReLU(x) = max{0, x}

        Parameters
        ----------
        x : int
            Argument of the function

        Returns
        -------
        int
            Value of the ReLU function in given point
        """

        return np.maximum(0, x)

    def ReLUDerivative(self, x):
        """
        Derivative of the ReLU function

        Parameters
        ----------
        x : int
            Argument of the function
        
        Returns
        -------
        int
            Value of the ReLU function derivative in given point
        """

        return np.heaviside(x, 0)

    def tanh(self, x):
        """
        Hyperbolic tangent function

        Parameters
        ----------
        x : int
            Argument of the function
        
        Returns
        -------
        int
            Value of the hyperbolic tangent function derivative in given point
        """

        return np.tanh(x)

    def tanhDerivative(self, x):
        """
        Hyperbolic tangent derivative function

        Parameters
        ----------
        x : int
            Argument of the function
        
        Returns
        -------
        int
            Derivative value of the hyperbolic tangent function derivative in given point
        """

        return 1 / np.cosh(x)**2

    def costMSE(self, desireOutput, output):
        """
        Cost function which is sum of squares of differences between desire output
        and output recived from the network (mean squared error)

        Parameters
        ----------
        desireOutput : array_like
            Desire output of the neural network
        output : array_like
            Output given by neural network
        
        Returns
        -------
        int
            Mean squared error between two vectors
        """

        return (desireOutput - output) ** 2

    def costMSEDerivative(self, desireOutput, output):
        """
        Derivative of the cost function

        Parameters
        ----------
        desireOutput : array_like
            Desire output of the neural network
        output : array_like
            Output given by neural network
        
        Returns
        -------
        int
            Derivative of the mean squared error between two vectors
        """

        return 2 * (desireOutput - output)

    def train(self, inputs, desireOutputs):
        for(i in range(20)):
            outpit
