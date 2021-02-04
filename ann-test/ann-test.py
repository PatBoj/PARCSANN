import numpy as np

class NeuralNetwork:
    """
    A class to represent an multi-layer perceptron 
    (the simplest artificial neural network)

    ...

    Attributes
    ----------
    layers : array_like
        Number of neurons in every layer

    activationFunction: string, optional
        Type of activation function: sigmoid, ReLU or hyperbolic tangent (tanh)
    
    """
    def __init__(self, layers, activationFunction = "sigmoid"):
        # Seed the random number generator
        np.random.seed(1)
        
        self.layers = layers
        self.activationFunction = activationFunction

        self.weights = []
        self.biases = []

        self.setRandomWeightsAndBiases()

    def setRandomWeightsAndBiases(self):
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.rand(self.layers[i+1], self.layers[i]) * 2 - 1)
            self.biases.append(np.random.rand(self.layers[i+1], 1) * 2 - 1)
    
    def getActivationFunction(self, functionName, x):
        if(functionName == "sigmoid"):
            return self.sigmoid(x)
        elif(functionName == "ReLU"):
            return self.ReLU(x)
        elif(functionName == "tanh"):
            return self.tanh(x)
        else:
            print("Given activation function does not exist. Sigmoid function was used")
            return self.sigmoid(x)

    def getActivationFunctionDerivative(self, functionName, x):
        if(functionName == "sigmoid"):
            return self.sigmoidDerivative(x)
        elif(functionName == "ReLU"):
            return self.ReLUDerivative(x)
        elif(functionName == "tanh"):
            return self.tanhDerivative(x)
        else:
            print("Given activation function does not exist. Sigmoid function was used")
            return self.sigmoidDerivative(x)

    def feedforward(self, input):
        # „z” is a set of vectors which contain value of every neuron BEFORE 
        # activation function was applied, also len(z) is equal to 
        z = []

        # „z” is a set of vectors which contain value of every neuron AFTER
        # activation function was applied
        a = [np.copy(input)]

        # for every layer „i”
        for i in range(len(self.layers - 1)):
            # first, compute „raw” neuron values in i-th layer
            # z = W * a + b, where „z” is from layer „L” and a from layer „L-1”
            z.append(self.weights[i].dot(a[-1]) + self.biases[i])

            # then apply activation function to the previos outputs
            a.append(self.getActivationFunction(self.activationFunction, z[-1]))
        
        return z, a

    def backpropagation(self, desireOutput, z, a):
        # set of derivatives of „C” (cost function) with respect to „W” (weights)
        # in every layer
        dW = []

        # set of derivatives of „C” (cost function) with respect to „b” (biases)
        # in every layer
        db = []

        # Initialize layer's error sets
        layersErrors = [None] * len(self.layers - 1)

        # last layer error is just derivative of the cost function with respect to
        # received values from neural network times derivative of activation function
        # in respect to the „raw” value of the neuron
        layersErrors[-1] = self.costMSEDerivative(desireOutput, a[-1]) 
        * self.getActivationFunctionDerivative(self.getActivationFunction, z[-1]) 

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
        return 0

