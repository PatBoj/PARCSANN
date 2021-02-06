import numpy as np

class NeuralNetwork:
    def __init__(self,n):
        # Seed the random number generator
        np.random.seed(1)
        # Set synaptic weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.rand((n, 1)) - 1

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.think(training_inputs)

            # Calculate the error rate
            error = training_outputs - output


            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def think(self, inputs):
        """
        Pass inputs through the neural network to get output
        """
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def read_inputs(self, directory, numb):
        data_in = []
        for i in range (1, numb+1):
            buff = []
            file = open(directory+"/"+str(i)+".txt", 'r')
            for line in file:
                buff.append(float(line))
            data_in.append(buff)
        return data_in

    def read_outputs(self, file):
        data_out = []
        file = open(file, 'r')
        data_out = []
        for line in file:
            data_out.append(float(line)/100000)
        return [data_out]

if __name__ == "__main__":
    # Initialize the single neuron neural network
    neural_network = NeuralNetwork(56)

    data_in = neural_network.read_inputs("patterns",10)
    data_out = neural_network.read_outputs("patterns/rho.txt")

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set, with 4 examples consisting of 3
    # input values and 1 output value
    training_inputs = np.array(data_in)
    print(training_inputs)
    training_outputs = np.array(data_out).T
    print(training_outputs)
    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 1000000)

    test = [5, 9, 5, 2, 4, 9, 9, 9, 1, 7, 4, 2, 4, 1, 7, 5, 9, 6, 2, 4, 5, 9, 2, 4, 5, 1, 9, 9, 1, 6, 2, 8, 7, 5, 2, 3, 8, 5, 4, 3, 8, 2, 2, 4, 2, 1, 2, 6, 2, 1, 6, 1, 9, 7, 3, 9]
    test_out = [8704.813447415798]

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    print(neural_network.think(np.array(test)))

