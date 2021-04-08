import numpy as np
from scipy.spatial import distance_matrix

class FireFlies:
    def __init__(self, gamma, beta, alpha, dimensions, particles, iterations, function, boundary):
        # Seed the random number generator
        np.random.seed(1)
        
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.dim = dimensions
        self.n = particles
        self.N = iterations
        self.function = function
        self.boundary = boundary

        self.norm = np.linalg.norm(self.boundary[1] - self.boundary[0]) / 2

        # create swarm as a matrix with n rows (n fireflies)
        # and dim columns that represent each parameter
        self.swarm = np.zeros((self.n, self.dim))

        # each intensity is releted to the given function at firefly position
        self.cost = np.zeros(self.n)
        self.intensity = np.zeros(self.n)

        # distance matrix
        self.distances = np.zeros((self.n, self.n))

        self.initialize()

        print("Parameters:\n", self.swarm)
        print("Average error:\n", self.cost)
        self.run()

    def initialize(self):
        # set random parameteres to all fireflies
        for i in range(self.n):
            self.swarm[i] = np.random.uniform(self.boundary[0], self.boundary[1])

        # sets intensity as the invert of the error function (because I want to find
        # minimum, not maximum, error)
        self.cost = self.function(self.swarm)
        self.intensity = 1/self.cost

        # computes distances between every firefly
        self.distances = distance_matrix(self.swarm, self.swarm)

    def moveOneFirefly(self, chosenOne):
        #light = self.intensity * np.exp(-self.gamma * self.distances[:, chosenOne] ** 2)
        light = self.intensity
        maxIndex = np.argmax(light)

        if(maxIndex != chosenOne):
            self.swarm[chosenOne, :] += self.beta * np.exp(-self.gamma * self.distances[chosenOne, maxIndex] ** 2) * (self.swarm[maxIndex, :]) - self.swarm[chosenOne, :]
        self.swarm[chosenOne] += self.alpha * np.random.uniform(-self.boundary[1]/2, self.boundary[1]/2)

        for i in range(len(self.swarm[chosenOne, :])):
            if(self.swarm[chosenOne, i] < self.boundary[0][i]):
                self.swarm[chosenOne, i] = self.boundary[0][i]
            if(self.swarm[chosenOne, i] > self.boundary[1][i]):
                self.swarm[chosenOne, i] = self.boundary[1][i]
        
        # update intensity and distance
        self.cost[chosenOne] = self.function(np.reshape(self.swarm[chosenOne], (1, self.dim)))
        self.intensity[chosenOne] = 1/self.cost[chosenOne]
        self.distances[chosenOne, :] = distance_matrix(np.reshape(self.swarm[chosenOne], (1, self.dim)), self.swarm) / self.norm
        self.distances[:, chosenOne] = self.distances[chosenOne, :]

    def oneEpoc(self):
        order = np.random.permutation(self.n)
        for i in range(self.n):
            self.moveOneFirefly(order[i])

    def run(self):
        for _ in range(self.N):
            self.oneEpoc()
            print("Parameters:\n", self.swarm)
            print("Average error:\n", self.cost)
