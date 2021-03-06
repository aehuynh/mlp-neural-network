import numpy as np
from toolbox import get_activation, get_activation_prime, get_cost_func

NN_settings = {
    "learn_rate" : 0.001,
    # L2 regularization parameter
    "weight_decay" : 0,
    # (number of neurons in layer, activation function of layer)
    # The first element is the input layer with no activation function
    "layers" : [ (2, None) , (7, "tanh"), (2, "softmax") ],
    # Name of cost function
    "cost_type" : "cross entropy loss",
}

class Layer:
    """Neural network layer"""

    def __init__(self, num_in, num_nodes, learn_rate, a_type="tanh", weight_decay=0):
        self.activate = get_activation(a_type)
        self.activate_p = get_activation_prime(a_type)
        self.W = np.random.randn(num_in, num_nodes) / np.sqrt(num_in)
        self.b = np.zeros((1, num_nodes))
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay

    def transform(self, o):
        z = np.dot(o, self.W) + self.b
        return self.activate(z)

    def update_weights(self, delta, o):
        dW = np.dot(o.T, delta)
        # L2 regularization
        dW += self.weight_decay * self.W

        self.b -= self.learn_rate * np.sum(delta, axis=0)
        self.W -= self.learn_rate * dW


class NeuralNetwork:
    "Multi-Layer Perceptron (ANN)"

    def __init__(self, settings=NN_settings):
        self.cost_func = get_cost_func(settings["cost_type"])
        self.layers = []

        for i in range(len(settings["layers"])-1):
            layer = Layer(settings["layers"][i][0],
                          settings["layers"][i+1][0],
                          settings["learn_rate"],
                          settings["layers"][i+1][1],
                          settings["weight_decay"])
            self.layers.append(layer)

    def fit(self, X, Y, epochs):
        """Fit this model to the training data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (number of samples, number of features)
        Y : numpy.ndarray
            Vector with shape(number samples, 1) that specifies the index
            of the expected class
        """
        for k in range(1, epochs + 1):
            o = self.forward_prop(X)
            self.back_prop(o, Y)
            if k % 1000 == 0:
                print "Loss after iteration %i: %f" %(k, self.cost(X, Y))
        print "Final Loss after last iteration %i: %f" %((epochs), self.cost(X, Y))

    def forward_prop(self, X):
        """Returns list of output of each layer after activation."""
        o = [X]
        for l in self.layers:
            o.append(l.transform(o[-1]))

        return o

    def back_prop(self, o, Y):
        """Calculate deltas and update layer weights.

        Parameters
        ---------
        o : list(numpy.ndarray)
            List of forward propagation output from each layer
        Y : numpy.ndarray
            Vector with shape(number samples, 1) that specifies the index
            of the expected class
        """
        error = np.array(o[-1])
        error[range(len(error)), Y] -= 1

        deltas = [error * self.layers[-1].activate_p(o[-1])]

        # Calculate deltas
        for i in range(len(self.layers) - 2, -1, -1):
            W = self.layers[i+1].W
            delta = np.dot(deltas[-1], W.T) * self.layers[i].activate_p(o[i+1])
            deltas.append(delta)
        deltas.reverse()

        # Update weights
        for l in range(len(self.layers) ):
            self.layers[l].update_weights(deltas[l], o[l])

    def predict(self,X):
        """Perform forward propagation and predict a class."""
        return np.argmax(self.forward_prop(X)[-1], axis=1)

    def cost(self,X, Y):
        result = self.forward_prop(X)[-1]
        return self.cost_func(result, Y)
