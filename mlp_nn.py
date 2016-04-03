import numpy as np
from toolbox import is_valid_activation, activation_prime_func, activation_func, cost_func, is_valid_cost_func

class Layer:
    """Neural network layer"""

    def __init__(self, num_in, num_nodes, learn_rate, a_type="tanh", weight_decay=None):
        if not is_valid_activation(a_type):
            raise NotImplementedError("Bad activation function type: %s" % a_type)

        self.W = np.random.randn(num_in, num_nodes) / np.sqrt(num_in)
        self.b = np.zeros((1, num_nodes))
        self.a_type = a_type
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay

    def transform(self, o):
        z = o.dot(self.W) + self.b
        return activation_func(self.a_type)(z)

    def update_weights(self, delta, o):
        dW = (o.T).dot(delta)

        if self.weight_decay is not None:
            dW += self.weight_decay * self.W

        self.b -= self.learn_rate * np.sum(delta, axis=0)
        self.W -= self.learn_rate * dW

class NeuralNetwork:

    def __init__(self, layer_sizes, layer_a_types, learn_rate, cost_type="cross entropy loss", weight_decay=None):
        if not is_valid_cost_func(cost_type):
            raise NotImplementedError("Bad cost function type: %s" % cost_type)

        self.cost_type = cost_type
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], learn_rate,layer_a_types[i],  weight_decay)
            self.layers.append(layer)

    def fit(self, X, Y, epochs):
        """Fit this model to the training data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data with shape (number of samples, number of features)
        Y : numpy.ndarray
            Expected output
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
            List of forward propogation output from each layer
        Y : numpy.ndarray
            Expected output
        """
        last_delta = np.array(o[-1])
        last_delta[range(len(last_delta)), Y] -= 1

        deltas = [last_delta]

        # Calculate deltas
        for i in range(len(self.layers) - 2, -1, -1):
            W = self.layers[i+1].W
            delta = deltas[-1].dot(W.T) * activation_prime_func(self.layers[i].a_type)(o[i+1])
            deltas.append(delta)
        deltas.reverse()

        # Update weights
        for l in range(len(self.layers) - 1):
            self.layers[l].update_weights(deltas[l], o[l])

    def predict(self,X):
        """Perform forward propagation and predict a class."""
        return np.argmax(self.forward_prop(X)[-1], axis=1)

    def cost(self,X, Y):
        result = self.forward_prop(X)[-1]
        return cost_func(self.cost_type)(result, Y)