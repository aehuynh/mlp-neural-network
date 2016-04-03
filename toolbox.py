import numpy as np

def softmax(z):
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(o):
    return o * (1.0 - o)

def tanh(z):
    return np.tanh(z)

def tanh_prime(o):
    return 1.0 - np.power(o, 2)

def cross_entropy_loss(result, expected):
    num_samples = len(result)
    cross_ent_err = -np.log(result[range(num_samples), expected])
    data_loss = np.sum(cross_ent_err)
    return 1./num_samples * data_loss

activation_function = {
    "sigmoid" : sigmoid,
    "tanh" : tanh,
    "softmax" : softmax
}

activation_prime_function = {
    "sigmoid": sigmoid_prime,
    "tanh" : tanh_prime
}

def is_valid_activation(name):
    return name in activation_function


def activation_func(name):
    if not is_valid_activation(name):
        raise NotImplementedError("Bad activation function name: %s" % name)

    return activation_function[name]

def activation_prime_func(name):
    if name not in activation_prime_function:
        raise NotImplementedError("No activation prime function found for: %s" % name)
    return activation_prime_function[name]

cost_function = {
    "cross entropy loss": cross_entropy_loss
}

def is_valid_cost_func(name):
    return name in cost_function

def cost_func(name):
    if not is_valid_cost_func(name):
        raise NotImplementedError("Bad cost function: %s" % name)
    return cost_function[name]
