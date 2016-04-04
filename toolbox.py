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

activation_functions = {
    "sigmoid" : sigmoid,
    "tanh" : tanh,
    "softmax" : softmax
}

activation_prime_functions = {
    "sigmoid": sigmoid_prime,
    "tanh" : tanh_prime,
    "softmax" : sigmoid_prime
}

def is_valid_activation(a_type):
    return a_type in activation_functions

def get_activation(a_type):
    if not is_valid_activation(a_type):
        raise NotImplementedError("Bad activation function type: %s" % a_type)
    return activation_functions[a_type]

def get_activation_prime(a_type):
    if not is_valid_activation(a_type):
        raise NotImplementedError("Bad activation function type: %s" % a_type)
    return activation_prime_functions[a_type]

cost_functions = {
    "cross entropy loss": cross_entropy_loss
}

def get_cost_func(name):
    if name not in cost_functions:
        raise NotImplementedError("Bad cost function: %s" % name)
    return cost_functions[name]
