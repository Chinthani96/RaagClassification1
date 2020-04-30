"""
Created on Sun Feb 23 14:49:59 2020

@author: athula
"""

# Program to train ANN for a 3 input Exclusive OR gate
# Single iteration


import numpy as np


# Define sigmoid function to normalize weighted summations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define derivative of sigmoid function which is used to correct synaptic weights in each iteration.
def sigmoid_derivative(x):
    return x * (1 - x)


training_inputs = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [1, 0, 1],
                            [1, 1, 1]])

training_outputs = np.array([[0, 1, 0, 0]]).T

print("training outputs")
print(training_outputs)

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1
print("wjdfbqkjebfdej2f")
print(synaptic_weights)

print('The initial random values of the synaptic weights: ')
print(synaptic_weights)

print('Starting to train ...')

for iteration in range(20000):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # bla bla bla
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights = synaptic_weights + np.dot(input_layer.T, adjustments)

print('Outputs after training:')
print(outputs)

# outputs for [0,1,1] and [1,0,0]

input_layer = [0, 1, 1]
outputs = sigmoid(np.dot(input_layer, synaptic_weights))
print(outputs)

input_layer = [1, 0, 0]
outputs = sigmoid(np.dot(input_layer, synaptic_weights))
print(outputs)