import numpy as np
import wavio


# Define sigmoid function to normalize weighted summations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define derivative of sigmoid function which is used to correct synaptic weights in each iteration.
def sigmoid_derivative(x):
    return x * (1 - x)


# import the wave file
from scipy.io.wavfile import read

a = read("yamen2.wav")
np.array(a[1], dtype=float)
audio = a[1]
print (len(audio))


# seperating the samples of the two channels into two vectors called left and right
left, right = zip(*audio)

# training the inputs
training_inputs = left
print(training_inputs)

# training outputs


# print("Array training outputs")

# print(training_outputs.size)


# creating the synaptic weights matrices
np.random.seed(1)

synaptic_weightsL1 = 2 * np.random.random((16, len(left))) - 1
synaptic_weightsL2 = 2 * np.random.random((1, 16)) - 1
print(len(synaptic_weightsL1))

print('The initial random values of the synaptic weights: ')
print(synaptic_weightsL1)
# print(synaptic_weights.__sizeof__())


print('Starting to train ...')

for iteration in range(1):
    input_layer = training_inputs

    middleLayer = sigmoid(np.dot(synaptic_weightsL1, input_layer))

print(middleLayer)

print(synaptic_weightsL2)

output = np.dot(synaptic_weightsL2, middleLayer)
print("akjs")
print(output)

for iteration in range(20000):
    input_layer = training_inputs

    middleLayer = sigmoid(np.dot(synaptic_weightsL1, input_layer))
    # bla bla bla
    output = np.dot(synaptic_weightsL2, middleLayer)

    training_output = 1

    error2 = training_output - output

    adjustments2 = error2 * sigmoid_derivative(output)
    synaptic_weights2 = synaptic_weights2 + np.dot(middleLayer, adjustments2)

    # error1 = middleLayer -
