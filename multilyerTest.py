# 29/04/2020
# 10-6-4 TWO LAYER ANN

# importing libraries
import numpy as np


# defining functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def convert_to_1d_arr(x, y):
    for i in range(6):
        y[i] = x[i][0]


#  ------------------------LAYER ONE---------------------------------------------------#

# synaptic weights between input layer and layer 1 are 6, 1x10 row vectors
# There are 6 biases, one for each perceptron

# initializing synaptic weights and biases for layer one


np.random.seed(1)
# let the training vector be 10x1 column vector
# weights_j_1 means the weights connecting to neuron j of layer 1.
# weights connecting inputs to perceptron 1 of layer 1
weights_1_1 = 2 * np.random.random((10, 1)) - 1

# weights connecting inputs to perceptron 2 of layer 1
weights_2_1 = 2 * np.random.random((10, 1)) - 1

# weights connecting inputs to perceptron 3 of layer 1
weights_3_1 = 2 * np.random.random((10, 1)) - 1

# weights connecting inputs to perceptron 4 of layer 1
weights_4_1 = 2 * np.random.random((10, 1)) - 1

# weights connecting inputs to perceptron 5 of layer 1
weights_5_1 = 2 * np.random.random((10, 1)) - 1

# weights connecting inputs to perceptron 6 of layer 1
weights_6_1 = 2 * np.random.random((10, 1)) - 1

# initializing the biases
biases_1 = 2 * np.random.random((6, 1)) - 1

input_layer_1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                          [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                          [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                          [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                          [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
                          [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                          [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 0, 0, 1]]
                         )
#
# # this gives a [6,50,1] array. To get a [6,50] array, we do the following.
#
#
#
#
#
# input_layer_1 = np.empty(6,50)


training_outputs = np.array([[0, 1, 0, 1],
                             [1, 0, 1, 0],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1],
                             [0, 1, 0, 1],
                             [1, 0, 0, 1]
                             ]).T

# the outputs of the first layer
y_1_1 = sigmoid(np.dot(input_layer_1, weights_1_1) + biases_1[0])
print(np.shape(y_1_1))
y_2_1 = sigmoid(np.dot(input_layer_1, weights_2_1) + biases_1[1])
y_3_1 = sigmoid(np.dot(input_layer_1, weights_3_1) + biases_1[2])
y_4_1 = sigmoid(np.dot(input_layer_1, weights_4_1) + biases_1[3])
y_5_1 = sigmoid(np.dot(input_layer_1, weights_5_1) + biases_1[4])
y_6_1 = sigmoid(np.dot(input_layer_1, weights_6_1) + biases_1[5])

outputs_1 = np.array([y_1_1,
                    y_2_1,
                    y_3_1,
                    y_4_1,
                    y_5_1,
                    y_6_1])

# print(len(outputs[1]))
# outputs_1 = np.empty(50,)
# print(np.shape(outputs))
# convert_to_1d_arr(outputs, outputs_1)
# print(outputs_1)



weights_1 = np.array([weights_1_1,
                      weights_2_1,
                      weights_3_1,
                      weights_4_1,
                      weights_5_1,
                      weights_6_1])

# print(weights_1)
# print(biases_1)
# print(outputs_1)

# --------------------------------LAYER TWO (OUTPUT LAYER)------------------------------------

# synaptic weights between middle layer and output layer are 4, 1x6 row vectors
# There are 4 biases, one for each perceptron

# initializing synaptic weights and biases for layer two

# the input to the second layer will be outputs_1
# weights_k_2 means the weights connecting to neuron k of layer 2.
# weights connecting inputs to perceptron 1 of layer 2
weights_1_2 = 2 * np.random.random((6, 1)) - 1

# weights connecting inputs to perceptron 2 of layer 1
weights_2_2 = 2 * np.random.random((6, 1)) - 1

# weights connecting inputs to perceptron 3 of layer 1
weights_3_2 = 2 * np.random.random((6, 1)) - 1

# weights connecting inputs to perceptron 4 of layer 1
weights_4_2 = 2 * np.random.random((6, 1)) - 1

# initializing the biases
biases_2 = 2 * np.random.random((4, 1)) - 1

input_layer_2 = outputs_1.T
# input_layer_2 = outputs_1

# the outputs of the second layer
y_1_2 = sigmoid(np.dot(input_layer_2, weights_1_2)) + biases_2[0]
y_2_2 = sigmoid(np.dot(input_layer_2, weights_2_2)) + biases_2[1]
y_3_2 = sigmoid(np.dot(input_layer_2, weights_3_2)) + biases_2[2]
y_4_2 = sigmoid(np.dot(input_layer_2, weights_4_2)) + biases_2[3]

# outputs_2 = np.array([y_1_2,
#                       y_2_2,
#                       y_3_2,
#                       y_4_2])

outputs_2 = np.array([y_1_2])

weights_2 = np.array([weights_1_2,
                      weights_2_2,
                      weights_3_2,
                      weights_4_2])

# print(weights_2)
# print(biases_2)
# print(outputs_2)
# print("----------------------")

# training the network

# define etah, the training rate
training_rate = 0.5

for iteration in range(1):
    input_layer = outputs_1
    print(type(input_layer))
    print(np.shape(input_layer))
    print(type(weights_1_2))
    print(np.shape(weights_1_2))

    # for perceptron 1 of layer 2
    outputs = sigmoid(np.dot(input_layer,weights_1_2))
    print(outputs)

    delta1_2 = -2*(training_outputs-outputs)*sigmoid_derivative(outputs)
    # print(weights_1_2)
    weights_1_2 = weights_1_2 - training_rate*np.dot(input_layer.T,delta1_2)
    # synaptic_weights = synaptic_weights-training_rate(np.dot(input_layer.T, delta2))
    # print(weights_1_2)

print('Outputs after training:')
print(outputs)
