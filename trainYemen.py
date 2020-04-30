import numpy as np



# Define sigmoid function to normalize weighted summations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define derivative of sigmoid function which is used to correct synaptic weights in each iteration.
def sigmoid_derivative(x):
    return x * (1 - x)


# import the wave file
from scipy.io.wavfile import read

clip1 = read("samples/clip1.wav")
np.array(clip1[1], dtype=float)
audio = clip1[1]
audio1 = audio[80:]
# seperating the samples of the two channels into two vectors called left and right
left1, right1 = zip(*audio1)
training_input1 = left1

clip2 = read("samples/clip2.wav")
np.array(clip2[1], dtype=float)
audio = clip2[1]
audio2 = audio[80:]
# seperating the samples of the two channels into two vectors called left and right
left2, right2 = zip(*audio2)
training_input2 = left2

clip3 = read("samples/clip3.wav")
np.array(clip3[1], dtype=float)
audio = clip3[1]
audio3 = audio[80:]
# seperating the samples of the two channels into two vectors called left and right
left3, right3 = zip(*audio3)
training_input3 = left3

clip4 = read("samples/clip4.wav")
np.array(clip4[1], dtype=float)
audio = clip4[1]
audio4 = audio[80:]
# seperating the samples of the two channels into two vectors called left and right
left4, right4 = zip(*audio4)
training_input4 = left4

clip5 = read("samples/clip5.wav")
np.array(clip5[1], dtype=float)
audio = clip5[1]
audio5 = audio[80:]
# seperating the samples of the two channels into two vectors called left and right
left5, right5 = zip(*audio5)
training_input5 = left5

clip6 = read("samples/clip6B.wav")
np.array(clip6[1], dtype=float)
audio = clip6[1]
audio6 = audio[80:]
# seperating the samples of the two channels into two vectors called left and right
left6, right6 = zip(*audio6)
training_input6 = left6

training_inputs = np.array([training_input1,
                            training_input2,
                            training_input3,
                            training_input4,
                            training_input5,
                            training_input6])

training_outputs = np.array([[1,1,1,1,1,0]]).T

np.random.seed(1)
synaptic_weights = 2*np.random.random((len(training_input1),1))-1
# print(synaptic_weights)


for iteration in range(10):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs
    # print(error)

    adjustments = error * sigmoid_derivative(outputs)

    # print (sigmoid_derivative(outputs))
    # print(np.dot(input_layer.T, adjustments))

    print(synaptic_weights)
    synaptic_weights = synaptic_weights + np.dot(input_layer.T, adjustments)
    # print(np.dot(input_layer.T, adjustments))
    print(synaptic_weights)


# print('Outputs after training:')
# print(outputs)
# print("-----------")
# print(error)











