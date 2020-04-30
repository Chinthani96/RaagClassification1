import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

from scipy.io.wavfile import read
clip1 = read("samples/clip1.wav")
np.array(clip1[1], dtype=float)
audio1 = clip1[1]
# seperating the samples of the two channels into two vectors called left and right
left1, right1 = zip(*audio1)
training_input1 = left1


print(training_input1)

# # x = np.array([-10000,-1000,-100,-10,-5,-3,0,3,5,10,100,1000,10000])
# print(sigmoid(np.array(training_input1)))
# print("---------")
# print("---------")
# print(sigmoid_derivative(np.array(training_input1)))
#
# y=89
# print(sigmoid(y))
# print(sigmoid_derivative(y))