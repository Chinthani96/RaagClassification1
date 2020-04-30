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

clip2 = read("samples/clip2.wav")
np.array(clip2[1], dtype=float)
audio2 = clip2[1]
# seperating the samples of the two channels into two vectors called left and right
left2, right2 = zip(*audio2)
training_input2 = left2

input_layer = np.array([training_input1,
                        training_input2])

np.random.seed(1)
synaptic_weights = 2*np.random.random((len(training_input1),1))-1
print(synaptic_weights)

print(sigmoid(np.dot(input_layer, synaptic_weights)))