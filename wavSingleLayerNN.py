import numpy as np
import csv

# sigmoid function to normalize weighted summations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

from scipy.io.wavfile import read

# Yemen Clips
clip1 = read("longSamples/Yemen/A1.wav")
np.array(clip1[1],dtype=float)
audio = clip1[1]
audio1 = audio[80:]
# print(type(audio1))
left1,right1 = zip(*audio1)
# print(left1)
training_input1 = left1
# print(np.shape(training_input1))

clip2 = read("longSamples/Yemen/A2.wav")
np.array(clip2[1],dtype=float)
audio = clip2[1]
audio2 = audio[80:]
left2,right2 = zip(*audio2)
training_input2 = left2

clip3 = read("longSamples/Yemen/A3.wav")
np.array(clip3[1],dtype=float)
audio = clip3[1]
audio3 = audio[80:]
left3,right3 = zip(*audio3)
training_input3 = left3

clip4 = read("longSamples/Yemen/A4.wav")
np.array(clip4[1],dtype=float)
audio = clip4[1]
audio4 = audio[80:]
left4,right4 = zip(*audio4)
training_input4 = left4

clip5 = read("longSamples/Yemen/B1.wav")
np.array(clip5[1],dtype=float)
audio = clip5[1]
audio5 = audio[80:]
left5,right5 = zip(*audio5)
training_input5 = left5

clip6 = read("longSamples/Yemen/B2.wav")
np.array(clip6[1],dtype=float)
audio = clip6[1]
audio6 = audio[80:]
left6,right6 = zip(*audio6)
training_input6 = left6

clip7 = read("longSamples/Yemen/B3.wav")
np.array(clip7[1],dtype=float)
audio = clip7[1]
audio7 = audio[80:]
left7,right7 = zip(*audio7)
training_input7 = left7

clip8 = read("longSamples/Yemen/B21.wav")
np.array(clip8[1],dtype=float)
audio = clip8[1]
audio8 = audio[80:]
left8,right8 = zip(*audio8)
training_input8 = left8

clip9 = read("longSamples/Yemen/B22.wav")
np.array(clip9[1],dtype=float)
audio = clip9[1]
audio9 = audio[80:]
left9,right9 = zip(*audio9)
training_input9 = left9

clip10 = read("longSamples/Yemen/B23.wav")
np.array(clip10[1],dtype=float)
audio = clip10[1]
audio10 = audio[80:]
left10,right10 = zip(*audio10)
training_input10 = left10

clip11 = read("longSamples/Yemen/C1.wav")
np.array(clip11[1],dtype=float)
audio = clip11[1]
audio11 = audio[80:]
left11,right11 = zip(*audio11)
training_input11 = left11

clip12 = read("longSamples/Yemen/D1.wav")
np.array(clip12[1],dtype=float)
audio = clip12[1]
audio12 = audio[80:]
left12,right12 = zip(*audio12)
training_input12 = left12

clip13 = read("longSamples/Yemen/D2.wav")
np.array(clip13[1],dtype=float)
audio = clip13[1]
audio13 = audio[80:]
left13,right13 = zip(*audio13)
training_input13 = left13

clip14 = read("longSamples/Yemen/E1.wav")
np.array(clip14[1],dtype=float)
audio = clip14[1]
audio14 = audio[80:]
left14,right14 = zip(*audio14)
training_input14 = left14

clip15 = read("longSamples/Yemen/E2.wav")
np.array(clip15[1],dtype=float)
audio = clip15[1]
audio15 = audio[80:]
left15,right15 = zip(*audio15)
training_input15 = left15

clip16 = read("longSamples/Yemen/E3.wav")
np.array(clip16[1],dtype=float)
audio = clip16[1]
audio16 = audio[80:]
left16,right16 = zip(*audio16)
training_input16 = left16

clip17 = read("longSamples/Yemen/F1.wav")
np.array(clip17[1],dtype=float)
audio = clip17[1]
audio17 = audio[80:]
left17,right17 = zip(*audio17)
training_input17 = left17

clip18 = read("longSamples/Yemen/F2.wav")
np.array(clip18[1],dtype=float)
audio = clip18[1]
audio18 = audio[80:]
left18,right18 = zip(*audio18)
training_input18 = left18

clip19 = read("longSamples/Yemen/F3.wav")
np.array(clip19[1],dtype=float)
audio = clip19[1]
audio19 = audio[80:]
left19,right19 = zip(*audio19)
training_input19 = left19

clip20 = read("longSamples/Yemen/F4.wav")
np.array(clip20[1],dtype=float)
audio = clip20[1]
audio20 = audio[80:]
left20,right20 = zip(*audio20)
training_input20 = left20

clip21 = read("longSamples/Yemen/F5.wav")
np.array(clip21[1],dtype=float)
audio = clip21[1]
audio21 = audio[80:]
left21,right21 = zip(*audio21)
training_input21 = left21

clip22 = read("longSamples/Yemen/G1.wav")
np.array(clip22[1],dtype=float)
audio = clip22[1]
audio22 = audio[80:]
left22,right22 = zip(*audio22)
training_input22 = left22

clip23 = read("longSamples/Yemen/G2.wav")
np.array(clip23[1],dtype=float)
audio = clip23[1]
audio23 = audio[80:]
left23,right23 = zip(*audio23)
training_input23 = left23

clip24 = read("longSamples/Yemen/G3.wav")
np.array(clip24[1],dtype=float)
audio = clip24[1]
audio24 = audio[80:]
left24,right24 = zip(*audio24)
training_input24 = left24

clip25 = read("longSamples/Yemen/G4.wav")
np.array(clip25[1],dtype=float)
audio = clip25[1]
audio25 = audio[80:]
left25,right25 = zip(*audio25)
training_input25 = left25

# Other clips
clip_1 = read("longSamples/Other/jog1.wav")
np.array(clip_1[1],dtype=float)
audio = clip_1[1]
audio_1 = audio[80:]
left_1,right_1 = zip(*audio_1)
training_input_1 = left_1

clip_2 = read("longSamples/Other/jog2.wav")
np.array(clip_2[1],dtype=float)
audio = clip_2[1]
audio_2 = audio[80:]
left_2,right_2 = zip(*audio_2)
training_input_2 = left_2

clip_3 = read("longSamples/Other/jog3.wav")
np.array(clip_3[1],dtype=float)
audio = clip_3[1]
audio_3 = audio[80:]
left_3,right_3 = zip(*audio_3)
training_input_3 = left_3

clip_4 = read("longSamples/Other/pdanashri1.wav")
np.array(clip_4[1],dtype=float)
audio = clip_4[1]
audio_4 = audio[80:]
left_4,right_4 = zip(*audio_4)
training_input_4 = left_4

clip_5 = read("longSamples/Other/pdanashri2.wav")
np.array(clip_5[1],dtype=float)
audio = clip_5[1]
audio_5 = audio[80:]
left_5,right_5 = zip(*audio_5)
training_input_5 = left_5

clip_6 = read("longSamples/Other/pdanashri3.wav")
np.array(clip_6[1],dtype=float)
audio = clip_6[1]
audio_6 = audio[80:]
left_6,right_6 = zip(*audio_6)
training_input_6 = left_6

training_inputs = np.array([training_input1,
                            training_input2,
                            training_input3,
                            training_input4,
                            training_input5,
                            training_input6,
                            training_input7,
                            training_input8,
                            training_input9,
                            training_input10,
                            training_input11,
                            training_input12,
                            training_input13,
                            training_input14,
                            training_input15,
                            training_input16,
                            training_input17,
                            training_input18,
                            training_input19,
                            training_input20,
                            training_input21,
                            training_input22,
                            training_input23,
                            training_input24,
                            training_input25,
                            training_input_1,
                            training_input_2,
                            training_input_3,
                            training_input_4,
                            training_input_5,
                            training_input_6
                            ])

training_outputs = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0]]).T

# generating the random synaptic weights
np.random.seed(1)
synaptic_weights = 2*np.random.random((len(training_input1),1))-1
# print(np.shape(synaptic_weights))

for iteration in range(1):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    print(synaptic_weights)
    synaptic_weights = synaptic_weights + np.dot(input_layer.T, adjustments)
    print(synaptic_weights)

# print('Outputs after training:')
# print(outputs)
# print("-----------")
# print(error)







