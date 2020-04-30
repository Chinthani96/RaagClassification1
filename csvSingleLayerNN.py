import numpy as np
import pandas as pd
from numpy import genfromtxt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def convertCSV(x,y):
    for i in range(len(x)):
        y[i] = x[i][0]
#   -----------------READING CSV FILES OF YAMEN RAAG--------------
# 1
data_set = pd.read_csv("longSamples/Yemen/A1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input1 = np.empty([2686764,])
convertCSV(training_input,training_input1)


# 2
data_set = pd.read_csv("longSamples/Yemen/A2_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input2 = np.empty([2686764,])
convertCSV(training_input,training_input2)

# 3
data_set = pd.read_csv("longSamples/Yemen/A3_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input3 = np.empty([2686764,])
convertCSV(training_input,training_input3)
print("break point")

# 4
data_set = pd.read_csv("longSamples/Yemen/A4_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input4 = np.empty([2686764,])
convertCSV(training_input,training_input4)
print("break point")
# 5
data_set = pd.read_csv("longSamples/Yemen/B1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input5 = np.empty([2686764,])
convertCSV(training_input,training_input5)
print("break point")

# 6
data_set = pd.read_csv("longSamples/Yemen/B2_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input6 = np.empty([2686764,])
convertCSV(training_input,training_input6)
print("break point")

# 7
data_set = pd.read_csv("longSamples/Yemen/B3_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input7 = np.empty([2686764,])
convertCSV(training_input,training_input7)
print("break point")

# 8
data_set = pd.read_csv("longSamples/Yemen/B21_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input8 = np.empty([2686764,])
convertCSV(training_input,training_input8)
print("break point")

# 9
data_set = pd.read_csv("longSamples/Yemen/B22_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input9 = np.empty([2686764,])
convertCSV(training_input,training_input9)
print("break point")
# 10
data_set = pd.read_csv("longSamples/Yemen/B23_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input10 = np.empty([2686764,])
convertCSV(training_input,training_input10)
print("break point")

# 11
data_set = pd.read_csv("longSamples/Yemen/C1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input11 = np.empty([2686764,])
convertCSV(training_input,training_input11)

# 12
data_set = pd.read_csv("longSamples/Yemen/D1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input12 = np.empty([2686764,])
convertCSV(training_input,training_input12)

# 13
data_set = pd.read_csv("longSamples/Yemen/D2_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input13 = np.empty([2686764,])
convertCSV(training_input,training_input13)

# 14
data_set = pd.read_csv("longSamples/Yemen/E1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input14 = np.empty([2686764,])
convertCSV(training_input,training_input14)

# 15
data_set = pd.read_csv("longSamples/Yemen/E2_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input15 = np.empty([2686764,])
convertCSV(training_input,training_input15)

# 16
data_set = pd.read_csv("longSamples/Yemen/E3_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input16 = np.empty([2686764,])
convertCSV(training_input,training_input16)

# 17
data_set = pd.read_csv("longSamples/Yemen/F1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input17 = np.empty([2686764,])
convertCSV(training_input,training_input17)

# 18
data_set = pd.read_csv("longSamples/Yemen/F2_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input18 = np.empty([2686764,])
convertCSV(training_input,training_input18)

# 19
data_set = pd.read_csv("longSamples/Yemen/F3_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input19 = np.empty([2686764,])
convertCSV(training_input,training_input19)

# 20
data_set = pd.read_csv("longSamples/Yemen/F4_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input20 = np.empty([2686764,])
convertCSV(training_input,training_input20)

# 21
data_set = pd.read_csv("longSamples/Yemen/F5_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input21 = np.empty([2686764,])
convertCSV(training_input,training_input21)

# 22
data_set = pd.read_csv("longSamples/Yemen/G1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input22 = np.empty([2686764,])
convertCSV(training_input,training_input22)

# 23
data_set = pd.read_csv("longSamples/Yemen/G2_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input23 = np.empty([2686764,])
convertCSV(training_input,training_input23)

# 24
data_set = pd.read_csv("longSamples/Yemen/G3_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input24 = np.empty([2686764,])
convertCSV(training_input,training_input24)

# 25
data_set = pd.read_csv("longSamples/Yemen/G4_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input25 = np.empty([2686764,])
convertCSV(training_input,training_input25)
print("BREAK POINT")

# -----------------READING OTHER RAAG CSVs--------------------------
# 1'
data_set = pd.read_csv("longSamples/Other/jog1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input_1 = np.empty([2686764,])
convertCSV(training_input,training_input_1)
print("break point")
# 2'
data_set = pd.read_csv("longSamples/Other/jog2_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input_2 = np.empty([2686764,])
convertCSV(training_input,training_input_2)
print("break point")
# 3'
data_set = pd.read_csv("longSamples/Other/jog3_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input_3 = np.empty([2686764,])
convertCSV(training_input,training_input_3)
print("break point")
# 4'
data_set = pd.read_csv("longSamples/Other/pdanashri1_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input_4 = np.empty([2686764,])
convertCSV(training_input,training_input_4)
print("break point")
# 5'
data_set = pd.read_csv("longSamples/Other/pdanashri2_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input_5 = np.empty([2686764,])
convertCSV(training_input,training_input_5)
#
# 6'
data_set = pd.read_csv("longSamples/Other/pdanashri3_fmod.csv", header=None)
data_frames = pd.DataFrame(data_set)
training_input = np.array(data_frames.values)
training_input_6 = np.empty([2686764,])
convertCSV(training_input,training_input_6)
print("break point-----------------------")


#               ASSIGNING THE INPUTS TO A NUMPY ARRAY
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
print(np.shape(synaptic_weights))
print("----------------------break point")

for iteration in range(1):
    input_layer = training_inputs
    # print(np.shape(input_layer))

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    print("before")
    print(synaptic_weights)
    synaptic_weights = synaptic_weights + np.dot(input_layer.T, adjustments)
    print("after")
    print(synaptic_weights)