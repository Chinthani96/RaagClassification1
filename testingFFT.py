import math
import numpy as np
from scipy.io.wavfile import read
from sympy import fft
from datetime import datetime

# 2^21 + 80
clip1 = read("longSamples/Yemen/A2.wav")
np.array(clip1[1], dtype=float)
audio = clip1[1]
print(len(audio))
# 80:2097232
audio1 = audio[80:83]
print(audio1)

# seperating the samples of the two channels into two vectors called left and right
# left1, right1 = zip(*audio1)
# training_input1 = left1
# print (len(training_input1))

# timeBefore = datetime.now()
# print(timeBefore)
# print("FFT Output")
# transform = fft(training_input1,4)
# # print(transform)
#
# with open('fftA1.txt', 'w') as f:
#     for item in transform:
#         f.write("%s\n" % item)
#
# print("Done")
# timeAfter = datetime.now()
# print(timeAfter)












# ----------------------------------GETTING THE MAGNITUDE AND THE PHASE FROM THE SEQUENCE-------------------

# def magnitude(num1, num2):
#     return math.sqrt(num1 * num1 + num2 * num2)
#
#
# def phase(num1, num2):
#     return math.atan(num2 / num1) * 180 / math.pi
# training_input1_m = np.arange(len(training_input1))
# training_input1_p = np.arange(len(training_input1))

# print("after")
# print (transform[1])

# for i in range(len(training_input1)):
#     s = str(transform[i])
#     print(s)
#     a = s.split('+')
#     b = a[1].split('*')
#     i = float(b[0])
#     r = float(a[0])
#
#     mag = magnitude(i,r)
#     ph = magnitude(i,r)
#     training_input1_m[i] = mag
#     training_input1_p[i] = ph
#     print("*")
#
#
# print("magnitude array of the FFT")
# print(training_input1_m)
#
# print("--------------------------")
# print("phase array of the FFT")
# print(training_input1_p)





