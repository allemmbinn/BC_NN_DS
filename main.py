import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

# Display Versions
""" print("Python Version: ", sys.version)
print("Numpy Version: ", np.__version__)
print("Matplotlib Version: ", matplotlib.__version__)
 """
# One Neuron
"""
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

outputs = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(outputs)
"""

# Three Neurons - One Layer
"""
inputs = [1.0, 2.0, 3.0, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5] 
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [ inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
            inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
            inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3
        ]

print(outputs)
"""

# Doing the above in python
""" inputs = [1.0, 2.0, 3.0, 2.5]
weights =  [ [0.2, 0.8, -0.5, 1.0],
             [0.5, -0.91, 0.26, -0.5],
             [-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5] """

""" layer_outputs = []
for neuron_weights, neuron_bias in zip(weights,biases):
    neuron_output = 0
    for n_input, weights in zip(inputs,neuron_weights):
        neuron_output += n_input*weights
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)
 """
# Doing the above with numpy

""" output = np.dot(weights, inputs) + biases
print(output)
 """
# Batches for Neural Network Inputs - 2 Layers
""" inputs = [ [1.0, 2.0, 3.0, 2.5],
           [2.0, 5.0, -1.0, 2.0],
           [-1.5, 2.7, 3.3, -0.8]
         ]

weights1 =  [ [0.2, 0.8, -0.5, 1.0],
             [0.5, -0.91, 0.26, -0.5],
             [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2,3,0.5]

weights2 =  [ [0.1, -0.14, 0.5],
              [-0.5, 0.12, -0.33],
              [-0.44, 0.73, -0.13]   
            ]    

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs) """

# Using Objects and Classes for Neural Networks

""" np.random.seed(0)

X = [ [1.0, 2.0, 3.0, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]
    ]

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# The number of Neurons are arbitary
layer1 = Dense_Layer(4,5)
layer2 = Dense_Layer(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output) """

# Using RELU Activation Function

""" np.random.seed(0)

X = [ [1.0, 2.0, 3.0, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]
    ]

inputs = [ 0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = []

# RELU Activation Function 
for i in inputs:
    outputs.append(max(0,i))

# Creating Data - Here it creates spirals with given no of classes and given no of points
def create_data(points, classes):
    X = np.zeros((points*classes, 2)) # This creates in X and Y coordinates
    Y = np.zeros(points*classes, dtype='uint8') # Denotes which class it belongs to
    for class_n in range(classes):
        ix = range(points*class_n, points*(class_n+1))
        r  = np.linspace(0.0,1,points) # Radius
        t  = np.linspace(class_n*4, (class_n+1)*4, points) + np.random.randn(points)*0.2 # Theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        Y[ix] = class_n
    return X, Y

X,Y = create_data(100,3)

# Plotting the points
plt.scatter(X[:,0],X[:,1], c=Y, cmap='brg') 
plt.show()

print(outputs) """

# RELU Activation Function

""" nnfs.init()

X, y = spiral_data(100,3)

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Dense_Layer(2,5) # The first 2 is for the coordinates X and Y
activation1 = ReLU()

layer1.forward(X)
# print(layer1.output[:5])
activation1.forward(layer1.output)
print(activation1.output[:5]) """


# Softmax Activation Function

""" layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values)) """

# Softmax Activation Function - Class

nnfs.init()

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

x, y = spiral_data(samples=100, classes=3)
layer1 = Dense_Layer(2,3) # The first 2 is for the coordinates X and Y
activation1 = ReLU()
dense2 = Dense_Layer(3,3)
activation2 = Softmax()

layer1.forward(x)
activation1.forward(layer1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])