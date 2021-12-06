from random import seed
from random import random
from math import exp
import numpy as np
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

def normalize_data(data, minmax):
		for i in range(len(data)):
			data[i] = (data[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
        # print ("act ", activation)
        # print ("inputs ", inputs)
    return inputs

def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# load and prepare data
filename = 'semua.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# konversi menjadi integer
str_column_to_int(dataset, len(dataset[0])-1)
# normalisasi data minmax
minmax = dataset_minmax(dataset)
#load model
netw = np.load("network-train.npy", allow_pickle=True, fix_imports=True, encoding='ASCII')

# test = [170.3104,0.11599999999999999,0.0227,-0.0005]

def klasifikasiBP(test):
    print ("awal :", test)
    normalize_data(test, minmax)
    print ("normal :", test)
    prediction = predict(netw, test)
    return prediction

# print ("hasil : ", klasifikasi(test))