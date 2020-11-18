import numpy as np
# import tensorflow as tf
import pandas as pd
# import csv
# import math
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer 
# from sklearn.utils import shuffle
import matplotlib.pyplot as plt

TRAIN_RATE = 0.75
LR = 1e-6
HIDDEN_SIZE = 8
ITERATION = 100000
'''
# Relative Compactness,
Surface Area,
Wall Area,
Roof Area,
Overall Height,
Orientation,
Glazing Area,
Glazing Area Distribution
'''
def forward(input, weights_1, weights_2):
	a = np.dot(input, weights_1)
	a[a<0] = 0
	
	output = np.dot(a, weights_2)
	return output, a

csv_data = pd.read_csv('energy_efficiency_data.csv')
ohe_data = pd.get_dummies(csv_data, columns=['Orientation', 'Glazing Area Distribution'])
N = csv_data.shape[0]
# x_data = ohe_data.drop(columns = ['Heating Load', 'Cooling Load'])
x_data = ohe_data[['# Relative Compactness', 'Wall Area', 'Roof Area', 'Overall Height', 'Glazing Area']]
# print(x_data.head())
y_heat_data = ohe_data['Heating Load']

idx = np.random.permutation(N)
train_idx = idx[:int(N*TRAIN_RATE)]
test_idx = idx[int(N*TRAIN_RATE):]

x_train = x_data.iloc[train_idx].to_numpy()
bias = np.ones((x_train.shape[0], 1))
x_train = np.append(x_train, bias, axis=1)
y_train = y_heat_data.iloc[train_idx].to_numpy()
y_train = np.expand_dims(y_train, axis=1)

x_test = x_data.iloc[test_idx].to_numpy()
bias = np.ones((x_test.shape[0], 1))
x_test = np.append(x_test, bias, axis=1)
y_heat_test = y_heat_data.iloc[test_idx].to_numpy()
y_heat_test = np.expand_dims(y_heat_test, axis=1)

# print('x_train.shape', x_train.shape)
# print('y_train.shape', y_train.shape)

max_x = np.max(x_train, axis=0)
x_train = x_train / max_x
x_test = x_test / max_x

weights_1 = np.random.randn(x_train.shape[1], HIDDEN_SIZE)
weights_2 = np.random.randn(HIDDEN_SIZE, 1)
# print('weights_1.shape', weights_1.shape)
# print('weights_2.shape', weights_2.shape)

loss_mem = np.zeros(ITERATION)

for iter in range(ITERATION):
	
	pred_y, mid = forward(x_train, weights_1, weights_2)
	
	loss = np.sum(np.square(y_train - pred_y))
	if iter < 100000:
		loss_mem[iter] = loss
	
	deri_a = mid
	deri_a[mid != 0] = 1
	grad_w1 = np.dot(x_train.T, np.multiply(np.dot((pred_y - y_train), weights_2.T), deri_a))
	weights_1 -= LR * grad_w1
	
	grad_w2 = np.dot(mid.T, pred_y - y_train)
	weights_2 -= LR * grad_w2
	
	if (iter+1) % 1000 == 0:
		print('iterations {:6d}: loss {:11.5f}'.format(iter+1, loss))
	
	
# print(weights_1, weights_2)
# print(np.sum(np.abs(weights_1), axis=1))
plt.figure()
plt.plot(loss_mem[:2000])

y_train_hat, a = forward(x_train, weights_1, weights_2)
y_heat_test_hat, a = forward(x_test, weights_1, weights_2)

train_RMS = np.sqrt(np.mean(np.square(y_train - y_train_hat)))
test_RMS = np.sqrt(np.mean(np.square(y_heat_test - y_heat_test_hat)))
print('train RMS error', train_RMS)
print('test RMS error', test_RMS)

part = 100
plt.figure()
plt.plot(y_train[:part])
plt.plot(y_train_hat[:part])
plt.title('prediction for training data')
plt.xlabel('n'), plt.ylabel('heating load')
plt.legend(['ground truth', 'prediction'])

plt.figure()
plt.plot(y_heat_test[:part])
plt.plot(y_heat_test_hat[:part])
plt.title('prediction for testing data')
plt.xlabel('n'), plt.ylabel('heating load')
plt.legend(['ground truth', 'prediction'])
plt.show()