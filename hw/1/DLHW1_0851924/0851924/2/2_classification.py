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
from mpl_toolkits.mplot3d import Axes3D

TRAIN_RATE = 0.8
LR = 1e-3
HIDDEN_SIZE = 50
ITERATION = 100000

def softmax(input, axis=0):
	return np.exp(input) / np.sum(np.exp(input), axis=axis)[:, None]

def sigmoid(input, deri=False):
	if deri:
		return np.divide(np.exp(-1*input) / np.square(1 + np.exp(-1*input)))
	else:
		return 1 / (1 + np.exp(-1*input))

def forward(input, weights1, weights2):
	a = np.dot(input, weights1)
	a[a<0] = 0
	
	output = np.dot(a, weights2)
	return softmax(output, 1), a
	
	
raw_data = pd.read_csv("ionosphere_data.csv", header=None)
onehot_data = pd.get_dummies(raw_data)
N = onehot_data.shape[0]

x_data = onehot_data.drop(columns = ["34_b", "34_g"])
y_data = onehot_data[["34_g", "34_b"]]

idx = np.random.permutation(N)
train_idx = idx[:int(N*TRAIN_RATE)]
test_idx = idx[int(N*TRAIN_RATE):]

x_train = x_data.iloc[train_idx].to_numpy()
b = np.ones((x_train.shape[0], 1))
x_train = np.append(x_train, b, axis=1)
y_train = y_data.iloc[train_idx].to_numpy()

test_x = x_data.iloc[test_idx].to_numpy()
b = np.ones((test_x.shape[0], 1))
test_x = np.append(test_x, b, axis=1)
test_y = y_data.iloc[test_idx].to_numpy()

# print("x_train.shape", x_train.shape)
# print("y_train.shape", y_train.shape)

weights_1 = np.random.randn(x_train.shape[1], HIDDEN_SIZE)
weights_2 = np.random.randn(HIDDEN_SIZE, 2)
# print("weights_1.shape", weights_1.shape)
# print("weights_2.shape", weights_2.shape)

loss_mem = np.zeros(ITERATION)

for iter in range(ITERATION):
	
	pred_y, a = forward(x_train, weights_1, weights_2)
	
	loss = -np.sum(np.multiply(y_train, np.log(pred_y+1e-6)))
	loss_mem[iter] = loss
	
	deri_a = a
	deri_a[a != 0] = 1
	grad_w1 = np.dot(x_train.T, np.multiply(np.dot(pred_y - y_train, weights_2.T), deri_a))
	weights_1 -= LR * grad_w1
	grad_w2 = np.dot(a.T, pred_y - y_train)
	weights_2 -= LR * grad_w2
	
	if iter == 0:
		ma = np.mean(a, 0, dtype=np.float32)
		ma0 = a - ma[None, :]
		cov = np.dot(ma0.T, ma0) / (N - 1)

		U, V = np.linalg.eigh(cov)
		pca_a = np.dot(a, V[:, -2:])

		plt.figure(1)
		plt.plot(pca_a[y_train[:, 0] == 1, 0], pca_a[y_train[:, 0] == 1, 1], 'o')
		plt.plot(pca_a[y_train[:, 1] == 1, 0], pca_a[y_train[:, 1] == 1, 1], 'x')
		plt.title("2D feature 1 iteration")
		plt.legend(["class 1", "class 2"])
		
		pca_a3d = np.dot(a, V[:, -3:])
		fig = plt.figure(3)
		ax = Axes3D(fig)
		ax.scatter(pca_a3d[y_train[:, 0] == 1, 0], pca_a3d[y_train[:, 0] == 1, 1], pca_a3d[y_train[:, 0] == 1, 2], 'o')
		ax.scatter(pca_a3d[y_train[:, 1] == 1, 0], pca_a3d[y_train[:, 1] == 1, 1], pca_a3d[y_train[:, 1] == 1, 2], 'x')
		ax.set_title("3D feature 1 iteration")
		plt.legend(["class 1", "class 2"])
		
	elif iter == ITERATION / 10 * 9:
		ma = np.mean(a, 0, dtype=np.float32)
		ma0 = a - ma[None, :]
		cov = np.dot(ma0.T, ma0) / (N - 1)

		U, V = np.linalg.eigh(cov)
		pca_a = np.dot(a, V[:, -2:])

		plt.figure(2)
		plt.plot(pca_a[y_train[:, 0] == 1, 0], pca_a[y_train[:, 0] == 1, 1], 'o')
		plt.plot(pca_a[y_train[:, 1] == 1, 0], pca_a[y_train[:, 1] == 1, 1], 'x')
		plt.title("2D feature %d iteration" %ITERATION)
		plt.legend(["class 1", "class 2"])
		
		pca_a3d = np.dot(a, V[:, -3:])
		fig = plt.figure(4)
		ax = Axes3D(fig)
		ax.scatter(pca_a3d[y_train[:, 0] == 1, 0], pca_a3d[y_train[:, 0] == 1, 1], pca_a3d[y_train[:, 0] == 1, 2], 'o')
		ax.scatter(pca_a3d[y_train[:, 1] == 1, 0], pca_a3d[y_train[:, 1] == 1, 1], pca_a3d[y_train[:, 1] == 1, 2], 'x')
		ax.set_title("3D feature %d iteration" %ITERATION)
		plt.legend(["class 1", "class 2"])
		
	if (iter+1) % 1000 == 0:
		print("iterations {:6d}: loss {:11.5f}".format(iter+1, loss))

plt.figure()
plt.plot(loss_mem[:2000])

np.save("weights_1", weights_1)
np.save("weights_2", weights_2)

idx = 20
pred_y, a = forward(x_train, weights_1, weights_2)
pred_y[pred_y >= 0.5] = 1
pred_y[pred_y < 0.5] = 0
pred_y = pred_y.astype(np.int8)
# print(pred_y[:idx])
# print(y_train[:idx])
print((np.dot(pred_y[:, 0], y_train[:, 0]) + np.dot(pred_y[:, 1], y_train[:, 1])) , y_train.shape[0])

pred_y, a = forward(test_x, weights_1, weights_2)
pred_y[pred_y >= 0.5] = 1
pred_y[pred_y < 0.5] = 0
pred_y = pred_y.astype(np.int8)
print((np.dot(pred_y[:, 0], test_y[:, 0]) + np.dot(pred_y[:, 1], test_y[:, 1])) , test_y.shape[0])
plt.show()