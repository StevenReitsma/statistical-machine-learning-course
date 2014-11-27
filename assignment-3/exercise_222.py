import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpltools import style
style.use('ggplot')

def gradient(phi, y, t):
	return np.dot(phi.T, y - t)

def hessian(phi, y):
	R = np.diag(np.squeeze(y * (1-y))) # squeeze out the extra dimension, otherwise np.diag does not work
	return np.dot(phi.T, np.dot(R, phi))

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

def error(t, y):
	s = 0
	for n in range(0, t.shape[0]):
		s += t[n] * np.log(y[n]) + (1-t[n]) * np.log(1-y[n])

	return -s

if __name__ == "__main__":
	read = np.loadtxt('irls_data.txt')
	w = np.array([[0], [0], [0]])
	x = read[:, 0:2]
	t = read[:, 2][np.newaxis].T

	phi = np.array([[1, xx[0], xx[1]] for xx in x])

	for i in range(0, 1000):
		y = sigmoid(np.dot(phi, w))

		cur_gradient = gradient(phi, y, t)
		cur_hessian = hessian(phi, y)

		w = w - np.dot(inv(cur_hessian), cur_gradient)

	print "Cross entropy error: ", error(t, y)
	print "Weights: ", w

	y = (y - np.min(y))
	y = y / np.max(y)

	cmap = plt.get_cmap('Reds')
	plt.scatter(x[:, 0], x[:, 1], c = np.squeeze(cmap(y)), alpha = 0.75)
	plt.show()