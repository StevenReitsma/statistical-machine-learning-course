import numpy as np
from numpy.linalg import inv

def gradient(phi, y, t):
	return np.dot(phi.T, y - t)

def hessian(phi, y):
	R = np.diag(np.squeeze(y * (1-y))) # squeeze out the extra dimension, otherwise np.diag does not work
	return np.dot(phi.T, np.dot(R, phi))

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

if __name__ == "__main__":
	w = np.array([[1], [1]])
	x = np.array([0.3, 0.44, 0.46, 0.6])
	t = np.array([[1], [0], [1], [0]])
	phi = np.array([[1, xx] for xx in x])

	for i in range(0, 10):
		y = sigmoid(np.dot(phi, w))

		cur_gradient = gradient(phi, y, t)
		cur_hessian = hessian(phi, y)

		w = w - np.dot(inv(cur_hessian), cur_gradient)

		print w