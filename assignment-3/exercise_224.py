import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpltools import style
from scipy.interpolate import Rbf
style.use('ggplot')

def gradient(phi, y, t):
	return np.dot(phi.T, y - t)

def hessian(phi, y):
	R = np.diag(np.squeeze(y * (1-y))) # squeeze out the extra dimension, otherwise np.diag does not work
	return np.dot(phi.T, np.dot(R, phi))

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

def gbf(x, m, s):
	sigma = s * np.identity(np.array(m).shape[0])
	a = (x-m)[np.newaxis].T
	normalizer = 1.0 / (2*np.pi)**(a.shape[0]/2) * 1.0 / np.linalg.det(sigma)**(1/2)
	return normalizer * np.exp(-(1.0/2)*np.dot(a.T, np.dot(inv(sigma), a)))

if __name__ == "__main__":
	read = np.loadtxt('irls_data.txt')
	w = np.array([[0], [0], [0]])
	x = read[:, 0:2]
	t = read[:, 2][np.newaxis].T

	phi = np.array([[1, gbf(xx, (0, 0), 0.3), gbf(xx, (1, 1), 0.3)] for xx in x])


	plt.scatter(phi[:, 1], phi[:, 2], c = t, alpha = 0.75)
	plt.show()