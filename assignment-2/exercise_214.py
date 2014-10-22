import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpltools import style
style.use('ggplot')

def plot(beta):
	D = [4.8, -2.7, 2.2, 1.1, 0.8, -7.3]
	D = [-1, 0, 1, -2, 2]
	n = 1000000
	y = np.zeros(n)
	i = 0
	x_mesh = np.linspace(-5, 5, n)

	for a in x_mesh:
		y[i] = np.product([beta / np.pi * (1 / ((x-a)**2 + beta**2)) for x in D])
		i = i + 1

	a_hat = np.argmax(y)
	
	print x_mesh[a_hat]

	plt.xlabel('$\\alpha$')
	plt.ylabel('$p(\\alpha | \mathcal{D}, \\beta = 1)$')

	plt.plot(x_mesh, y, alpha=0.5, color='red')

	plt.show()

def plot_single():
	n = 1000000
	x_mesh = np.linspace(-10, 20, n)
	alpha = 7.11950
	beta = 1.24540
	y = [beta / (np.pi * (beta**2 + (x - alpha)**2)) for x in x_mesh]
	y = y

	plt.xlabel('$x_k$')
	plt.ylabel('$p(x_k \\vert \\alpha, \\beta)$')
	plt.plot(x_mesh, y, alpha=0.5, color='red')
	plt.show()

if __name__ == "__main__":
	plot_single()