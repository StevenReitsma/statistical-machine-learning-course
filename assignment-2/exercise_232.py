import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpltools import style
from numpy import unravel_index, inf
np.set_printoptions(threshold=np.inf)

def plot_likelihood():
	x = np.loadtxt('lighthouse_set.csv', delimiter=',')

	k_range = [20]
	for k in k_range:
		alpha_grid = np.linspace(-10, 10, 200)
		beta_grid = np.linspace(0.01, 5, 50)
		meshX, meshY = np.meshgrid(alpha_grid, beta_grid)

		likelihood = k * np.log(meshY/np.pi)

		for kk in range(0, k):
			likelihood = likelihood - np.log(meshY**2 + (x[kk] - meshX)**2)

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(meshX, meshY, likelihood, rstride=1, cstride=1, cmap=cm.YlOrRd, linewidth=0, antialiased=True)

		fig.colorbar(surf, shrink=1, aspect=5)
		plt.show()

def plot_likelihood_as_function_of_k():
	a = []
	b = []
	x = np.loadtxt('lighthouse_set.csv', delimiter=',')

	alpha_grid = np.linspace(-10, 10, 200)
	beta_grid = np.linspace(0.01, 5, 50)
	meshX, meshY = np.meshgrid(alpha_grid, beta_grid)

	for k in range(1, 201):
		likelihood = k * np.log(meshY/np.pi)

		for kk in range(0, k):
			likelihood = likelihood - np.log(meshY**2 + (x[kk] - meshX)**2)

		bb,aa = unravel_index(np.argmax(likelihood), likelihood.shape)
		a.append(alpha_grid[aa])
		b.append(beta_grid[bb])

	plt.xlabel('$k$')
	plt.ylabel('value')

	plt.plot(range(1, 201), a, alpha=0.5, color='red', label='$\\alpha$')
	plt.plot(range(1, 201), b, alpha=0.5, color='blue', label='$\\beta$')
	plt.plot(range(1, 201), np.repeat(7.11950, 200), alpha=1, color='brown', label='$\\alpha_t$')
	plt.plot(range(1, 201), np.repeat(1.24540, 200), alpha=1, color='cyan', label='$\\beta_t$')

	plt.legend()
	
	plt.show()

if __name__ == "__main__":
	plot_likelihood_as_function_of_k()