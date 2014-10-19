import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpltools import style

def plot_probability_density(mu, sigma):
	x, y = np.mgrid[-0.5:2:.025, -0.5:2:.025]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x
	pos[:, :, 1] = y

	pdf = multivariate_normal.pdf(pos, mu, sigma)
	print pdf

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(x, y, pdf, rstride=1, cstride=1, cmap=cm.YlOrRd, linewidth=0, antialiased=True)

	fig.colorbar(surf, shrink=1, aspect=5)
	plt.show()

if __name__ == "__main__":
	plot_probability_density([0.8, 0.8], [[0.1, -0.1], [-0.1, 0.12]])