import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpltools import style
style.use('ggplot')

def plot():
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.arange(-2, 2, 0.1)
	Y = np.arange(-1, 3, 0.1)
	X, Y = np.meshgrid(X, Y)
	R = 100*(Y-X**2)**2 + (1-X)**2
	surf = ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap=cm.coolwarm,
	        linewidth=0, antialiased=True)

	fig.colorbar(surf, shrink=1, aspect=5)
	fig.savefig('exercise21.png')



plot()