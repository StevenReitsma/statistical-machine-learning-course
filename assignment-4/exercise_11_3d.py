import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpltools import style
style.use('ggplot')

def plot(data):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = data[:, 3], alpha = 0.5)

	plt.show()

if __name__ == "__main__":
	data = np.loadtxt('data/a011_mixdata.txt')
	plot(data)