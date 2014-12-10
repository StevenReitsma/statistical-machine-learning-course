import numpy as np
import matplotlib.pyplot as plt
from mpltools import style
style.use('ggplot')

def plot(data):
	plt.scatter(data[:, 0], data[:, 1], c = data[:, 2] + data[:, 3], alpha = 0.75)

	plt.show()

if __name__ == "__main__":
	data = np.loadtxt('data/a011_mixdata.txt')
	plot(data)