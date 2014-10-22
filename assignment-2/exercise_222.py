import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpltools import style
style.use('ggplot')
import scipy.stats as st


alpha = 7.11950
beta = 1.24540

def generate_set():
	x = []
	for i in range(0, 200):
		rand_theta = np.random.uniform(-np.pi/2, np.pi/2)
		x.append(beta * np.tan(rand_theta) + alpha)

	np.savetxt('lighthouse_set.csv', x, delimiter=',')
	return x

if __name__ == "__main__":
	dist = generate_set()

	plt.hist(dist, bins = 100)
	mean = []

	for n in range(1, 201):
		mean.append(np.mean(dist[0:n]))

	#plt.plot(range(1, 201), mean, alpha=0.5, color='red')
	#plt.plot(range(1, 201), np.repeat(alpha, 200), alpha=0.5, color='blue')

	#plt.xlabel('# of data points')
	#plt.ylabel('Mean of the data points')

	plt.show()
