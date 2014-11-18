import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpltools import style
import matplotlib.patches as mpatches
style.use('ggplot')

def phi(x):
	return np.array(((1), (x)))

def m(x):
	m_n = np.array(((-0.0445), (-0.2021)))

	return np.dot(m_n.T, phi(x))

def s2(x):
	S_N_inv = np.array(((22, 10), (10, 7.2)))

	return 1/10.0 + np.dot(phi(x).T, np.dot(inv(S_N_inv), phi(x)))

def plot():
	x_range = np.linspace(0, 1, 100)
	m_vals = np.array([m(x) for x in x_range])
	s2_vals = np.array([s2(x) for x in x_range])

	plt.plot(x_range, m_vals, alpha = 0.5, color = 'blue')
	plt.fill_between(x_range, m_vals+s2_vals, m_vals-s2_vals, facecolor='red', alpha=0.5)
	plt.scatter([0.4, 0.6], [0.05, -0.35], s = 75, c = 'black', alpha = 0.5)

	# 5 samples
	rands = np.random.multivariate_normal((-0.0445, -0.2021), inv(((22, 10), (10, 7.2))), 5)
	
	for r in rands:
		x_range = np.linspace(0, 1, 100)
		m_vals = np.array([np.dot(r.T, phi(x)) for x in x_range])
		plt.plot(x_range, m_vals, alpha = 0.5, color='green')

	plt.xlabel('$x$')
	plt.ylabel('$m(x)$')

	x1,x2,y1,y2 = plt.axis()
	plt.axis((0, 1, y1, y2))

	plt.show()

if __name__ == "__main__":
	plot()