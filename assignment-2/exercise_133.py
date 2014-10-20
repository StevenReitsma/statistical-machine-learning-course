import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpltools import style
style.use('ggplot')

def sequential_mean(pairs):
	mu = np.zeros((np.size(pairs, 0)+1, np.size(mu_p, 0)))
	mu[0] = 0
	for i in range(1, np.size(pairs, 0)+1):
		mu[i] = mu[i-1] + 1.0 / i * (pairs[i-1] - mu[i-1])
	
	return mu

def sequential_MAP(pairs, mu_p, mu_t, sigma_p, sigma_t):
	mu = np.zeros((np.size(pairs, 0)+1, np.size(mu_p, 0)))
	mu[0] = mu_p

	sigma = np.zeros((np.size(pairs, 0)+1, np.size(sigma_p, 0), np.size(sigma_p, 1)))
	sigma[0] = sigma_p

	for i in range(1, np.size(pairs, 0)+1):
		S = np.linalg.inv(np.linalg.inv(sigma[i-1]) + np.linalg.inv(sigma_t))
		mu[i] = np.dot(S, (np.dot(np.linalg.inv(sigma_t), pairs[i-1]) + np.dot(np.linalg.inv(sigma[i-1]), mu[i-1])))
		sigma[i] = S

		print mu[i]

	plot(pairs, mu, mu_t)

def plot(pairs, mu, mu_t):
	ml = sequential_mean(pairs)

	a = plt.plot(np.linspace(0, 1000, 1001), np.repeat(mu_t[0], 1001), alpha=0.5, color='red', label='$\mu_{t1}$')
	b = plt.plot(np.linspace(0, 1000, 1001), np.repeat(mu_t[1], 1001), alpha=0.5, color='orange', label='$\mu_{t2}$')
	c = plt.plot(np.linspace(0, 1000, 1001), mu[:, 0], 'b-', alpha=0.5, label='MAP for $\mu_{t1}$')
	d = plt.plot(np.linspace(0, 1000, 1001), mu[:, 1], 'b-', alpha=1, label='MAP for $\mu_{t2}$', linestyle='dotted')
	e = plt.plot(np.linspace(0, 1000, 1001), ml[:, 0], 'g-', alpha=0.5, label='ML for $\mu{t1}$')
	f = plt.plot(np.linspace(0, 1000, 1001), ml[:, 1], 'g-', alpha=1, label='ML for $\mu{t2}$', linestyle='dotted')

	plt.legend()

	axes = plt.gca()
	axes.set_xlim([0,1000])
	axes.set_ylim([0,1.5])

	plt.show()

if __name__ == "__main__":
	pairs = np.loadtxt('pairs.csv', delimiter=',')

	mu_p = np.array([0.8, 0.8])
	mu_t = np.array([0.78848608, 0.87686679])
	sigma_p = np.array([[0.1, -0.1], [-0.1, 0.12]])
	sigma_t = np.array([[2.0, 0.8], [0.8, 4.0]])

	sequential_MAP(pairs, mu_p, mu_t, sigma_p, sigma_t)