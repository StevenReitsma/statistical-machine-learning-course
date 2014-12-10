import numpy as np
import matplotlib.pyplot as plt
from mpltools import style
from scipy.stats import multivariate_normal
style.use('ggplot')

def gaussian(x, mu, sigma):
	"""
	This is the pdf for the Gaussian defined by mu and sigma for a certain sample x.
	"""
	normalizer = 1. / (2*np.pi)**(x.shape[0]/2) * 1. / np.sqrt(np.linalg.det(sigma))
	return normalizer * np.exp(-1./2 * np.dot(x - mu, np.dot(np.linalg.inv(sigma), x - mu)))
	#return multivariate_normal.pdf(x, mu, sigma) # significantly slower than implementation above, results are equal

def loglikelihood(pi, mu, sigma, data):
	"""
	Computes the log likelihood as a function of the mixing coefficients, mu, sigma and the data.
	"""
	return np.sum([np.log(np.sum([pi[k] * gaussian(x, mu[k], sigma[k]) for k in range(0, K)])) for x in data])

def EM(data, K, iterations = 100):
	mu = [np.ravel(np.mean(data, 0) + np.random.rand(1, data.shape[1]) * 2 - 1) for _ in range(0, K)]
	sigma = [np.diagflat(np.random.rand(1, data.shape[1]) * 4 + 2) for _ in range(0, K)]
	pi = np.repeat(1.0/K, K)

	for i in range(0, iterations):
		# gamma = np.zeros((data.shape[0], K))
		# for k in range(0, K):
		# 	gamma[:, k] = [pi[k] * gaussian(sample, mu[k], sigma[k]) for sample in data]

		# # Normalize the gamma values
		# gamma = gamma / np.repeat(np.sum(gamma, 1)[np.newaxis], K, axis=0).T

		# # Compute N
		# N = np.sum(gamma, 0)

		# for k in range(0, K):
		# 	mu[k] = 1./N[k] * np.dot(gamma[:, k], data)
		# 	sigma[k] = 1./N[k] * np.dot(gamma[:, k], (data - mu[k])[np.newaxis].T * (data - mu[k])[np.newaxis])

		# pi = N / np.sum(N)

		# E STEP
		gamma = np.zeros((data.shape[0], K))
		sample_id = 0

		print loglikelihood(pi, mu, sigma, data)

		for sample in data:
			denom = np.sum([pi[j] * gaussian(sample, mu[j], sigma[j]) for j in range(0, K)])
			gamma[sample_id, :] = [pi[k] * gaussian(sample, mu[k], sigma[k]) / denom for k in range(0, K)]

			# Validation to check correctness of implementation
			if np.abs(np.sum(gamma[sample_id, :]) - 1) > 0.00001:
				raise ValueError("Gamma values should sum up to 1, sum is %.5f" % np.sum(gamma[sample_id, :]))

			sample_id += 1

		# M STEP
		N = [np.sum(gamma[:, k]) for k in range(0, K)]
		mu = np.array([1./N[k] * np.sum([gamma[n, k] * data[n, :] for n in range(0, data.shape[0])], 0) for k in range(0, K)])
		sigma = [1./N[k] * np.sum([gamma[n, k] * (data[n, :] - mu[k, :])[np.newaxis].T * (data[n, :] - mu[k, :])[np.newaxis] for n in range(0, data.shape[0])], 0) for k in range(0, K)]
		pi = N/np.sum(N)

	print "Final log likelihood: %.2f" % loglikelihood(pi, mu, sigma, data)
	print "Final distribution: %s" % pi
	plot(data, gamma)

def plot(data, gamma):
	plt.scatter(data[:, 0], data[:, 1], c = np.argmax(gamma, 1), alpha = 0.75)
	plt.show()

if __name__ == "__main__":
	data = np.loadtxt('data/a011_mixdata.txt')
	K = 2

	EM(data, K)