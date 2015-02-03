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
	#return multivariate_normal.pdf(x, mu, sigma) # Scipy method: significantly slower than implementation above, results are equal

def loglikelihood(pi, mu, sigma, data):
	"""
	Computes the log likelihood as a function of the mixing coefficients, mu, sigma and the data.
	"""
	return np.sum([np.log(np.sum([pi[k] * gaussian(x, mu[k], sigma[k]) for k in range(0, K)])) for x in data])

def EM(data, K, iterations = 100, min_delta = 1.0e-4):
	"""
	Runs the EM algorithm on data with K classes for a set amount of iterations (default is 100).
	The min_delta parameter sets the minimum log likelihood difference per iteration.
	If the difference between two iterations is smaller than this value, the algorithm is stopped early.
	"""
	# Initialize mu to the mean of the data set +- a random value between 0 and 1
	mu = [np.ravel(np.mean(data, 0) + np.random.rand(1, data.shape[1]) * 2 - 1) for _ in range(0, K)]

	# Initialize sigma to a diagonal matrix with random numbers between 2 and 6 on the diagonals.
	sigma = [np.diagflat(np.random.rand(1, data.shape[1]) * 4 + 2) for _ in range(0, K)]

	# Initialize equal mixing coefficients
	pi = np.repeat(1. / K, K)

	# Initialize gamma array
	gamma = np.zeros((data.shape[0], K))
	previous_loglikelihood = 0

	for i in range(0, iterations):
		# Check for convergence
		current_loglikelihood = loglikelihood(pi, mu, sigma, data)
		if np.abs(previous_loglikelihood - current_loglikelihood) < min_delta:
			# We have converged well enough
			print "Stopping early because of convergence after %i iterations." % i
			break

		# Recompute log likelihood and print it
		previous_loglikelihood = current_loglikelihood
		print "Log likelihood: %.2f" % current_loglikelihood

		# Compute unnormalized gamma values for each class for each data point.
		# We normalize later, because part of the normalization coefficient has to be calculated later on anyway (for the other class).
		# Doing it now would mean we have to calculate it multiple times.
		for k in range(0, K):
			gamma[:, k] = [pi[k] * gaussian(sample, mu[k], sigma[k]) for sample in data]

		# Normalize the gamma values
		gamma = gamma / np.repeat(np.sum(gamma, 1)[np.newaxis], K, axis=0).T

		# Compute N
		N = np.sum(gamma, 0)

		# M step, compute mu and sigma for each class/Gaussian
		for k in range(0, K):
			mu[k] = 1./N[k] * np.dot(gamma[:, k], data)
			sigma[k] = 1./N[k] * np.dot(gamma[:, k], (data - mu[k])[np.newaxis].T * (data - mu[k])[np.newaxis])

		# Recompute the mixing coefficients
		pi = N / np.sum(N)

	# Print the final log likelihood and the final distribution
	print "Final log likelihood: %.2f" % loglikelihood(pi, mu, sigma, data)
	print "Final distribution (red, blue, purple, green, pink, gray, yellow): %s" % pi

	# Compute the correlation for each class
	correlation = [sigma[k][0, 1] / np.sqrt(sigma[k][0, 0] * sigma[k][1, 1]) for k in range(0, K)]
	print "Correlation vector (x1, x2): %s" % correlation

	# Do question five
	question5(pi, mu, sigma, K)

	# Show the plot
	plot(data, gamma)


def question5(pi, mu, sigma, K):
	A = np.array([11.85, 2.2, 0.5, 4.0])
	B = np.array([11.95, 3.1, 0.0, 1.0])
	C = np.array([12.00, 2.5, 0.0, 2.0])
	D = np.array([12.00, 3.0, 1.0, 6.3])
	A_likelihood = [pi[k] * gaussian(A, mu[k], sigma[k]) for k in range(0, K)]
	B_likelihood = [pi[k] * gaussian(B, mu[k], sigma[k]) for k in range(0, K)]
	C_likelihood = [pi[k] * gaussian(C, mu[k], sigma[k]) for k in range(0, K)]
	D_likelihood = [pi[k] * gaussian(D, mu[k], sigma[k]) for k in range(0, K)]

	A_likelihood /= np.sum(A_likelihood)
	B_likelihood /= np.sum(B_likelihood)
	C_likelihood /= np.sum(C_likelihood)
	D_likelihood /= np.sum(D_likelihood)

	print "Likelihoods for sample A: %s" % A_likelihood
	print "Likelihoods for sample B: %s" % B_likelihood
	print "Likelihoods for sample C: %s" % C_likelihood
	print "Likelihoods for sample D: %s" % D_likelihood

def plot(data, gamma):
	# First make some normal colors
	color_cycle = ['#E24A33', '#348ABD', '#988ED5', '#8EBA42', '#FFB5B8', '#777777', '#FBC15E']

	assert gamma.shape[1] <= len(color_cycle)

	plt.scatter(data[:, 0], data[:, 1], c = [color_cycle[x] for x in np.argmax(gamma, 1)], alpha = 0.75)
	plt.show()

if __name__ == "__main__":
	data = np.loadtxt('data/a011_mixdata.txt')
	K = 4

	EM(data, K)