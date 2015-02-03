import numpy as np
import matplotlib.pyplot as plt
from mpltools import style
from scipy import misc
from scipy.stats import multivariate_normal
from exercise_21 import readData, readLabels
style.use('ggplot')

def bernoulli(x, mu):
	"""
	This is the pdf for the Bernoulli defined by mu for a certain sample x.
	"""
	return np.product(mu**x * (1-mu)**(1-x))

def loglikelihood(pi, mu, data, K):
	"""
	Computes the log likelihood as a function of the mixing coefficients, mu, sigma and the data.
	"""
	return np.sum(np.log([np.sum([pi[k] * bernoulli(x, mu[k]) for k in range(0, K)]) for x in data]))

def EM(data, K, iterations = 100, min_delta = 1.0e-4):
	"""
	Runs the EM algorithm on data with K classes for a set amount of iterations (default is 100).
	The min_delta parameter sets the minimum log likelihood difference per iteration.
	If the difference between two iterations is smaller than this value, the algorithm is stopped early.
	"""
	# Initialize mu to the mean of the data set +- a random value between 0 and 1
	mu = np.array([np.ravel(np.random.rand(data.shape[1])*0.5 + 0.25) for _ in range(0, K)])

	# Initialize true means (for exercise 2.4)
	mu = true_means(data, readLabels())

	plt.ion()
	f, axes = plt.subplots(ncols=K)
	plt.show()
	plot(mu, f, axes)

	# Initialize equal mixing coefficients
	pi = np.repeat(1. / K, K)

	# Initialize gamma array
	gamma = np.zeros((data.shape[0], K))
	previous_loglikelihood = 0

	for i in range(0, iterations):
		# Check for convergence
		current_loglikelihood = loglikelihood(pi, mu, data, K)
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
			gamma[:, k] = [pi[k] * bernoulli(sample, mu[k]) for sample in data]

		# Normalize the gamma values
		gamma = gamma / np.repeat(np.sum(gamma, 1)[np.newaxis], K, axis=0).T

		# Compute N
		N = np.sum(gamma, 0)

		# M step, compute mu for each class/Gaussian
		for k in range(0, K):
			mu[k] = 1./N[k] * np.dot(gamma[:, k], data)

		# Recompute the mixing coefficients
		pi = N / np.sum(N)

		plot(mu, f, axes)

	# Print the final log likelihood and the final distribution
	print "Final log likelihood: %.2f" % loglikelihood(pi, mu, data, K)
	print "Final distribution (red, blue, purple, green, pink, gray, yellow): %s" % pi

	# Check labels
	checkLabels(gamma, mu, pi)

	# Persist plot
	plt.ioff()
	plt.show()

def checkLabels(gamma, mu, pi):
	labels = readLabels()
	predictions = np.argmax(gamma, 1)

	inpt = raw_input("Enter prototype class labels separated by space, ending in newline: ")
	convert = inpt.split(' ')

	errors = [index for index, pred in enumerate(predictions) if int(convert[pred]) != labels[index]]
	print errors
	print len(errors) / 800.0

	check_custom_digit(convert, mu, K, pi)

def true_means(data, labels):
	means = np.zeros((3, np.size(data, 1)))
	means[0] = np.mean(np.array([image for index, image in enumerate(data) if labels[index] == 2]), 0)
	means[1] = np.mean(np.array([image for index, image in enumerate(data) if labels[index] == 3]), 0)
	means[2] = np.mean(np.array([image for index, image in enumerate(data) if labels[index] == 4]), 0)

	return means

def check_custom_digit(convert, mu, K, pi):
	# Drawn in photoshop, converted to np array here
	custom = misc.imread('custom_image.png')
	custom = np.ravel(custom)

	# Initialize gamma array
	gamma = np.zeros((1, K))

	for k in range(0, K):
		gamma[0, k] = pi[k] * bernoulli(custom, mu[k])

	prediction = np.argmax(gamma)
	print "Prediction was: %s" % convert[prediction]

def plot(mu, f, axes):
	for i, m in enumerate(mu):
		img = axes[i].imshow(m.reshape(28, 28, order='F'))
		img.set_interpolation('nearest')
		img.set_cmap('gray')

	f.canvas.draw()

if __name__ == "__main__":
	data = readData()
	K = 3

	EM(data, K, iterations = 100)