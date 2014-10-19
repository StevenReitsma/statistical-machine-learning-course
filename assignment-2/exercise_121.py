import numpy as np

def generate_random_number_pair(mu, sigma):
	return np.random.multivariate_normal(mu, sigma, 1000)

if __name__ == "__main__":
	np.savetxt('pairs.csv', generate_random_number_pair([0.78848608, 0.87686679], [[2.0, 0.8], [0.8, 4.0]]), delimiter=',')