import numpy as np

def generate_random_number_pair(mu, sigma):
	return np.random.multivariate_normal(mu, sigma, 1)

if __name__ == "__main__":
	print generate_random_number_pair([0.8, 0.8], [[0.1, -0.1], [-0.1, 0.12]])