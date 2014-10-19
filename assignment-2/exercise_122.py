import numpy as np

if __name__ == "__main__":
	pairs = np.loadtxt('pairs.csv', delimiter=',')
	mu_ml = 1.0/np.size(pairs, 0) * sum(pairs)
	
	sigma_ml = 1.0/(np.size(pairs, 0)) * np.dot((pairs - mu_ml).T, (pairs - mu_ml))
	print sigma_ml