import numpy as np

def sequential_mean(pairs):
	mu = 0
	for i in range(1, np.size(pairs, 0)+1):
		mu = mu + 1.0 / i * (pairs[i-1] - mu)
		print mu

if __name__ == "__main__":
	pairs = np.loadtxt('pairs.csv', delimiter=',')
	sequential_mean(pairs)