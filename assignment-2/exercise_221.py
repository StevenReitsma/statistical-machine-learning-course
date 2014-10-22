import numpy as np

def sample_random_position(interval_coast, interval_sea):
	print np.random.uniform(interval_coast[0], interval_coast[1])
	print np.random.uniform(interval_sea[0], interval_sea[1])

if __name__ == "__main__":
	sample_random_position([0, 10], [1, 2])