import numpy as np
import matplotlib.pyplot as plt
from mpltools import style
style.use('ggplot')

def readData():
	"""
	Reads the input binary file.
	"""
	N = 800
	D = 28*28
	X = np.zeros((N, D), dtype=np.uint8)

	f = open("data/a012_images.dat", 'rb')

	for i in range(0, N):
		X[i, :] = np.fromstring(f.read(D), dtype='uint8')

	f.close()

	return X

def readLabels():
	"""
	Reads the labels.
	"""
	N = 800
	X = np.zeros((N, 1), dtype=np.uint8)

	f = open("data/a012_labels.dat", 'rb')

	for i in range(0, N):
		X[i] = np.fromstring(f.read(1), dtype='uint8')

	f.close()

	return X

if __name__ == "__main__":
	X = readData()
	
	img = plt.imshow(X[1, :].reshape(28, 28, order='F')) # Fortran order is required for MATLAB compliance
	img.set_interpolation('nearest')
	img.set_cmap('gray')
	plt.show()