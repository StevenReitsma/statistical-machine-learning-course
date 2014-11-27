import numpy as np
import matplotlib.pyplot as plt
from mpltools import style
style.use('ggplot')

if __name__ == "__main__":
	read = np.loadtxt('irls_data.txt')

	plt.scatter(read[:, 0], read[:, 1], c = read[:, 2], alpha = 0.75)
	plt.show()