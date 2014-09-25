import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt

# The signal to be plotted.
def f(x):
	return 1 + np.sin(6*(x-2))

# Adds Gaussian noise to a signal f(x). x is a vector.
def sample_gaussian(f, mean, std, x):
	noise = np.random.normal(mean, std, np.size(x))
	y = [f(x_sample) for x_sample in x]
	return noise + y

# Plots the original signal and the noisy signal.
def plot():
	# Use 10 sample points for the noisy signal.
	x_training = np.linspace(0, 1, 10)
	y_training = sample_gaussian(f, 0, 0.3, x_training)

	# Use 1000 sample points for the real signal. Needs to be higher for higher resolution images.
	x_real = np.linspace(0, 1, 1000)
	y_real = [f(x_sample) for x_sample in x_real]

	fig, ax = plt.subplots(1)
	ppl.plot(ax, x_training, y_training, '-o', linewidth=0.75, label='Observations')
	ppl.plot(ax, x_real, y_real, linewidth=0.75, label='Function')

	ppl.legend(ax, loc='upper right', ncol=2)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('Noisy observations versus real function')

	fig.savefig('exercise_11.pdf')

plot()