import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt

def f(x):
	return 1 + np.sin(6*(x-2))

def sample_gaussian(f, mean, std, x):
	noise = np.random.normal(mean, std, np.size(x))
	y = [f(x_sample) for x_sample in x]
	return noise + y

def plot():
	x_training = np.linspace(0, 1, 10)
	y_training = sample_gaussian(f, 0, 0.3, x_training)
	x_test = np.linspace(0, 1, 100)
	y_test = sample_gaussian(f, 0, 0.3, y_training)

	x_real = np.linspace(0, 1, 1000)
	y_real = [f(x_sample) for x_sample in x_real]

	fig, ax = plt.subplots(1)
	ppl.plot(ax, x_training, y_training, linewidth=0.75, label='Observations')
	ppl.plot(ax, x_real, y_real, linewidth=0.75, label='Function')

	ppl.legend(ax, loc='upper right', ncol=2)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('Noisy observations versus real function')

	fig.savefig('test.png')