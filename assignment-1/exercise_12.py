import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt
import exercise_11

def A(i, j, inpt):
	return np.sum([x[0]**(i+j) for x in inpt])

def T(i, inpt):
	return np.sum([x[1] * (x[0]**i) for x in inpt])

def PolCurFit(inpt, order):
	order = order + 1
	A_matrix = np.array([[A(i, j, inpt) for j in np.arange(order)] for i in np.arange(order)])
	T_vector = np.array([T(i, inpt) for i in np.arange(order)])

	return np.linalg.solve(A_matrix, T_vector)

def eval_polynomial(w, x):
	return np.sum([w[i]*(x**i) for i in np.arange(0, np.size(w))])

def run_on_data():
	x_training = np.linspace(0, 1, 10)
	y_training = exercise_11.sample_gaussian(exercise_11.f, 0, 0.3, x_training)
	weights0 = PolCurFit(zip(x_training, y_training), 0)
	weights1 = PolCurFit(zip(x_training, y_training), 1)
	weights3 = PolCurFit(zip(x_training, y_training), 3)
	weights9 = PolCurFit(zip(x_training, y_training), 9)

	plot([weights0, weights1, weights3, weights9])


def plot(weights_vector):
	x_real = np.linspace(0, 1, 1000)
	y_real = [exercise_11.f(x_sample) for x_sample in x_real]
	fig, ax = plt.subplots(1)
	ppl.plot(ax, x_real, y_real, linewidth=0.75, label='Function')

	i = 0
	which = [0, 1, 3, 9]
	for weights in weights_vector:
			poly_output = [eval_polynomial(weights, x) for x in x_real]
			ppl.plot(ax, x_real, poly_output, linewidth=0.75, label='M = ' + str(which[i]))
			i = i + 1

	ppl.legend(ax, loc='upper right', ncol=2)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('Polynomial approximation of sine function')

	fig.savefig('exercise13.png')

run_on_data()