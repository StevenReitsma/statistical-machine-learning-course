import numpy as np
import prettyplotlib as ppl
import matplotlib.pyplot as plt
import exercise_11

# The definition of the A-matrix
def A(i, j, inpt):
	return np.sum([x[0]**(i+j) for x in inpt])

# The definition of the T-vector
def T(i, inpt):
	return np.sum([x[1] * (x[0]**i) for x in inpt])

# Returns weights for the optimal polynomial curve fit.
def PolCurFit(inpt, order, labda = 0):
	order = order + 1
	
	# Build the A-matrix
	A_matrix = np.array([[A(i, j, inpt) for j in np.arange(order)] for i in np.arange(order)])

	# Regularize, which is as simple as adding the lambda term to the A_matrix diagonal
	A_matrix = A_matrix + labda * np.identity(np.size(A_matrix, 0))
	
	# Build the T-vector
	T_vector = np.array([T(i, inpt) for i in np.arange(order)])

	# Solve the linear system
	return np.linalg.solve(A_matrix, T_vector)

# Returns y(x; w). x is scalar, w is a vector.
def eval_polynomial(w, x):
	return np.sum([w[i]*(x**i) for i in np.arange(0, np.size(w))])

def run_on_data():
	x_training = np.linspace(0, 1, 40)
	y_training = exercise_11.sample_gaussian(exercise_11.f, 0, 0.3, x_training)
	
	# Loop through polynomial orders.
	weights = []
	for i in range(0, 10):
		# Zip x and y pairs together, fit a curve.
		weights.append(PolCurFit(zip(x_training, y_training), i, labda=10**-6))

	plot([weights[0], weights[1], weights[3], weights[9]])
	plot_rmse(y_training, weights)

# Plot polynomial of order 0, 1, 3 and 9.
def plot(weights_vector):
	# Use a temporal resolution of 1000.
	x_real = np.linspace(0, 1, 1000)
	y_real = np.array([exercise_11.f(x_sample) for x_sample in x_real])
	
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

	fig.savefig('exercise_polynomial_plot40.pdf')

# Plots the root mean squared error of the polynomials given by weights_vector.
# Shows the error on the training set, given by y_training, and on the test set, which is generated on-the-fly.
def plot_rmse(y_training, weights_vector):
	x_test = np.linspace(0, 1, 100)
	y_test = exercise_11.sample_gaussian(exercise_11.f, 0, 0.3, x_test)

	rmse_training = np.zeros((len(weights_vector)))
	rmse_test = np.zeros((len(weights_vector)))
	i = 0
	for weights in weights_vector:
		poly_output_100 = [eval_polynomial(weights, x) for x in np.linspace(0, 1, 100)]
		poly_output_40 = [eval_polynomial(weights, x) for x in np.linspace(0, 1, 40)]
		rmse_training[i] = np.sqrt(np.mean((poly_output_40 - y_training)**2))
		rmse_test[i] = np.sqrt(np.mean((poly_output_100 - y_test)**2))
		i = i + 1

	fig, ax = plt.subplots(1)
	ppl.plot(ax, np.arange(10), rmse_training, linewidth=0.75, label='RMSE on training set')	
	ppl.plot(ax, np.arange(10), rmse_test, linewidth=0.75, label='RMSE on test set')

	ppl.legend(ax, loc='upper right', ncol=2)
	ax.set_xlabel('Polynomial order')
	ax.set_ylabel('RMSE')
	ax.set_title('RMSE for the polynomial approximation of sine function')

	fig.savefig('exercise_rmse_plot40.pdf')

def plot_lambda_effect():
	x_training = np.linspace(0, 1, 40)
	y_training = exercise_11.sample_gaussian(exercise_11.f, 0, 0.3, x_training)
	
	# Loop through lambda's.
	weights_vector = []
	for l in range(-5, 15):
		# Zip x and y pairs together, fit a curve.
		weights_vector.append(PolCurFit(zip(x_training, y_training), 9, labda=10**-l))

	x_test = np.linspace(0, 1, 100)
	y_test = exercise_11.sample_gaussian(exercise_11.f, 0, 0.3, x_test)

	rmse_training = np.zeros((len(weights_vector)))
	rmse_test = np.zeros((len(weights_vector)))
	i = 0
	weights_vector = weights_vector[::-1]
	for weights in weights_vector:
		poly_output_100 = [eval_polynomial(weights, x) for x in np.linspace(0, 1, 100)]
		poly_output_40 = [eval_polynomial(weights, x) for x in np.linspace(0, 1, 40)]
		rmse_training[i] = np.sqrt(np.mean((poly_output_40 - y_training)**2))
		rmse_test[i] = np.sqrt(np.mean((poly_output_100 - y_test)**2))
		i = i + 1

	fig, ax = plt.subplots(1)

	ppl.plot(ax, np.arange(-14, 6), rmse_training, linewidth=0.75, label='RMSE on training set')	
	ppl.plot(ax, np.arange(-14, 6), rmse_test, linewidth=0.75, label='RMSE on test set')

	ppl.legend(ax, loc='upper right', ncol=2)
	ax.set_xlabel('$log_{10} \lambda$')
	ax.set_ylabel('RMSE')
	ax.set_title('RMSE for the polynomial approximation of sine function')

	fig.savefig('exercise_lambda_rmse_plot40.pdf')

run_on_data()
plot_lambda_effect()