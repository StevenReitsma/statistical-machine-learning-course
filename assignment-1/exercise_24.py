import numpy as np
import matplotlib.pyplot as plt
from mpltools import style
style.use('ggplot')

# The derivative to x.
def rule_x(x, y):
	return 2*(200*x**3 - 200*x*y + x - 1)

# The derivative to y.
def rule_y(x, y):
	return 200*(y - x**2)

# Performs gradient descent and plots the trajectory on a contour color mesh.
def gradient_descent(x_init, y_init, eta):
	x = x_init
	y = y_init

	x_prev = 0
	y_prev = 0

	its = 0
	first = True

	point_list = []
	point_list.append((x, y))

	# While we have not yet converged.
	while first or np.abs(x_prev - x) > 10**-20 or np.abs(y_prev - y) > 10**-20:
		x_prev = x
		y_prev = y

		# Gradient descent rules.
		x = x_prev - eta * rule_x(x_prev, y_prev)
		y = y_prev - eta * rule_y(x_prev, y_prev)

		point_list.append((x, y))

		its = its + 1
		first = False

	print 'X = ' + str(x)
	print 'Y = ' + str(y)
	print 'Iterations = ' + str(its)

	plot_trajectory(point_list)

# Plots the gradient descent trajectory on a contour color mesh.
def plot_trajectory(point_list):
	fig, ax = plt.subplots(1)

	X = np.arange(-1, 1.1, 0.01)
	Y = np.arange(-0.1, 1.1, 0.01)
	X, Y = np.meshgrid(X, Y)

	Z = 100*(Y-X**2)**2 + (1-X)**2

	ax.contour(X, Y, Z, 100, linewidths = 0.5, colors='k')
	plt.pcolormesh(X, Y, Z, cmap = plt.get_cmap('rainbow_r'))
	ax.plot([point[0] for point in point_list], [point[1] for point in point_list], color='#FFFF4C')

	ax.annotate('Init', (point_list[0][0]+0.01, point_list[0][1] + 0.01))
	ax.annotate('1,1', (point_list[-1][0]+0.01, point_list[-1][1] + 0.01))
	plt.colorbar()

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title(r'Gradient descent trajectory with $\eta = 0.001$')

	fig.savefig('trajectory.png', dpi=300)

gradient_descent(-0.5, 0.5, 0.001)
