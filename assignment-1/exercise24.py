import numpy as np

def rule_x(x, y):
	return 2*(200*x**3 - 200*x*y + x - 1)

def rule_y(x, y):
	return 200*(y - x**2)

def gradient_descent(x_init, y_init, eta):
	x = x_init
	y = y_init

	x_prev = 0
	y_prev = 0

	its = 0
	first = True

	while first or np.abs(x_prev - x) > 10**-20 or np.abs(y_prev - y) > 10**-20:
		x_prev = x
		y_prev = y

		x = x - eta * rule_x(x, y)
		y = x - eta * rule_y(x, y)

		its = its + 1
		first = False

	print 'X = ' + str(x)
	print 'Y = ' + str(y)
	print 'Iterations = ' + str(its)

gradient_descent(0, 0, 0.001)