from math import cos, sin

def next(x):
	return x + cos(x)/sin(x)

if __name__ == "__main__":
	x = -1
	print x
	for i in range(0, 5):
		x = next(x)
		print x