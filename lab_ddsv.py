import matplotlib.pyplot as plt
import numpy as np
import math
import lab1
from mpl_toolkits.mplot3d import Axes3D
from time import time
from scipy import stats
import mmod_lab

r = 100
X2 = 0

def histogram3d(ddrv):
	ddrv_x, ddrv_y = zip(*ddrv)
	freq_x = lab1.get_relative_frequency(ddrv_x)
	freq_y = lab1.get_relative_frequency(ddrv_y)
	m, n = len(freq_x), len(freq_y)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	xpos = [i for i in range(n)] * m
	ypos = []
	for i in range(0, m):
		ypos += ([j for j in range(n)][m-i:] + [j for j in range(n)][:m-i])
	zpos = [0 for i in range(n*m)]

	dx = np.ones(n*m)
	dy = np.ones(n*m)
	print("QWEQWEQW  ", xpos, "!!!!!!!!   ", ypos)
	dz = [freq_x[f[0]] * freq_y[f[1]] for f in zip(xpos, ypos)]

	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='red')
	plt.show()


def gen_ddsv(brv, p_matrix):

	rows, columns = np.shape(p_matrix)
	Fx, Fxy = [], []
	xk = [0 for i in range(rows)]
	for row in range(rows):
		for col in range(columns):
			xk[row] += p_matrix[row][col]
		Fx.append(sum(xk))


	global X2
	X2 = mmod_lab.get_value()
	conditional_probability = 0
	for row in range(rows):
		Fxy.append([])
		for col in range(columns):
			try:
				conditional_probability += (p_matrix[row][col] / xk[row])
			except ZeroDivisionError:
				pass
			Fxy[row].append(conditional_probability)
		conditional_probability = 0


	ddrv_x, ddrv_y = [], []
	for value in brv:
		i = 0
		for k, Fk in enumerate(Fx):
			if value < Fk:
				ddrv_x.append(k)
				i = k
				break
		for j, Fij in enumerate(Fxy[i]):
			if value < Fij:
				ddrv_y.append(j)
				break

	return list(zip(ddrv_x, ddrv_y))


def get_emperical_matrix(ddrv, shape):
	m, n = shape
	p_emp_matrix = [[0] * n] * m

	s = 0
	for rv in ddrv:
		x, y = rv
		p_emp_matrix[x][y] += 1.0/(len(ddrv))
	print(p_emp_matrix)
	print(s)


def geometric_distribution(k, arg):
	p = arg
	q = 1 - p
	return q**k * p

def even_distribution(k, arg):
	r = arg
	return 1/r


def puasson_distribution(k, arg):
	lambd = arg
	return (lambd**k * math.e**-lambd) / math.factorial(k)


def equal_uniform_distribution(mas, stat=False):
	len_row = 10
	variation_row = [0] * len_row
	p = 1 / len_row
	mt = p * len(mas)
	x2 = 0

	if stat is True:
			return X2
	else:
		for n,i in enumerate(mas):
			j = i % len_row
			variation_row[j] += 1

	for n,i in enumerate(variation_row):
		x2 += ((i - mt) ** 2) / mt
	return x2

def get_drv(brv, distribution, arg):
	Fx = [0 for k in range(r)]
	p = 0

	for k in range(r):
		p += distribution(k, arg)
		Fx[k] = p

	drv = []

	for value in brv:
		for k, Fk in enumerate(Fx):
			if value < Fk:
				drv.append(k)
				break
	return drv



if __name__ == "__main__":
	N = 124000

	mas = mmod_lab.congruent_method(98)
	bsv = mmod_lab.fib_delay_method(mas, N)
	p_matrix = [[0.01, 0.03, 0.04, 0.01],
	[0.03, 0.04, 0.12, 0.10],
	[0.20, 0.01, 0.02, 0.03],
	[0.09, 0.02, 0.03, 0.10],
	[0.03, 0.03, 0.03, 0.03]]
	ddsv = gen_ddsv(bsv, p_matrix)



	#dsv = get_drv(bsv, puasson_distribution, 50)
	#dsv = get_drv(bsv, even_distribution, r)
	dsv = get_drv(bsv, geometric_distribution, 0.12)
	x2_dsv = equal_uniform_distribution(dsv, True)
	print("x2 = ", x2_dsv)
	print("math drv ", np.mean(dsv))
	print("disp drv ", np.var(dsv))
	interval_math_exp, interval_dispersion = lab1.get_interval_estimates(np.mean(dsv), np.var(dsv))
	print('confidence_interval D :{} < {} < {}\n'.format(interval_dispersion[0], np.var(dsv),  interval_dispersion[1]))

	mmod_lab.check_confidence_interval(dsv)
	plt.hist(dsv, 30)
	plt.show()

	histogram3d(ddsv)


	ddsv_x, ddsv_y = zip(*ddsv)
	math_exp_x = np.mean(ddsv_x)
	math_exp_y = np.mean(ddsv_y)
	
	interval_math_exp_x, interval_dispersion_x =  lab1.get_interval_estimates(np.mean(ddsv_x), np.var(ddsv_x))
	interval_math_exp_y, interval_dispersion_y =  lab1.get_interval_estimates(np.mean(ddsv_y), np.var(ddsv_y))
	print('Mx  {} <= {} <= {}'.format(interval_math_exp_x[0], math_exp_x, interval_math_exp_x[1]))
	print('My  {} <= {} <= {}\n'.format(interval_math_exp_y[0],math_exp_y, interval_math_exp_y[1]))




	#get_emperical_matrix(ddsv, np.shape(p_matrix))
	# ddsv_x, ddsv_y = zip(*ddsv)
	# plt.hist(ddsv_x, 10)
	# plt.show()
	# plt.hist(ddsv_y, 10)
	# plt.show()

	# ddsv_x_freq = lab1.get_relative_frequency(ddsv_x)
	# ddsv_y_freq = lab1.get_relative_frequency(ddsv_y)
	# lab1.draw_histohram(ddsv_x_freq)
	# lab1.draw_histohram(ddsv_y_freq)



