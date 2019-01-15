import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt

MOD = 2**32 - 1
N = 12400

def base_number():
	num = str(time.time())
	result = (int(num[0:10]) * int(num[11:len(num)])) % MOD
	return result


def congruent_method(count):
	first = base_number()
	mas = []
	k = 2321998517
	for _ in range(count):
		num = (first * k) % MOD
		first = num
		mas.append(num / MOD)
	return mas


def fib_delay_method(mas, count):
	for i in range(count):
		next_num = mas[i+33] - mas[i+97]
		if next_num >= 0:
			mas.append(next_num)
		else:
			mas.append(next_num + 1)
	return mas


def statistical_independence(mas):
	i,j = random.randint(0,len(mas)), random.randint(0,len(mas))
	first_num = str(mas[i])[2:]
	second_num = str(mas[j])[2:]
	two_num_str = first_num + second_num
	x2 = equal_uniform_distribution(two_num_str, True)
	print("statistical independence x2 : ", x2)


# def normal():
# 	norm_arr = []
# 	for i in range(300):
# 		z = math.sqrt((-2) * math.log(random.random())) * math.cos(2 * 3.14 * random.random())
# 		norm_arr.append(z)
# 	return norm_arr

	
def equal_uniform_distribution(mas, stat=False):
	len_row = 10
	variation_row = [0] * len_row
	p = 1 / len_row
	mt = p * len(mas)
	x2 = 0


	if stat is True:
		for i in mas:
			j = int(i) % len_row
			variation_row[j] += 1
	else:
		for i in mas:
			j = round(i * 100 ) % len_row
			variation_row[j] += 1
		# if i >= 0 and i < 0.1:
		# 	variation_row[0] += 1
		# if i >= 0.1 and i < 0.2:
		# 	variation_row[1] += 1
		# if i >= 0.2 and i < 0.3:
		# 	variation_row[2] += 1
		# if i >= 0.3 and i < 0.4:
		# 	variation_row[3] += 1
		# if i >= 0.4 and i < 0.5:
		# 	variation_row[4] += 1
		# if i >= 0.5 and i < 0.6:
		# 	variation_row[5] += 1
		# if i >= 0.6 and i < 0.7:
		# 	variation_row[6] += 1
		# if i >= 0.7 and i < 0.8:
		# 	variation_row[7] += 1
		# if i >= 0.8 and i < 0.9:
		# 	variation_row[8] += 1
		# if i >= 0.9 and i < 1:
		# 	variation_row[9] += 1

	for i in variation_row:
		x2 += ((i - mt) ** 2) / mt
	#print(variation_row)
	return x2

def inverse_funtions_method():
	pass


def check_exponential_distribution(crv):
	lambd = 1 / np.mean(crv)
	interval_count = 10
	interval_width = (max(crv) - min(crv)) / interval_count
	#variation_row = [0] * interval_count
	low = 0
	x2 = 0

	for i in range(interval_count):
		hight = low + interval_width
		p = math.exp(-lambd * low) - math.exp(-lambd * hight)
		count_hits = 0

		for num in crv:
			if num >= low and num < hight:
				count_hits += 1

		mt = p * len(crv)
		x2 += ((count_hits - mt) ** 2) / mt
		low = hight

	#print(variation_row)
	return(x2)
	#for i in range(interval_len):




def get_crv(count):
	crv_arr = []
	brv_arr = []
	for _ in range(count):
	 	brv_arr.append(random.random())
	lambd = 1 / np.mean(brv_arr)
	for i in brv_arr:
		crv_arr.append((-1 / lambd) * math.log(i))
	return crv_arr
	# for value in brv_arr:
	#     crv_arr.append((-1 / lambd) * math.log(value))
	# return crv_arr

def check_confidence_interval(mas):
	t = 2
	M = np.mean(mas)
	sigma = math.sqrt(np.var(mas))
	standart_error = sigma / math.sqrt(len(mas))
	delta = standart_error * t
	print("confidence_interval : {} < {} < {}".format(M-delta, M, M+delta))


def rand_mas():
	mas = [0] * (N+98)
	for j in range(1):
		for i in range(len(mas)):
			mas[i] = random.random()

	return mas


def main():
	#first = random.randint(1, 10000)
	#second = random.randint(1, 10000)

	mas = congruent_method(98)
	mas = fib_delay_method(mas, N)
	statistical_independence(mas)

	# x3 = 0
	# c = 0
	# while(x3 < 24):
	# 	c += 1
	# 	mas = rand_mas()
	# 	var_r = equal_uniform_distribution(mas, (N + 98))
	# 	x3 = var_r
	# 	print(c)
	#mas = rand_mas()
	#print(mas)
	x2_brv = equal_uniform_distribution(mas)
	print("uniform distribution  x2 : ", x2_brv)
	print("math brv ", np.mean(mas))
	print("disp brv ", np.var(mas))
	check_confidence_interval(mas)

	crv = get_crv(10000)
	x2_crv = check_exponential_distribution(crv)
	print("exponential distribution  x2 : ", x2_crv)
	print("math crv ", np.mean(crv))
	print("disp crv ", np.var(crv))

	check_confidence_interval(crv)

	#mas2 = how_much(mas)
	#print(mas2)
	#x = np.arange(0, 1 , 0.1);
	#y = mas
	#plt.bar(x, y, width=1, color='green', linewidth=1, edgecolor='black')
	#plt.plot(x, y)
	#plt.show()
	#plt.plot(range(len(mas)), mas, 'o')
	plt.hist(mas, 20)
	plt.show()
	plt.hist(crv, 20)
	plt.show()


if __name__ == "__main__":
	main()