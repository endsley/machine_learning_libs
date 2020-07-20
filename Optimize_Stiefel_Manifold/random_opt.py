#!/usr/bin/python

import numpy as np

class random_opt:
	def __init__(self, cost_function, gradient_function):
		self.cost_function = cost_function
		self.gradient_function = gradient_function
		self.x_opt = None
		self.cost_opt = None
		self.db = {}


	def run(self, x_init, max_rep=5000):
		d = x_init.shape[0]
		q = x_init.shape[1]

		old_cost = self.cost_function(x_init)
		for i in range(max_rep):
			rand_x = np.random.randn(d,q)


			[new_x, R] = np.linalg.qr(rand_x)		# QR ensures orthogonality

			new_cost = self.cost_function(new_x)

			if new_cost < old_cost:
				old_cost = new_cost
				print(new_cost)
		return self.x_opt	
