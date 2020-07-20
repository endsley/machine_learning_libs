#!/usr/bin/python
#	Note : This is designed for Python 3


import numpy as np

class ism:
	def __init__(self, cost_function, compute_Φ):
		self.cost_function = cost_function
		self.compute_Φ = compute_Φ
		self.x_opt = None
		self.cost_opt = None
		self.db = {}


	def eig_solver(self, L, k, mode='smallest'):
		#L = ensure_matrix_is_numpy(L)
		eigenValues,eigenVectors = np.linalg.eigh(L)
	
		if mode == 'smallest':
			U = eigenVectors[:, 0:k]
			U_λ = eigenValues[0:k]
		elif mode == 'largest':
			n2 = len(eigenValues)
			n1 = n2 - k
			U = eigenVectors[:, n1:n2]
			U_λ = eigenValues[n1:n2]
		else:
			raise ValueError('unrecognized mode : ' + str(mode) + ' found.')
		
		return [U, U_λ]

	def run(self, old_x, max_rep=200):
		q = old_x.shape[1]
		new_x = old_x
		old_λ = np.random.randn(1,q)

		for i in range(max_rep):
			Φ = self.compute_Φ(old_x)
			[new_x, new_λ] = self.eig_solver(Φ, q)
			old_x = new_x
			if np.linalg.norm(old_λ - new_λ)/np.linalg.norm(old_λ) < 0.01: break
			old_λ = new_λ

			print('Cost %.3f'%self.cost_function(new_x))
