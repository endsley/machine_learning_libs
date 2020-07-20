#!/usr/bin/python3
#	Note : This is designed for Python 3


import numpy as np

class orthogonal_optimization:
	def __init__(self, cost_function, gradient_function):
		self.cost_function = cost_function
		self.gradient_function = gradient_function
		self.x_opt = None
		self.cost_opt = None
		self.db = {}

	def calc_A(self, x):
		G = self.gradient_function(x)
		A = G.dot(x.T) - x.dot(G.T)
		return [A,G]

	def compute_gradient(self, x):
		[A,G] = self.calc_A(x)
		return A.dot(x)

	#	Applying Sherman-Morrison-Woodbury Theorem ( A faster way to update instead of recalculating inverse )
	def constant_update_inv(self, x, G, M_inv, α_Δ):
		if α_Δ == 0: return M_inv
		d = x.shape[1]
		I = np.eye(d)

		#	1st update
		U = α_Δ*G
		V = x
		E = np.linalg.inv(I + V.T.dot(M_inv).dot(U))
		M_inv = M_inv - M_inv.dot(U).dot(E).dot(V.T).dot(M_inv)

		#	2nd update
		U = -α_Δ*x
		V = G
		E = np.linalg.inv(I + V.T.dot(M_inv).dot(U))
		M_inv = M_inv - M_inv.dot(U).dot(E).dot(V.T).dot(M_inv)
	
		return M_inv

	def run(self, x_init, max_rep=400):
		d = x_init.shape[0]
		self.x_opt = x_init
		I = np.eye(d)
		converged = False
		x_change = np.linalg.norm(x_init)
		m = 0

		while( (converged == False) and (m < max_rep)):
			old_α = 2
			new_α = 2
			α_Δ = 0
			cost_1 = self.cost_function(self.x_opt)
			[A,G] = self.calc_A(self.x_opt)
			M_inv = np.linalg.inv(I + new_α*A)

			while(new_α > 0.000000001):	
				if True: M_inv = self.constant_update_inv(self.x_opt, G, M_inv, α_Δ)		#	Using Woodbury inverse update
				else:	M_inv = np.linalg.inv(I + new_α*A) 									#	Using slow inverse
				next_x = M_inv.dot(I - new_α*A).dot(self.x_opt)
				cost_2 = self.cost_function(next_x)
				x_change = 0

				if 'run_debug_1' in self.db: print(new_α, cost_1, cost_2)
				if((cost_2 < cost_1) or (abs(cost_1 - cost_2)/abs(cost_1) < 0.0000001)):
					x_change = np.linalg.norm(next_x - self.x_opt)
					[self.x_opt,R] = np.linalg.qr(next_x)		# QR ensures orthogonality
					self.cost_opt = cost_2
					break
				else:
					old_α = new_α
					new_α = new_α*0.2
					α_Δ = new_α - old_α

			m += 1

			if 'run_debug_2' in self.db: print('Cost Norm : %.3f'%cost_2)
			if 'run_debug_3' in self.db: print('Gradient Norm : %.3f'%np.linalg.norm(self.compute_gradient(self.x_opt)))

			#print(x_change)
			if(x_change < 0.001*np.linalg.norm(self.x_opt)): converged = True

		return self.x_opt	
