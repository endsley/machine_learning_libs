#!/usr/bin/python
#	Note : This is designed for Python 3

import numpy as np
import sklearn.metrics

class DimGrowth:
	def __init__(self, X, U):
		Ku = U.dot(U.T)
		self.N = N = U.shape[0]
		H = np.eye(N) - (1.0/N)*np.ones((N,N))

		self.previous_gw = np.ones((N,N))
		self.Γ = H.dot(Ku).dot(H)
		
		self.X = X
		self.U = U
		self.σ = 1
		self.σ2 = 1.0/(self.σ*self.σ)
		self.γ = 0.5

		self.x_opt = None
		self.cost_opt = None

	def get_Γ(self):
		return self.Γ

	def get_orthogonal_vector(self, W, m, input_vector):
		for cn in range(m):
			w_prev = W[:,cn]
			w_prev = w_prev[np.newaxis].T

			projected_direction = (np.dot(w_prev.T, input_vector)/np.dot(w_prev.T, w_prev))*w_prev
			input_vector = input_vector - projected_direction	

		input_vector = input_vector/np.linalg.norm(input_vector)
		return input_vector

	def compute_Degree_matrix(self, M):
		return np.diag(np.sum(M, axis=0))


	def optimize_direction(self, w):
		Γ = self.get_Γ()
		X = self.X
		Δ = 0

		if False:
			K = sklearn.metrics.pairwise.rbf_kernel(X.dot(w), gamma=self.γ)
			Ψ = self.σ2*Γ*self.previous_gw*K
			D = self.compute_Degree_matrix(Ψ)
			Φ = D - Ψ
			Δ = -2*X.T.dot(Φ).dot(X).dot(w)
		else:
			K = sklearn.metrics.pairwise.rbf_kernel(X.dot(w), gamma=self.γ)
			Ψ = self.σ2*Γ*self.previous_gw*K

			for i in range(self.N):
				for j in range(self.N):
					x_dif = X[i] - X[j]
					x_dif = x_dif[np.newaxis]
					A_ij = np.dot(np.transpose(x_dif), x_dif)

					Δ = Δ - Ψ[i,j]*A_ij.dot(w)

		return Δ

	def get_objective(self, W, use_previous_gw=True):
		Γ = self.get_Γ()
		X = self.X

		K = sklearn.metrics.pairwise.rbf_kernel(X.dot(W), gamma=self.γ)
		if use_previous_gw:
			return np.sum(Γ*self.previous_gw*K)
		else:
			return np.sum(Γ*K)

	def GD_update(self, w, Δ):
		α = 0.9
		old_obj = self.get_objective(w)

		while α > 0.000001:
			new_w = np.sqrt(1-α*α)*w + α*Δ	
			new_w = new_w/np.linalg.norm(new_w)
			new_obj = self.get_objective(new_w)
			
			if new_obj > old_obj: 
				print('\tα : %.3f , new_obj : %.4f , old_obj : %.4f'%(α, new_obj,old_obj))
				return new_w
			else: α = α/2


	def update_previous_gw(self, new_w):
		X = self.X
		K = sklearn.metrics.pairwise.rbf_kernel(X.dot(new_w), gamma=self.γ)
		self.previous_gw = self.previous_gw*K


	def run(self, W):
		self.q = W.shape[1]
		max_count = 3000

		for m in range(self.q):
			print('m = %d'%m)
			W[:,m] = np.squeeze(self.get_orthogonal_vector(W, m, W[:,m][np.newaxis].T))
			w = W[:,m][np.newaxis].T

			for n in range(max_count):
				Δ = self.optimize_direction(w)	
				Δ = self.get_orthogonal_vector(W, m+1, Δ)	
				new_w = self.GD_update(w, Δ)

				if np.linalg.norm(new_w - w) < 0.01*np.linalg.norm(w): 
					W[:,m] = np.squeeze(new_w)
					#print('m = %d'%m)
					#print(W[:,m])
					self.update_previous_gw(new_w)
					break
				else:
					W[:,m] = np.squeeze(new_w)
					w = new_w

		obj = self.get_objective(W, False)
		print('Cost : %.3f'%-obj)


