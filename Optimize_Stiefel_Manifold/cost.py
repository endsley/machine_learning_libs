#!/usr/bin/env python

import numpy as np
import sklearn.metrics

class cost:
	def __init__(self, X, U):
		Ku = U.dot(U.T)
		n = U.shape[0]
		H = np.eye(n) - (1.0/n)*np.ones((n,n))
		self.Γ = H.dot(Ku).dot(H)
		
		self.X = X
		self.U = U
		self.σ = 1
		self.γ = 0.5

	def compute_inverted_Degree_matrix(self, M):
		return np.diag(1.0/np.sqrt(M.sum(axis=1)))
	
	def compute_Degree_matrix(self, M):
		return np.diag(np.sum(M, axis=0))
	

	def compute_Φ(self, W):
		X = self.X
		K = sklearn.metrics.pairwise.rbf_kernel(self.X.dot(W), gamma=self.γ)
		Ψ = (1/self.σ)*K*self.Γ
		D = self.compute_Degree_matrix(Ψ)
		Φ = X.T.dot(D-Ψ).dot(X)

		return Φ

	def compute_cost(self, W):
		X = self.X.dot(W)
		rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=self.γ)
		return -np.sum(rbk*self.Γ)

	def compute_gradient(self, W):
		X = self.X
		K = sklearn.metrics.pairwise.rbf_kernel(self.X.dot(W), gamma=self.γ)
		Ψ = (1/self.σ)*K*self.Γ
		D = self.compute_Degree_matrix(Ψ)

		#print(D)
		#import pdb; pdb.set_trace()
		gradient = 2*X.T.dot(D-Ψ).dot(X).dot(W)

		return gradient
