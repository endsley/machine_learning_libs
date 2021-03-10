#!/usr/bin/env python

import numpy as np
import numpy.matlib
import sklearn.metrics
import torch
from torch.autograd import Variable
import torch.nn.functional as F


#	Written by Chieh Wu
#	This function calculates the Gaussian Kernel by approximate it through Random fourier Feature technique.

class RFF:
	# rff_width, the larger the better approximation
	def __init__(self, σ=1, rff_width=20000):
		self.σ = σ
		self.rff_width = rff_width
		self.phase_shift = None
		self.RFF_scale = np.sqrt(2.0/self.rff_width)

	def initialize_RFF(self, x, sigma, output_torch, dtype):
		self.x = x
		
		if self.phase_shift is not None: return
			#if x.shape[0] == self.N: return

		self.N = x.shape[0]
		self.d = x.shape[1]
		self.sigma = sigma


		if type(x) == torch.Tensor or type(x) == np.ndarray:	
			self.phase_shift = 2*np.pi*np.random.rand(1, self.rff_width)
			#self.phase_shift = np.matlib.repmat(b, self.N, 1)	
			self.rand_proj = np.random.randn(self.d, self.rff_width)/(self.sigma)
		else:
			raise ValueError('An unknown datatype is passed into get_rbf as %s'%str(type(x)))

		self.use_torch(output_torch, dtype)
		
	def use_torch(self, output_torch, dtype):
		if not output_torch: return

		dvc = self.x.device
		self.phase_shift = torch.from_numpy(self.phase_shift)
		self.phase_shift = Variable(self.phase_shift.type(dtype), requires_grad=False)
		self.phase_shift = self.phase_shift.to(dvc, non_blocking=True )										# make sure the data is stored in CPU or GPU device

		self.rand_proj = torch.from_numpy(self.rand_proj)
		self.rand_proj = Variable(self.rand_proj.type(dtype), requires_grad=False)
		self.rand_proj = self.rand_proj.to(dvc, non_blocking=True )										# make sure the data is stored in CPU or GPU device

	def __call__(self, x, dtype=torch.FloatTensor):
		self.initialize_RFF(x, self.σ, True, dtype)

		if type(self.x) == np.ndarray:
			self.x = torch.from_numpy(self.x)
			self.x = Variable(self.x.type(self.dtype), requires_grad=False)

		elif type(self.x) != torch.Tensor:
			raise ValueError('An unknown datatype is passed into get_rbf as %s'%str(type(self.x)))

		P = self.RFF_scale*torch.cos(torch.mm(self.x,self.rand_proj) + self.phase_shift)
		return P

	def torch_rbf(self, x, σ):
		P = self.__call__(x, σ)
		K = torch.mm(P, P.transpose(0,1))
		#K = (2.0/self.rff_width)*K
		K = F.relu(K)

		return K

	def np_feature_map(self, x):
		const = np.sqrt(2.0/self.rff_width)
		feature_map = const*np.cos(x.dot(self.rand_proj) + self.phase_shift)
	

		return feature_map

	def np_rbf(self):
		P = np.cos(self.x.dot(self.rand_proj) + self.phase_shift)
		K = (2.0/self.rff_width)*(P.dot(P.T))
		K = np.maximum(K, 0)
		K = np.minimum(K, 1)
		return K

	def get_rbf(self, x, sigma, output_torch=False, dtype=torch.FloatTensor):
		self.dtype = dtype
		self.initialize_RFF(x,sigma, output_torch, dtype)

		if output_torch: return self.torch_rbf(x, sigma)
		else: return self.np_rbf()


if __name__ == "__main__":
	np.set_printoptions(precision=4)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)
	torch.set_printoptions(edgeitems=3)
	torch.set_printoptions(threshold=10_00)
	torch.set_printoptions(linewidth=400)


	X = np.random.randn(5,2)
	σ = 1
	gamma = 1.0/(2*σ*σ)

	rff = RFF(30000)
	rbf_np = rff.get_rbf(X, σ)

	rff_torch = RFF(30000)	
	rbf_torch = rff_torch.get_rbf(torch.from_numpy(X).type(torch.FloatTensor), σ, True)


	rff_torch_inner = RFF(30000)	
	φ = rff_torch_inner(torch.from_numpy(X).type(torch.FloatTensor), σ)
	rbf_torch_inner = torch.matmul(φ, φ.T)

	sk_rbf = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)

	print('Using SKlearn library')
	print(sk_rbf)
	print('\n')
	print('Using my own code with Pytorch')
	print(rbf_torch.cpu().detach().numpy())
	print('\n')
	print('Using my own code with Pytorch through Inner Product')
	print(rbf_torch_inner.cpu().detach().numpy())
	print('\n')
	print('Using My own code')
	print(rbf_np)

