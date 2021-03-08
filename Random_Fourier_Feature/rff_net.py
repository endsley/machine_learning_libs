#!/usr/bin/env python

import torch
from torch.autograd import Variable
from RFF import RFF
import numpy as np
import torch.nn.functional as F
import time 

class rff_net(torch.nn.Module):
	def __init__(self, db, learning_rate=0.001):
		super(rff_net, self).__init__()
		self.db = db
		self.learning_rate = learning_rate
		self.Φ = RFF(db['RFF_Width'])		# RFF width
		self.l = []	
		
		self.l.append(self.linear(db['d'], db['W_Width']))
		for ᘔ in np.arange(2, db['depth']):
			self.l.append(self.linear(db['RFF_Width'], db['W_Width']))
		self.l.append(self.linear(db['RFF_Width'], db['out_dim']))

		self.output_network()

	def linear(self, inDim, outDim):
		W = torch.zeros(inDim, outDim)
		W = Variable(W.type(torch.FloatTensor), requires_grad=True)
		W = W.to(self.db['device'], non_blocking=True )										# make sure the data is stored in CPU or GPU device
		torch.nn.init.kaiming_normal_(W , a=0, mode='fan_in')	
		return W

	def output_network(self):
		print('\tConstructing Kernel Net')
		for α, W in enumerate(self.l):
			print('layer %d : %s'%(α, str(W.shape)))

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

	def forward(self, y0):
		for m, layer in enumerate(self.children(),1):
			if m == self.num_of_linear_layers:
				cmd = 'self.y_pred = self.l' + str(m) + '(y' + str(m-1) + ')'
				#print(cmd)
				exec(cmd)
				#return self.y_pred
			else:
				var = 'y' + str(m)
				cmd = var + ' = F.' + self.Φ_type + '(self.l' + str(m) + '(y' + str(m-1) + '))'
				#print(cmd)
				exec(cmd)

		if self.loss_function == 'CE_loss':
			return torch.nn.functional.cross_entropy(self.y_pred, y_true)
		else:
			y_out = torch.sum(self.y_pred, dim=1)
			return torch.nn.functional.mse_loss(y_out,y_true)


if __name__ == "__main__":
	np.set_printoptions(precision=4)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)
	torch.set_printoptions(edgeitems=3)
	torch.set_printoptions(threshold=10_00)
	torch.set_printoptions(linewidth=400)

	X = np.random.rand(100,3)
	Y = np.random.rand(1,100)

	db = {}
	db['loss_function'] = 'MSE_loss'
	db['d'] = X.shape[1]
	db['W_Width'] = 10
	db['RFF_Width'] = 500
	db['depth'] = 4
	db['device'] = 'cuda'
	db['out_dim'] = 1

	rff_net(db)
	import pdb; pdb.set_trace()
