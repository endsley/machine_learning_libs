#!/usr/bin/env python

import torch
from torch.autograd import Variable
from RFF import RFF
from basic_optimizer import basic_optimizer
from torch.utils.data import Dataset, DataLoader
from DManager import DManager
from terminal_print import *
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import time 

class rff_net(torch.nn.Module):
	def __init__(self, db, learning_rate=0.001):
		super(rff_net, self).__init__()
		self.db = db
		self.learning_rate = learning_rate
		self.Φ = RFF(db['RFF_Width'])		# RFF width
		self.l = []	


		#self.W = torch.zeros(db['d'], db['W_Width'])
		#torch.nn.init.kaiming_normal_(self.W , a=0, mode='fan_in')	
		#self.W = self.W.cuda()
		#param = torch.nn.Parameter(self.W.type(db['dataType']), requires_grad=True)
		#self.register_parameter(name='W%d'%len(self.l), param=param)


		self.W = torch.nn.Parameter(torch.randn((db['d'], db['W_Width']), device=db['device'] ), requires_grad=True)

        #self.a = torch.nn.Parameter(torch.randn(()))
        #self.b = torch.nn.Parameter(torch.randn(()))
        #self.c = torch.nn.Parameter(torch.randn(()))
        #self.d = torch.nn.Parameter(torch.randn(()))

		#self.l.append(self.add_linear(db['d'], db['W_Width']))

		
		#self.l.append(self.add_linear(db['d'], db['W_Width']))
		#for ᘔ in np.arange(2, db['depth']):
		#	self.l.append(self.add_linear(db['RFF_Width'], db['W_Width']))
		#self.l.append(self.add_linear(db['RFF_Width'], db['out_dim']))

		self.output_network()

	def add_linear(self, inDim, outDim):
		W = torch.zeros(inDim, outDim)
		W = Variable(W.type(db['dataType']), requires_grad=True)
		W = W.to(self.db['device'], non_blocking=True )										# make sure the data is stored in CPU or GPU device
		torch.nn.init.kaiming_normal_(W , a=0, mode='fan_in')	

		self.register_parameter(name='W%d'%len(self.l), param=torch.nn.Parameter(W))
		return W

	def output_network(self):
		print('\tConstructing Kernel Net')
		for α, W in enumerate(self.l):
			print('layer %d : %s'%(α, str(W.shape)))

	def get_optimizer(self):
		#for name, W in self.named_parameters(): print(name, W.shape)
	
		return torch.optim.Adam(self.parameters(), lr=self.db['learning_rate'])
	
	def optimization_initialization(self):
		pass

	def on_new_epoch(self, loss, epoch, lr):
		#print('loss: %.6f, epoch: %d, lr:%.7f'%(loss, epoch, lr))
		write_to_current_line('loss: %.6f, epoch: %d, lr:%.7f'%(loss, epoch, lr))

	def predict(self, x):
		x = x.cuda()
		yᵢ = torch.matmul(x,self.W)
		yᵢ = F.softmax(yᵢ, dim=1)
	
		return torch.round(yᵢ)

	def forward(self, x, y, i):
		#Σ = torch.nn.Sigmoid()
		#yᵢ = torch.matmul(x,self.l[0])
		yᵢ = torch.matmul(x,self.W)

		#yᵢ = Σ(yᵢ)
		yᵢ = F.softmax(yᵢ, dim=1)
		y = y.to(device=db['device'], dtype=torch.int64)
		loss = db['loss_function'](yᵢ, y)
		return loss


#		db = self.db
#		σ = 0.01
#		Φyᵢ = x
#		depth = len(self.l)
#
#		for ι, α in enumerate(self.l, 1):
#			yᵢ = torch.matmul(Φyᵢ,α)
#			if ι != depth: Φyᵢ = self.Φ(yᵢ, σ)
#
#		if db['loss_function'] == torch.nn.MSELoss():
#			yᵢ = yᵢ.view(x.shape[0])
#		else:
#			yᵢ = F.softmax(yᵢ, dim=1)
#			y = y.to(device=db['device'], dtype=torch.int64)
#
#		loss = db['loss_function'](yᵢ, y)
#		return loss
		

if __name__ == "__main__":
	np.set_printoptions(precision=4)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)
	torch.set_printoptions(edgeitems=3)
	torch.set_printoptions(threshold=10_00)
	torch.set_printoptions(linewidth=400)

	N = 50
	X1 = np.random.rand(N,3)
	X2 = np.random.rand(N,3) + 10
	X = np.vstack((X1,X2))

	Y = np.hstack((np.ones(N), np.zeros(N)))
	#Y = np.reshape(Y,(len(Y),1))
	#Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(Y)

	db = {}
	db['loss_function'] = torch.nn.CrossEntropyLoss()			# torch.nn.functional.cross_entropy, torch.nn.MSELoss, torch.nn.CrossEntropyLoss
	db['d'] = X.shape[1]
	db['W_Width'] = 2
	db['RFF_Width'] = 1000
	db['depth'] = 4
	db['device'] = 'cuda'
	db['out_dim'] = 2				# 1 if regression
	db['max_ℓ#'] = 1000
	db['learning_rate'] = 0.001
	db['dataType'] = torch.FloatTensor

	DM = DManager(X, Y, db['dataType'])
	loader = DataLoader(dataset=DM, batch_size=16, shuffle=True, pin_memory=True, drop_last=True)

	R = rff_net(db)
	basic_optimizer(R, loader)
	Ŷ = R.predict(DM.X_Var)
	import pdb; pdb.set_trace()
