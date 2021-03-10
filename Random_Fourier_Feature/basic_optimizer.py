#!/usr/bin/env python

import torch
import numpy as np
import sys
from sklearn import linear_model
from torch import nn
from torch.autograd import Variable
import collections

def basic_optimizer(model, train_loader):
	db = model.db
	optimizer = model.get_optimizer()	
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)

	for epoch in range(db['max_ℓ#']):

		loss_list = []	
		for (i, data) in enumerate(train_loader):
			[inputs, labels, indices] = data
			inputs = Variable(inputs.type(torch.FloatTensor), requires_grad=False)
			labels = Variable(labels.type(torch.FloatTensor), requires_grad=False)

			inputs = inputs.to(db['device'], non_blocking=True )										# make sure the data is stored in CPU or GPU device
			labels = labels.to(db['device'], non_blocking=True )										# make sure the data is stored in CPU or GPU device

			loss = model(inputs, labels, indices)
			loss.backward()
			loss.retain_grad()


			#print(model.W)
			#print(model.W.requires_grad)
			#print(model.W.grad)
			#print(model.W.is_leaf)
			#print('\n')

			#with torch.no_grad():
			#	print(model.W)
			#	print(model.W.grad)
			##	model.W -= 0.001* model.W.grad
			##	model.W.grad = None
			#import pdb; pdb.set_trace()
			print(loss.item())


			#import pdb; pdb.set_trace()
			#for param in model.parameters(): 	print(param)
			#	print(param.grad)
			#	import pdb; pdb.set_trace()

			optimizer.step()
			optimizer.zero_grad()

			loss_list.append(loss.item())

		loss_avg = np.array(loss_list).mean()
		scheduler.step(loss_avg)

		#early_exit = model.on_new_epoch(loss_avg, (epoch+1), scheduler._last_lr[0])
		#if early_exit: break
		#if loss_avg < 0.0001: break
		#if scheduler._last_lr[0] < 1e-9: break




