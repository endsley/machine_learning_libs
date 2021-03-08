#!/usr/bin/env python

import torch
import numpy as np
import sys
from sklearn import linear_model
from torch import nn
from src.tools.gumbel_softmax import *
from torch.autograd import Variable
import collections

def basic_optimizer(model, train_loader):
	db = model.db
	model.optimization_initialization()

	optimizer = torch.optim.Adam(model.parameters(), lr=db['learning_rate'])	
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)

	for epoch in range(db['max_ℓ#']):

		loss_list = []	
		for (i, data) in enumerate(train_loader):
			[inputs, labels, indices] = data
			inputs = Variable(inputs.type(db['dataType']), requires_grad=False)
			labels = Variable(labels.type(db['dataType']), requires_grad=False)
			optimizer.zero_grad()
			
			loss = model(inputs, labels, indices)
			loss.backward()
			optimizer.step()

			loss_list.append(loss.item())

		loss_avg = np.array(loss_list).mean()
		scheduler.step(loss_avg)
		early_exit = model.on_new_epoch(loss_avg, (epoch+1), scheduler._last_lr[0])
		if early_exit: break
		#if loss_avg < 0.0001: break
		#if scheduler._last_lr[0] < 1e-9: break




