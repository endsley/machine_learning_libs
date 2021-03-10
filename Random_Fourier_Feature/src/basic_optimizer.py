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
			optimizer.step()
			optimizer.zero_grad()
			loss_list.append(loss.item())

		loss_avg = np.array(loss_list).mean()
		scheduler.step(loss_avg)
		model.on_new_epoch(loss_avg, epoch, scheduler._last_lr[0])



