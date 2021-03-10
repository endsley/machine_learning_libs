#!/usr/bin/env python3

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import numpy as np

class DManager(Dataset):
	def __init__(self, X, Y, Torch_dataType):
		self.X = preprocessing.scale(X)

		self.N = self.X.shape[0]
		self.d = self.X.shape[1]
		self.X_Var = torch.tensor(self.X)
		self.X_Var = Variable(self.X_Var.type(Torch_dataType), requires_grad=False)

		self.Y = Y
		self.Y_Var = torch.tensor(self.Y)
		self.Y_Var = Variable(self.Y_Var.type(Torch_dataType), requires_grad=False)

		print('\tData of size %dx%d was loaded ....'%(self.N, self.d))

	def __getitem__(self, index):
		return self.X[index], self.Y[index], index


	def __len__(self):
		try: return self.X.shape[0]
		except: return 0

