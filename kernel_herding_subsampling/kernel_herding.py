#!/usr/bin/env python

import numpy as np
import sklearn.metrics
import torch
from torch.autograd import Variable
from RFF import *
import torch.nn.functional as F

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


#	Written by Chieh Wu
#	This function calculates the Gaussian Kernel by approximate it through Random fourier Feature technique.

class kernel_herding:
	# sample_num, the larger the better approximation
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def obtain_subsamples(self):


if __name__ == "__main__":
	X = genfromtxt('../datasets/moon.csv', delimiter=',')
	Y = genfromtxt('../datasets/moon_label.csv', delimiter=',')

	kh = kernel_herding(X,Y)

