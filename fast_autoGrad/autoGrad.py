#!/usr/bin/env python

#This example show how to use pytorch to 
#solve a convex optimization problem
#	Optimize x^T A x + b^T x
#	A = [1 0;0 2] , b = [1, 2] , solution = -[1/2 1/2]


import torch
from torch.autograd import Variable
import numpy as np
from minConf_PQN import *

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

learning_rate = 0.1

x = torch.from_numpy(np.ones((2,1)))
x = Variable(x.type(dtype), requires_grad=True)

A = torch.from_numpy(np.array([[1,0],[0,2]]))
A = Variable(A.type(dtype), requires_grad=False)

b = torch.from_numpy(np.array([[1],[2]]))
b = Variable(b.type(dtype), requires_grad=False)

for m in range(30):
	opt1 = torch.mm(x.transpose(0,1), A)
	loss = torch.mm(opt1, x) + torch.mm(b.transpose(0,1),x) 
	loss.backward()


	minConf_PQN(funObj, x, funProj, options=None)



	x.data -= learning_rate*x.grad.data
	x.grad.data.zero_()
	print x.data.numpy()

import pdb; pdb.set_trace()

