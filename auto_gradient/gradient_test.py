#!/usr/bin/python

import autograd.numpy as np
from autograd import grad

# Automatically find the gradient of a function
# Define a function Tr(W.T*A*W), we know that gradient = (A+A')*W
def trance_quad(W, A): 
	return np.trace(np.dot(np.dot(np.transpose(W),A), W))

#	Initial setup
n = 5
A = np.random.random((n,n))
W = np.random.random((n,1))



grad_foo = grad(trance_quad)       # Obtain its gradient function
print 'Autogen Gradient : \n', grad_foo(W,A)
print 'Theoretical Gradient : \n', np.dot((A+np.transpose(A)), W)

import pdb; pdb.set_trace()
