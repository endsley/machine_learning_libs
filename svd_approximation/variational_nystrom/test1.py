#!/usr/bin/python

from variational_nystrom import *
from scipy.linalg import eigh
import numpy as np


np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
np.set_printoptions(linewidth=900)

example_size = 10
desired_rank = 3

Q,R = np.linalg.qr( np.random.normal(size=(example_size, example_size)) , mode='reduced')

#Q = Q[:,0:desired_rank]
#eigValues = np.diag(np.array(range(desired_rank)) + 1)
P = np.sort(np.random.randn(example_size))
P[0] = P[0]*30
P[8] = P[8]*30

eigValues = np.diag(P)
noise = np.diag(np.random.normal(scale=0.0001, size=(example_size)))
M = Q.dot(eigValues).dot(Q.T) + noise


#[D,V] = eigh(M)

[estimated_V,estimated_D] = variational_nystrom(M, desired_rank, 0.40)

print Q, '\n'
print np.sort(np.diag(eigValues))
print '\n\n--------------\n\n'
#print V,'\n'
#print D

print estimated_V,'\n'
print np.sort(estimated_D)
import pdb; pdb.set_trace()
