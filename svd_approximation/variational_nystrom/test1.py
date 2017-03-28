#!/usr/bin/python

from variational_nystrom import *
from scipy.linalg import eigh
import numpy as np


np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
np.set_printoptions(linewidth=900)

example_size = 20

Q,R = np.linalg.qr( np.random.normal(size=(example_size, example_size)) , mode='reduced')
eigValues = np.diag(np.sort(np.random.randn(example_size)))
noise = np.diag(np.random.normal(scale=0.0001, size=(example_size)))
M = Q.dot(eigValues).dot(Q.T) + noise


[D,V] = eigh(M)

[estimated_V,estimated_D] = variational_nystrom(M, 3, 0.60)

print Q, '\n'
print np.diag(eigValues)
print '\n\n--------------\n\n'
#print V,'\n'
#print D

print estimated_V,'\n'
print estimated_D
