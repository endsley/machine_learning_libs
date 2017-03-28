#!/usr/bin/python

from variational_nystrom import *
import numpy as np


Q,R = np.linalg.qr( np.random.normal(size=(example_size, example_size)) , mode='reduced')
eigVecs = Q[:,0:desired_rank]
eigVals = np.diag(np.array(range(desired_rank)) + 1)
noise = np.diag(np.random.normal(scale=0.0001, size=(example_size)))
M = eigVecs.dot(eigVals).dot(eigVecs.T) + noise

