#!/usr/bin/env python

from __future__ import division
import numpy as np

def cluster_acc(Y_pred, Y):
	from sklearn.utils.linear_assignment_ import linear_assignment
	assert Y_pred.size == Y.size
	D = max(Y_pred.max(), Y.max())+1
	w = np.zeros((D,D), dtype=np.int64)
	for i in range(Y_pred.size):
		w[Y_pred[i], int(Y[i])] += 1
	ind = linear_assignment(w.max() - w)

	return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size #w


if __name__ == '__main__':
	Y_pred = np.array([0,1,0,1,1,0,0,1,1,1,1,2,3,1,2,2,2,3,2,3,3,3])
	Y = np.array([0,1,0,0,0,3,1,1,1,1,1,2,3,1,2,2,2,3,2,3,0,3])
	print cluster_acc(Y_pred, Y)
