#!/usr/bin/python

import numpy as np

#	This only generate symmetric positive definite matrices
def random_matrix(desired_rank, n):
	#	print setting
	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=5)
	np.set_printoptions(linewidth=900)


	#	-------------------------
	X = np.random.normal(size=(n, n))
	Q,R = np.linalg.qr(X, mode='reduced')
	eigVecs = Q[:,0:desired_rank]
	eigVals = np.diag(np.array(range(desired_rank)) + 1)
	noise = np.diag(np.random.normal(scale=0.0001, size=(n)))
	M = eigVecs.dot(eigVals).dot(eigVecs.T) + noise
	return M

if __name__ == '__main__':
	M = random_matrix(2, 10)
	V,D = np.linalg.eig(M)
	print V, '\n\n', D

