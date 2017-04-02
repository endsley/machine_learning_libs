#!/usr/bin/python

from scipy.linalg import eigh
import numpy as np
import time

#	Note : the sample should be thousands before it start getting accurate

#	X must be positive semidefinite, if not, u must use column sampling on svd
#	sampling_percentage is between 0 to 1
#	note that X3 = [W G21.T; G21 G22]
def variational_nystrom(X, return_rank, sampling_percentage):
	p = sampling_percentage
	num_of_columns = np.floor(p*X.shape[1])
	Mp = np.random.permutation(X.shape[1])

	rc = Mp[num_of_columns:]	#	residual columns
	rp = Mp[0:num_of_columns]	#	random permutation

	X2 = np.hstack((X[:,rp], X[:,rc]))		#	restack horizontally
	X3 = np.vstack((X2[rp,:], X2[rc,:]))	#	restack vertically

	W = X3[0:num_of_columns, 0:num_of_columns]
	G21 = X3[num_of_columns:, 0:num_of_columns]
	G22 = X3[num_of_columns:, num_of_columns:]
	C = np.vstack( (W,G21) )
	D = np.vstack( (G21.T, G22) )
	E = np.hstack( (C,D) )

	#	Now we solve for generalized eigenproblem AV= BV Lambda
	B = W.dot(W) + G21.T.dot(G21)
	A = (B.dot(W) + (G21.T.dot(G21).dot(W)).T + G21.T.dot(G22).dot(G21))

	[D,V] = eigh(A, B)

	U = np.fliplr(V)
	D = np.flipud(D)

	nV = C.dot(U)
	eigenVector = np.zeros( nV.shape )
	eigenVector[Mp,:] = nV


	##	Sort them
	#idx = D.argsort()[::-1]   
	#D = D[idx]
	#eigenVector = eigenVector[:,idx]	


	return [eigenVector, D]
	




if __name__ == '__main__':
	def eig_sorted(X):
		D,V = np.linalg.eig(X)	
		idx = D.argsort()[::-1]   
		D = D[idx]
		V = V[:,idx]	
	
		return [V,D] 

	#	print setting
	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=5)
	np.set_printoptions(linewidth=900)

	#	program settings
	desired_rank = 3
	example_size = 20


#	#	Run without nystrom
#	X = np.random.normal(size=(example_size, example_size))
#	Q,R = np.linalg.qr(X, mode='reduced')
#	eigVecs = Q[:,0:desired_rank]
#	eigVals = np.diag(np.array(range(desired_rank)) + 1)
#	noise = np.diag(np.random.normal(scale=0.0001, size=(example_size)))
#	M = eigVecs.dot(eigVals).dot(eigVecs.T) + noise
#
#	[V,D] = eig_sorted(M)
#	print D[0:desired_rank]


	#	Run with Nystrom
	total = np.zeros(desired_rank)
	avg_amount = 5

	X = np.random.normal(size=(example_size, example_size))
	Q,R = np.linalg.qr(X, mode='reduced')
	eigVecs = Q[:,0:desired_rank]
	eigVals = np.diag(np.array(range(desired_rank)) + 1)
	noise = np.diag(np.random.normal(scale=0.0001, size=(example_size)))
	M = eigVecs.dot(eigVals).dot(eigVecs.T) + noise

	start = time.time()
	[U,S,V] = np.linalg.svd(M)
	print("--- SVD Time : %s seconds ---" % (time.time() - start))

	start = time.time()
	[estimated_V,estimated_D] = variational_nystrom(M, desired_rank, 0.30)
	print("--- V Nystrom Time : %s seconds ---\n\n" % (time.time() - start))


	print 'True Eig Value and Vectors : '
	print eigVecs[0:5,0:desired_rank], '\n\n'
	print np.diag(eigVals), '\n------------\n'


	print 'Estimated Eig Value and Vectors : '
	print estimated_V[0:5,0:desired_rank], '\n\n'
	print estimated_D[0:desired_rank], '\n------------\n'

