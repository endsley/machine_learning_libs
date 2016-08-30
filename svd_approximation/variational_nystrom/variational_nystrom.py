#!/usr/bin/python

from scipy.linalg import eigh
import numpy as np
#	Note : the sample should be thousands before it start getting accurate

#	X must be positive semidefinite, if not, u must use column sampling on svd
#	sampling_percentage is between 0 to 1
#	note that X3 = [W G21.T; G21 G22]
def nystrom(X, return_rank, sampling_percentage):
	p = sampling_percentage
	num_of_columns = np.floor(p*X.shape[1])
	rp = np.random.permutation(X.shape[1])

	rc = rp[num_of_columns:]	#	residual columns
	rp = rp[0:num_of_columns]	#	random permutation

	X2 = np.hstack((X[:,rp], X[:,rc]))		#	restack horizontally
	X3 = np.vstack((X2[rp,:], X2[rc,:]))	#	restack vertically

	W = X3[0:num_of_columns, 0:num_of_columns]
	G21 = X3[num_of_columns:, 0:num_of_columns]
	G22 = X3[num_of_columns:, num_of_columns:]
	C = np.vstack( (W,G21) )


	#	Now we solve for generalized eigenproblem AV= BV Lambda
	B = W.dot(W) + G21.T.dot(G21)
	A = (B.dot(W) + (G21.T.dot(G21).dot(W)).T + G21.T.dot(G22).dot(G21))

	[D,V] = eigh(A, B)
	U = np.fliplr(V)
	D = np.flipud(D)


	nV = C.dot(V)
	#for m in range(nV.shape[1]): nV[:,m] = nV[:,m]/np.linalg.norm(nV[:,m])

	print nV , '\n\n'
	print D , '\n\n'


	[D,V] = eigh(X)
	V = np.fliplr(V)
	D = np.flipud(D)

	print V , '\n\n'
	print D , '\n\n'
	import pdb; pdb.set_trace()



#	estimated_eig_value = ratio*D[0:return_rank]	
#	bottom_estimate = G21.dot(V).dot(np.linalg.inv(np.diag(D)))
#	eigVector = np.vstack((V,bottom_estimate))
#	eigVector = eigVector[:,0:num_of_columns]
#
#	eigVector = eigVector / np.linalg.norm(eigVector, axis=0)[np.newaxis]
#	return [eigVector, estimated_eig_value]


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
	example_size = 8


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
	
	print eigVals

	#for m in range(avg_amount):
	nystrom(M, desired_rank, 0.50)

	#[V,D] = nystrom(M, desired_rank, 0.10)
	#print D[0:desired_rank]

