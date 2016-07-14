#!/usr/bin/python

import numpy as np


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

	[V,D] = eig_sorted(W)

	ratio = float(X.shape[1])/num_of_columns
	estimated_eig_value = ratio*D[0:return_rank]
	print estimated_eig_value

	#print W , '\n'
	#print V , '\n'
	#print D , '\n'
	#print V.dot(np.diag(D)).dot(V.T)

#	output = {}
#	output['W'] = W
#	output['G21'] = G21
#	output['G22'] = G22
#
#	print 'X : \n' , X3
#	print 'W : \n' , W
#	print 'G21 : \n', G21
#	print 'G22 : \n' , G22
#


if __name__ == '__main__':
	def eig_sorted(X):
		D,V = np.linalg.eig(X)	
		lastV = None
		sort_needed = False
		for m in D:
			if m > lastV and lastV != None:
				sort_needed = True
			lastV = m
		
		if sort_needed:
			idx = D.argsort()[::-1]   
			D = D[idx]
			V = V[:,idx]	
	
		return [V,D] 

	#	print setting
	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=5)
	np.set_printoptions(linewidth=900)

	#	program settings
	desired_rank = 5
	example_size = 1000

	#	-------------------------
	X = np.random.normal(size=(example_size, example_size))
	Q,R = np.linalg.qr(X, mode='reduced')
	eigVecs = Q[:,0:desired_rank]
	eigVals = np.diag(np.array(range(desired_rank)) + 1)
	noise = np.diag(np.random.normal(scale=0.0001, size=(example_size)))
	M = eigVecs.dot(eigVals).dot(eigVecs.T) + noise

	nystrom(M, desired_rank, 0.4)


#	U,Se,V1 = nystrom(X, desired_rank, num_of_random_column)
#	U1 = U[0:10, 0:desired_rank]
#
#
#	U,S,V2 = linalg.svd(X)
#	U2 = U[0:10, 0:desired_rank]
#
#	print 'Estimated Eigen Values : \n' , Se[0:desired_rank]
#	print 'Real Eigen Values : \n' , S[0:desired_rank] , '\n'
#	print 'Estimated U : '
#	print U1 , '\n'
#	print 'Real U : '
#	print U2 , '\n'
#	print 'Estimated V : '
#	print V1 , '\n'
#	print 'Real V : '
#	print V2 , '\n'



