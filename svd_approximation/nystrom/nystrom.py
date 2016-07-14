#!/usr/bin/python

from numpy import *

#	X must be positive semidefinite, if not, u must use column sampling on svd
#	sampling_percentage is between 0 to 1
def nystrom(X, sampling_percentage):
	p = sampling_percentage
	print floor(p*X.shape[1])

#	if rank + num_of_random_column > X.shape[1]:
#		num_of_random_column = X.shape[1] - rank
#
#	random_matrix = random.normal(size=(X.shape[1], num_of_random_column))
#	omega, r = linalg.qr(random_matrix, mode='reduced')
#	X_hat = X.dot(X.T)
#
#	Q, R = linalg.qr(X_hat.dot(X_hat).dot(X).dot(omega), mode='reduced')
#	smaller_matrix = Q.T.dot(X)
#	U,S,V = linalg.svd(smaller_matrix)
#	U = Q.dot(U)
#
#	return U,S,V

if __name__ == '__main__':
	#	print setting
	set_printoptions(suppress=True)
	set_printoptions(precision=5)
	set_printoptions(linewidth=900)

	#	program settings
	desired_rank = 2
	example_size = 10

	#	-------------------------
	X = random.normal(size=(example_size, example_size))
	Q,R = linalg.qr(X, mode='reduced')
	eigVecs = Q[:,0:desired_rank]
	eigVals = diag(array(range(desired_rank)) + 1)
	noise = diag(random.normal(scale=0.0001, size=(example_size)))
	M = eigVecs.dot(eigVals).dot(eigVecs.T) + noise

	nystrom(M, 0.5)


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



