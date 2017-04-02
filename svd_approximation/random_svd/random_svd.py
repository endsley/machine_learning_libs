#!/usr/bin/python

from numpy import *

#	Input
#	X is the data matrix itself
#	rank is the expected number of eigen values
#	num_of_random_column is the number of random vectors to project down to

def random_svd(X, rank, num_of_random_column):
	
	if rank + num_of_random_column > X.shape[1]:
		num_of_random_column = X.shape[1] - rank

	random_matrix = random.normal(size=(X.shape[1], num_of_random_column))
	omega, r = linalg.qr(random_matrix, mode='reduced')
	X_hat = X.dot(X.T)

	Q, R = linalg.qr(X_hat.dot(X_hat).dot(X).dot(omega), mode='reduced')
	smaller_matrix = Q.T.dot(X)
	U,S,V = linalg.svd(smaller_matrix)
	U = Q.dot(U)

	return U,S,V

if __name__ == '__main__':
	#	print setting
	set_printoptions(suppress=True)
	set_printoptions(precision=3)

	#	program settings
	desired_rank = 5
	num_of_random_column = 10
	example_size = 20

	#	-------------------------
	X = random.normal(size=(example_size, example_size))
	U,Se,V1 = random_svd(X, desired_rank, num_of_random_column)
	U1 = U[0:10, 0:desired_rank]


	U,S,V2 = linalg.svd(X)
	U2 = U[0:10, 0:desired_rank]

	print 'Estimated Eigen Values : \n' , Se[0:desired_rank]
	print 'Real Eigen Values : \n' , S[0:desired_rank] , '\n'
	print 'Estimated U : '
	print U1 , '\n'
	print 'Real U : '
	print U2 , '\n'
	print 'Estimated V : '
	print V1 , '\n'
	print 'Real V : '
	print V2 , '\n'



