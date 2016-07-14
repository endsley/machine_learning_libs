#!/usr/bin/python

import numpy as np

def svd_sorted(X):
	U,S,V = np.linalg.svd(X)	

	[i, d] = np.argsort(S)
	print S ,'\n'
	print i ,'\n'
	print d ,'\n'

#	print U ,'\n'
#	print S ,'\n'
#	print V ,'\n'

if __name__ == '__main__':
	from random_matrix import *
	M = random_matrix(2, 5)
	svd_sorted(M)
	
