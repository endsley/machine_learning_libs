#!/usr/bin/python

import numpy as np

def svd_sorted(X):
	U,S,V = np.linalg.svd(X)	
	lastV = None
	for m in S:
		if lastV == None:
			lastV = m
		else:

	print np.argsort(S)[::-1]
#	print i ,'\n'
#	print d ,'\n'

#	print U ,'\n'
#	print S ,'\n'
#	print V ,'\n'

if __name__ == '__main__':
	from random_matrix import *
	M = random_matrix(2, 5)
	M = np.random.normal(size=(30,30))
	svd_sorted(M)
	
