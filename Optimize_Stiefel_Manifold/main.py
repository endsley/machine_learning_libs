#!/usr/bin/env python3

import numpy as np
from orthogonal_optimization import *
from random_opt import *
from DimGrowth import *
from ism import *
from cost import *
import time 


#	Optimization on a Stiefel Manifold
#	Written by Chieh Wu , Aug/16/2018
#
#	Solves the problem  min f(x) s.t x^T x = I
#	Here, we solve it in 3 ways 
#		1. Randomly check for good solutions
#		2. Orthogonal Optimization (Paper : A feasible method for optimization with orthogonality constraints)
#		3. ISM	(Paper : Iterative Spectral Method for Alternative Clustering)



X = np.array([[4,4,0],[3,3,0],[-5,-4,-5],[-4,-5,-4]])
U = np.array([[0,1],[0,1],[1,0],[1,0]])

#rand_x = np.random.randn(3,2)
rand_x = np.array([[2,1],[3,1],[1,1]])
[x_init, R] = np.linalg.qr(rand_x)		# QR ensures orthogonality

c = cost(X, U)

print('Random optimization of 5000 samples')
RO = random_opt(c.compute_cost, c.compute_gradient)
start_time = time.time() 
RO.run(x_init)
print("--- %s seconds ---\n" % (time.time() - start_time))

print('\nOrthogonal optimization')
OO = orthogonal_optimization(c.compute_cost, c.compute_gradient)
OO.db['run_debug_2'] = True
start_time = time.time() 
OO.run(x_init)
print("--- %s seconds ---\n" % (time.time() - start_time))

print('\nDG optimization')
OO = DimGrowth(X,U)
start_time = time.time() 
OO.run(np.eye(3)[:,0:2])
#OO.run(np.random.randn(3,2))
print("--- %s seconds ---\n" % (time.time() - start_time))

print('\nISM optimization')
OO = ism(c.compute_cost, c.compute_Φ)
start_time = time.time() 
OO.run(np.zeros((3,2)))
print("--- %s seconds ---\n" % (time.time() - start_time))


