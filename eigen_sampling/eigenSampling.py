#!/usr/bin/env python

from numpy import genfromtxt
import sklearn
import numpy as np
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


#	If there are a lot of samples in an experiment, you can discover
#	a subset that represents the total set.


np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)



X = genfromtxt('../datasets/data_4.csv', delimiter=',')
sigma = np.median(sklearn.metrics.pairwise.pairwise_distances(X))
gamma_value = 1.0/(2*sigma*sigma)

#	This is the ground truth
rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma_value)
true_rank = np.linalg.matrix_rank(rbk)	#	<- This is the ground truth that we don't know
pxrange = np.arange(2,42,2)


#	This is the sampling 

avging_list = np.empty((0, 20))
for p in range(5000):
	total_list = np.empty((0, 2))
	growth_list = []
	sampling_list = np.split(np.random.permutation(40), 20)	#	each time I sample only 2
	for i in sampling_list:
		total_list = np.vstack((total_list, X[i,:]))		
		new_rbk = sklearn.metrics.pairwise.rbf_kernel(total_list, gamma=gamma_value)
		newRank = np.linalg.matrix_rank(new_rbk, tol=0.1)
		growth_list.append(newRank)

	avging_list = np.vstack((avging_list, growth_list))


avgedList = np.mean(avging_list, axis=0)
plt.plot(pxrange, avgedList, 'r-')
plt.show()

import pdb; pdb.set_trace()


