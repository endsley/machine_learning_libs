#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize			# version : 0.17
import matplotlib.pyplot as plt




def cluster_plot(X, Y, title=None, save_path=None):
	cmap = ['b', 'g', 'r', 'c', 'm', 'y','k']
	labels = np.unique(Y)
	if len(labels) > 7:
		print('Error : The plot can currently only handle at most 7 classes')
		sys.exit()

	if np.ndim(Y) == 2:	#Must use 1D label, not Yₒ
		Y = np.squeeze(Y)

	for i, j in enumerate(labels):
		subX = X[Y == labels[i]]
		plt.plot(subX[:,0], subX[:,1], cmap[i] + '.')
		plt.plot(subX[-1,0], subX[-1,1], 'r.')			# showing the last sample added, remove later


	plt.xlabel('x')
	plt.ylabel('y')

	if title is None: plt.title('Clustering Results')
	else: plt.title(title)

	if save_path is None: plt.show()
	else:
		plt.savefig(save_path)
		plt.close()


if __name__ == "__main__":
	n = 100
	x1 = np.random.randn(n,2) + np.array([4,4])
	x2 = np.random.randn(n,2) + np.array([-4,-4])
	X = np.vstack((x1,x2))
	
	Y = np.concatenate([np.zeros(n), np.ones(n)])
	
	cluster_plot(X, Y)
