#!/usr/bin/python

from sklearn.cluster import SpectralClustering
import sklearn
import numpy as np
import pickle
import sys
sys.path.append('./libs')
from draw_heatMap import *


class spectral_clustering:
	def __init__(self, data, data_is_kernel=False):	# data is n x d
		self.X = data
		self.data_is_kernel = data_is_kernel
		self.use_default_sigma(1)

	def use_default_sigma(self, ratio):
		if self.data_is_kernel:
			sigma = ratio*np.median(self.X)
			self.gammav = 1.0/(2*sigma*sigma)
		else:
			sigma = ratio*np.median(sklearn.metrics.pairwise.pairwise_distances(self.X, metric='euclidean'))
			self.gammav = 1.0/(2*sigma*sigma)

	def run(self, k):
		if self.data_is_kernel:
			clf = SpectralClustering(n_clusters=k, gamma=self.gammav, affinity='precomputed')	
			self.allocation = clf.fit_predict(self.X)
			self.kernel = self.X
		else:
			clf = SpectralClustering(n_clusters=k, gamma=self.gammav)		#, affinity='precomputed'
			self.allocation = clf.fit_predict(self.X)
			self.kernel = clf.affinity_matrix_
	
		return self.allocation

	def show_heatMap(self, xLabel=[], Title=''):
		hMap = heatMap()
		sorted_kernel = hMap.sort_kernel(self.kernel, self.allocation)
		hMap.draw_HeatMap(sorted_kernel, xlabel=xLabel, ylabel=xLabel, title=Title)


#SC = spectral_clustering(data)
#SC.use_default_sigma(sigma_scale)
#SC.run(k)
#SC.show_heatMap()

