#!/usr/bin/env python

import numpy as np
import sklearn.metrics
from numpy import genfromtxt
from lib.sort_samples_into_classes import *
from lib.kernel_herding_debug import *
from lib.RFF import *

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


#	Written by Chieh Wu
#	This function finds a subset of samples that represents the total via kernel herding

class kernel_herding:
	# sample_num, the larger the better approximation
	def __init__(self, X, Y, debug_mode=False, data_name=None):
		self.debug_mode = debug_mode
		self.data_name = data_name
		self.kH_debug = kernel_herding_debug(self)

		self.n = X.shape[0]
		self.d = X.shape[1]
		self.X = X
		self.Y = Y

		self.sSamples = sort_samples_into_class(X,Y)
		self.sSamples.shuffled_X = {}
		self.sSamples.subSamples = {}
		self.sSamples.subSamples_in_RKHS = {}
		self.sSamples.class_Φx = {}
		self.sSamples.class_ᶙ = {}			# kernel embedding for each class
		self.sSamples.rff_list = {}
		self.X_shuffled = {}

		self.l = np.unique(Y)
		for i in self.l:
			ẋ = shuffle(self.sSamples.X_list[i])		# samples shuffed within a class
			self.sSamples.shuffled_X[i] = ẋ
			σ = np.median(sklearn.metrics.pairwise.pairwise_distances(ẋ, metric='euclidean'))

			self.sSamples.rff_list[i] = RFF(300)
			self.sSamples.rff_list[i].initialize_RFF(ẋ,σ)
			self.sSamples.class_Φx[i] = self.sSamples.rff_list[i].np_feature_map(ẋ)
			self.sSamples.class_ᶙ[i] = np.mean(self.sSamples.class_Φx[i],axis=0)

			self.sSamples.subSamples_in_RKHS[i] = self.sSamples.class_Φx[i][0:10,:]			# randomly choose 10 samples to seed each class
			self.sSamples.subSamples[i] = ẋ[0:10,:]


	def obtain_subsamples(self, exit_threshold=0.0025):
		for j in range(self.n):
			oldError = 0
			for i in self.l:
				ẋ = self.sSamples.shuffled_X[i]
				n = self.sSamples.subSamples_in_RKHS[i].shape[0]
				μ = self.sSamples.class_ᶙ[i]									# This is μ from the total population
				μᵃᵖᵖ = np.sum(self.sSamples.subSamples_in_RKHS[i], axis=0) 				# This is the approximate μ for subpopulation
				Φx = self.sSamples.class_Φx[i] 
				#Φx = self.sSamples.class_Φx[i] + 0.001*np.random.randn(self.sSamples.class_Φx[i].shape[0], self.sSamples.class_Φx[i].shape[1])
	
				με = μ - (μᵃᵖᵖ + Φx)/(n+1)
				ε = np.linalg.norm(με, axis=1)
	
				newError = np.min(ε)
				minIndex = np.argmin(ε)

				if newError > oldError: oldError = newError
	
				self.sSamples.subSamples_in_RKHS[i] = np.vstack((self.sSamples.subSamples_in_RKHS[i], np.atleast_2d(Φx[minIndex])))
				self.sSamples.subSamples[i] = np.vstack((self.sSamples.subSamples[i], np.atleast_2d(ẋ[minIndex,:])))

				#Remove chosen sample without replacement
				self.sSamples.class_Φx[i] = np.delete(self.sSamples.class_Φx[i], minIndex, axis=0)
				self.sSamples.shuffled_X[i] = np.delete(self.sSamples.shuffled_X[i], minIndex, axis=0)

			self.kH_debug.save_results(oldError)
			if oldError < exit_threshold: break

		self.kH_debug.collect_result(self.n)
		return self.combine_subSamples()

	def combine_subSamples(self):
		subSample = np.empty((0, self.d))
		subLabels = np.empty((0, 1))
		for i in self.l:
			ẋ = np.unique(self.sSamples.subSamples[i], axis=0)

			subSample = np.vstack((subSample, ẋ))
			subLabels = np.vstack((subLabels, self.sSamples.Y_list[i][0:ẋ.shape[0], :]))

		return [subSample, subLabels]

	def obtain_residual_samples(self):
		[subSample, subLabels] = self.combine_subSamples()

		delete_index_list = []
		for i in range(subSample.shape[0]):
			V = (subSample[i,:] == self.X).astype(int)	
			loc = np.prod(V, axis=1)
			if np.sum(loc) == 0: import pdb; pdb.set_trace()
			if np.sum(loc) > 1: import pdb; pdb.set_trace()
			ind = np.where(loc == 1)[0][0]
			delete_index_list.append(ind)


		self.X = np.delete(self.X, delete_index_list, axis=0)
		self.Y = np.delete(self.Y, delete_index_list, axis=0)

		return self.X, self.Y



if __name__ == "__main__":
	data_name = 'moon'

	#X = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]])
	#Y = np.array([0,0,0,0,0,1,1,1,1,1])
	X = genfromtxt('../datasets/' + data_name + '.csv', delimiter=',')
	Y = genfromtxt('../datasets/' + data_name + '_label.csv', delimiter=',')

	kh = kernel_herding(X,Y, debug_mode=True, data_name=data_name)
	[ẋ, ӯ] = kh.obtain_subsamples(exit_threshold=0.004)
	[Ẋ, Ý] = kh.obtain_residual_samples()
	import pdb; pdb.set_trace()
