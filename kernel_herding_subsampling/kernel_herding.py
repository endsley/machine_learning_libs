#!/usr/bin/env python

import numpy as np
import sklearn.metrics
from numpy import genfromtxt
from sort_samples_into_classes import *
from RFF import *

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


#	Written by Chieh Wu
#	This function calculates the Gaussian Kernel by approximate it through Random fourier Feature technique.

class kernel_herding:
	# sample_num, the larger the better approximation
	def __init__(self, X, Y):
		self.n = X.shape[0]
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


	def obtain_subsamples(self):
		for j in range(self.n):
			for i in self.l:
				ẋ = self.sSamples.shuffled_X[i]
				n = self.sSamples.subSamples_in_RKHS[i].shape[0]
				μ = self.sSamples.class_ᶙ[i]									# This is μ from the total population
				μᵃᵖᵖ = np.sum(self.sSamples.subSamples_in_RKHS[i], axis=0) 				# This is the approximate μ for subpopulation
				Φx = self.sSamples.class_Φx[i]
	
				με = μ - (μᵃᵖᵖ + Φx)/(n+1)
				ε = np.linalg.norm(με, axis=1)
	
				newError = np.min(ε)
				minIndex = np.argmin(ε)
	
				self.sSamples.subSamples_in_RKHS[i] = np.vstack((self.sSamples.subSamples_in_RKHS[i], np.atleast_2d(Φx[minIndex])))
				self.sSamples.subSamples[i] = np.vstack((self.sSamples.subSamples[i], np.atleast_2d(ẋ[170,:])))
				print(newError)
	
			print(j, '\n')
				#np.linalg.norm(μ - (μᵃᵖᵖ + Φx)/(n+1), axis=1)


if __name__ == "__main__":
	#X = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]])
	#Y = np.array([0,0,0,0,0,1,1,1,1,1])
	X = genfromtxt('../datasets/moon.csv', delimiter=',')
	Y = genfromtxt('../datasets/moon_label.csv', delimiter=',')

	kh = kernel_herding(X,Y)
	kh.obtain_subsamples()

