#!/usr/bin/env python
#	Written by Chieh Wu

import numpy as np
import sklearn.metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder


np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

class sort_samples_into_class:
	# sample_num, the larger the better approximation
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
		self.d = X.shape[1]
		self.n = X.shape[0]

		if type(Y) == type([]): self.Y = np.array(Y)
		self.Y = np.reshape(self.Y,(len(Y),1))
		self.Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(self.Y)
		self.c = self.Yₒ.shape[1]

		self.X_list = {}
		self.Y_list = {}
		self.Yₒ_list = {}

		self.l = np.unique(Y)
		for i in self.l:
			indices = np.where(Y == i)[0]
			self.X_list[i] = X[indices, :]
			self.Y_list[i] = self.Y[indices]
			self.Yₒ_list[i] = self.Yₒ[indices, :]





if __name__ == "__main__":
	X = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]])
	Y = np.array([0,0,0,0,0,1,1,1,1,1])


	sortS = sort_samples_into_class(X,Y)
	print(sortS.X_list.keys())

