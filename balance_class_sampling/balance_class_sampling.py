#!/usr/bin/env python

import numpy as np
import sys
from numpy import genfromtxt
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder


np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

#	Take data and out equal number of samples from each class a batch
class balance_class_sampling:
	def __init__(self, X, Y, with_replacement=True):
		self.X = X
		self.Y = Y
		self.d = X.shape[1]
		self.n = X.shape[0]

		if type(Y) == type([]): self.Y = np.array(Y)
		self.Y = np.reshape(self.Y,(len(Y),1))
		self.Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(self.Y)
		self.c = self.Yₒ.shape[1]

		self.X_list = {}
		self.X_shuffled = {}
		self.Y_list = {}
		self.Yₒ_list = {}

		self.l = np.unique(Y)
		for i in self.l:
			indices = np.where(Y == i)[0]
			self.X_list[i] = X[indices, :]
			self.Y_list[i] = self.Y[indices]
			self.Yₒ_list[i] = self.Yₒ[indices, :]

			self.X_shuffled[i] = shuffle(self.X_list[i])		

		#for each class, gen a new X[]
		#each time we sample, we pick a subset from each class 
		#combine the subset into 1 X and 1 Y, output that subset

	def sample(self, samples_per_class=10):
		if samples_per_class > self.n:
			print('Error : Your batch size is more than the number of samples\n')
			sys.exit()


		# Check if each class has enough samples per class
		for i in self.l:
			if self.X_shuffled[i].shape[0] < samples_per_class:
				for j in self.l:		# reshuffle each class
					self.X_shuffled[j] = shuffle(self.X_list[j])		

				#print('\n\n\nSHuffle')
				#print(self.X_shuffled)
				#print('\n\n\n')
				#import pdb; pdb.set_trace()	
				break
			


		Xout = np.empty((0, self.d))	
		Yout = np.empty((0,1))	
		Yₒout = np.empty((0, self.c))	

		for i in self.l:
			newX = self.X_shuffled[i][0:samples_per_class, :]
			self.X_shuffled[i] = self.X_shuffled[i][samples_per_class:, :]

			Xout = np.vstack((Xout, newX))
			Yout = np.vstack((Yout, self.Y_list[i][0:samples_per_class]))
			Yₒout = np.vstack((Yₒout, self.Yₒ_list[i][0:samples_per_class, :]))

		return Xout, Yout, Yₒout



if __name__ == '__main__':
	#X = genfromtxt('../datasets/moon.csv', delimiter=',')
	#Y = genfromtxt('../datasets/moon_label.csv', delimiter=',')

	X = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]])
	Y = np.array([0,0,0,0,0,1,1,1,1,1])

	BCS = balance_class_sampling(X,Y)
	for i in range(20):
		[Xout, Yout, Yₒout] = BCS.sample(samples_per_class=3)
		print(Xout, '\n--------------\n', Yₒout, '\n\n')


