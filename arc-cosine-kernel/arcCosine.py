#!/usr/bin/env python

import numpy as np
import math
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class arcCosine():
	def __init__(self):
		self.πˉᑊ = 1/np.pi


	def acKernel(self, x,y,l,n=1):
		return self.Kᵣ(x,y,l,n)

	def J(self, θ, n=1): 
		if n == 1:
			return np.sin(θ) + (np.pi - θ)*np.cos(θ)

	def Kᵣ(self, x,y,l,n=1): # assume n = 1 for now
		J = self.J
		πˉᑊ = self.πˉᑊ
		Kᵣ = self.Kᵣ

		if l == 1:
			ㅣxㅣ = np.linalg.norm(x)
			ㅣyㅣ = np.linalg.norm(y)
			ㅣxㅣㅣyㅣ = ㅣxㅣ*ㅣyㅣ
			xᵀy = x.dot(y)

			R = xᵀy/ㅣxㅣㅣyㅣ
			if R > 1: R = 1

			θ = np.arccos(R)
			Kᵢﺫ = πˉᑊ*ㅣxㅣㅣyㅣ*J(θ)
		elif l > 1:
			KₓₓKᵧᵧ = np.sqrt(Kᵣ(x,x,l-1,n)*Kᵣ(y,y,l-1,n))
			Kₓᵧ = Kᵣ(x,y,l-1)
			θ = np.arccos(Kₓᵧ/KₓₓKᵧᵧ)
			Kᵢﺫ = πˉᑊ*KₓₓKᵧᵧ*J(θ)

		if math.isnan(Kᵢﺫ):
			import pdb; pdb.set_trace()

		return Kᵢﺫ

	def train_on_aKernel(self, X_train, Y_train, X_test, Y_test):
		l = 2		# using only 2 layers for now
		Xbig = np.vstack((X_train, X_test))
		Ybig = np.hstack((Y_train, Y_test))
		
		N = Xbig.shape[0]
		Ƙ = np.zeros((N,N))
		for i in range(N):
			for j in range(N):
				x = Xbig[i,:]
				y = Xbig[j,:]
				Ƙ[i][j] = self.acKernel(x,y,l,n=1)


		clf = SVC(kernel = "precomputed", C = 1, cache_size = 100000)
	
		K_train = Ƙ[0:X_train.shape[0], 0:X_train.shape[0]]
		K_test = Ƙ[X_train.shape[0]:Xbig.shape[0], 0:X_train.shape[0]]

	
		clf.fit(K_train, Y_train)

		Ł_train = clf.predict(K_train)
		Ł_test = clf.predict(K_test)

		train_acc = accuracy_score(Ł_train, Y_train)
		test_acc = accuracy_score(Ł_test, Y_test)

		return [train_acc, test_acc]


if __name__ == "__main__":
	X = genfromtxt('dat/wine_2.csv', delimiter=',')
	Y = genfromtxt('dat/wine_2_label.csv', delimiter=',')
	ẋ = genfromtxt('dat/wine_2_test.csv', delimiter=',')
	ý = genfromtxt('dat/wine_2_label_test.csv', delimiter=',')
	
	AC = arcCosine()
	[train_acc, test_acc] = AC.train_on_aKernel(X,Y, ẋ, ý)

	print([train_acc, test_acc])
