#!/usr/bin/env python

import numpy as np
import scipy.stats
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
from matplotlib.patches import Circle
from numpy import genfromtxt


#	This example determines the optimal sigmal value of 
#	a Gaussian kernel based on the highest entropy

def suggest_σs(D, X):
	all_best_σs = []
	for m in range(D.shape[0]):
		max_entropy = 0
		best_σ = 0
		best_ratio = 0

		single_row = D[m,:]
		mpd = np.median(single_row)


		for ratio in np.arange(0.01,3,0.01):
			σ = ratio*mpd
			kvals = np.exp((-single_row*single_row)/(2*σ*σ))
			[a,b] = np.histogram(kvals, bins=20, range=(0,1))
			distribution = a/np.sum(a)
			H = scipy.stats.entropy(distribution)
			
			#print('\tratio : %.3f \t σ: %.3f \t Entropy : %.3f'%(ratio, σ, H))
			if H > max_entropy: 
				max_entropy = H
				best_σ = σ
				best_ratio = ratio
			if max_entropy > 0 and H == 0: break

		print('Sample %d , Best σ %.4f'%(m,best_σ))
		all_best_σs.append(best_σ)

	return all_best_σs

	import pdb; pdb.set_trace()


def suggest_σ(X):
	D = sklearn.metrics.pairwise.pairwise_distances(X)
	distances = np.reshape(D, (D.shape[0]*D.shape[0], 1))
	mpd = np.median(distances)
	max_entropy = 0
	best_σ = 0
	best_ratio = 0
	
	for ratio in np.arange(0.01,3,0.01):
		σ = ratio*mpd
		kvals = np.exp((-distances*distances)/(2*σ*σ))
		[a,b] = np.histogram(kvals, bins=20, range=(0,1))
		distribution = a/np.sum(a)
		H = scipy.stats.entropy(distribution)
		
		print('%.3f \t %.3f \t %.3f'%(ratio, σ, H))
		if H > max_entropy: 
			max_entropy = H
			best_σ = σ
			best_ratio = ratio
		if max_entropy > 0 and H == 0: break
		
			
	return [best_σ, best_ratio, max_entropy]

if __name__ == '__main__':
	Y = None
	#X = genfromtxt('../datasets/data_4.csv', delimiter=',')
	X = genfromtxt('../datasets/moon.csv', delimiter=',')
	Y = genfromtxt('../datasets/moon_label.csv', delimiter=','); k = 2
	#X = genfromtxt('../datasets/spiral.csv', delimiter=',')
	#Y = genfromtxt('../datasets/spiral_label.csv', delimiter=','); k = 3
	#X = np.array([[-1,-1],[1,1],[1.01,0.99]])
	

	#d = 30
	#A = 0.4*np.random.randn(10,2) + np.array([-d,-d])
	#B = 0.4*np.random.randn(10,2) + np.array([d,d])
	#C = 0.4*np.random.randn(10,2) + np.array([d,-d])
	#D = 0.4*np.random.randn(10,2) + np.array([-d,d])
	#X = np.vstack((A,B,C,D))



#	D = sklearn.metrics.pairwise.pairwise_distances(X)
#	all_best_σs = suggest_σs(D, X)
#	Kσ = np.outer(all_best_σs, all_best_σs)
#	kvals = np.exp(-D*D*(1/Kσ))
#
#	allocation = SpectralClustering(k, affinity=kvals).fit_predict(X)
#	nmi = normalized_mutual_info_score(allocation, Y)
#	print('nmi : %.3f'%nmi)
#
#	import pdb; pdb.set_trace()










	[best_σ, best_ratio, max_entropy] = suggest_σ(X)
	print('Best σ : %.3f'%best_σ)
	print('Best σ ratio : %.3f'%best_ratio)
	print('maximum entropy : %.3f'%max_entropy)

	if Y is not None:
		Vgamma = 1/(2*best_σ*best_σ)
		allocation = SpectralClustering(k, gamma=Vgamma).fit_predict(X)
		nmi = normalized_mutual_info_score(allocation, Y)
		print('nmi : %.3f'%nmi)
   

	plt.plot(X[:,0], X[:,1], 'x')

	for i in range(X.shape[0]):
		#if np.random.rand() > 0.2: continue
		circle = plt.Circle((X[i,0], X[i,1]), radius=best_σ, fill=None)
		plt.gca().add_patch(circle)

	plt.show()

