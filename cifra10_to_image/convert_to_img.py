#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random
import os
 
def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict
def get_data(file):
	absFile = os.path.abspath("data/"+file)
	dict = unpickle(absFile)
	#for key in dict.keys():
	#	print(key)
	#print("Unpacking {}".format(dict[b'batch_label']))
	X = np.asarray(dict[b'data'].T).astype("uint8")
	Yraw = np.asarray(dict[b'labels'])

	#Y = np.zeros((10,10000))
	#for i in range(10000):
	#	Y[Yraw[i],i] = 1
	names = np.asarray(dict[b'filenames'])
	return X,Yraw,names
def visualize_image(X,Y,names,id):
	rgb = X[:,id]
	pth = './data/' + str(Y[id])
	
	if not os.path.exists(pth):
		os.mkdir(pth)

	#print(rgb.shape)
	img = rgb.reshape(3,32,32).transpose([1, 2, 0])
	#print(img.shape)
	plt.imshow(img)
	plt.title(names[id])
	#print(Y[id])
	#plt.show()
	dir = os.path.abspath(pth)
	plt.savefig(dir+"/"+names[id].decode('ascii'))


X,Y,names = get_data('./data_batch_1')

for indx in range(1,10001):
	visualize_image(X,Y,names, indx)
