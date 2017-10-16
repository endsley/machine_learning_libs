#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
#from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt
import numpy.matlib
from sklearn.metrics.cluster import normalized_mutual_info_score
import pickle
import sklearn
import time 
import matplotlib.pyplot as plt
import matplotlib 
from PIL import Image
colors = matplotlib.colors.cnames
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing




class imglib():
	def __init__(self, imgfile):
		im = Image.open(imgfile)
		
		rgb_im = im.convert('RGB')
		self.Img_3d_array = np.asarray(rgb_im)
		self.before_preprocess_data = np.empty((0, 3), dtype=np.uint8)
		data = np.empty((0, 3))
		data_dic = {}
		
		for i in range(self.Img_3d_array.shape[0]):
			for j in range(self.Img_3d_array.shape[1]):
				data_dic[ str(self.Img_3d_array[i,j]) ] = self.Img_3d_array[i,j]
		
		for i,j in data_dic.items():
			self.before_preprocess_data = np.vstack((self.before_preprocess_data, j))
			data = np.vstack((data, j))
		
		self.data = preprocessing.scale(data)
		
		d_matrix = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='euclidean')
		self.orig_sigma = np.median(d_matrix)

		print self.data.shape
		import pdb; pdb.set_trace()	


#	def update_allocation_original_img(self):
#		for m in range(len(self.orig_label)):
#			self.data_alloc[str(self.before_preprocess_data[m,:])] = self.orig_label[m]
#
#	
#		self.out_img = np.zeros(self.Img_3d_array.shape[0:2])
#		for i in range(self.Img_3d_array.shape[0]):
#			for j in range(self.Img_3d_array.shape[1]):
#				self.out_img[i,j] = 255*(self.data_alloc[str(self.Img_3d_array[i,j])] - 1)
#	
#	def draw_original_allocation(self):
#		self.out_img = self.out_img.astype('uint8')
#		img = Image.fromarray(self.out_img, 'L') 
#		img.save('img/my.png') 
#		img.show()
#
#	def draw_alt_image(self):
#		data_alloc2 = {}
#		for m in range(len(self.alt_allocation)):
#			data_alloc2[str(self.before_preprocess_data[m,:])] = self.alt_allocation[m]
#	
#		alt_img = np.zeros(self.Img_3d_array.shape[0:2])
#		for i in range(self.Img_3d_array.shape[0]):
#			for j in range(self.Img_3d_array.shape[1]):
#				alt_img[i,j] = 255*(data_alloc2[str(self.Img_3d_array[i,j])] - 1)
#	
#	
#		alt_img = alt_img.astype('uint8')
#		img = Image.fromarray(alt_img, 'L') 
#		img.save('img/alternative_img.png') 
#		img.show()
#
#
#
#
#	def plot_original_clustering_results(self):
#		db = self.ASC.db
#
#		X = db['data']
#		fig = plt.figure()
#		ax = fig.add_subplot(111, projection='3d')
#		Uq_a = np.unique(self.orig_label)
#		
#		group1 = X[self.orig_label == Uq_a[0]]
#		group2 = X[self.orig_label == Uq_a[1]]
#		
#		ax.scatter(group1[:,0], group1[:,1], group1[:,2], c='b', marker='o')
#		ax.scatter(group2[:,0], group2[:,1], group2[:,2], c='r', marker='x')
#		ax.set_xlabel('Feature 1')
#		ax.set_ylabel('Feature 2')
#		ax.set_zlabel('Feature 3')
#		ax.set_title('Original Clustering')
#		
#		plt.show()
#
#	def set_up_alternative(self):
#		im = Image.open("data_sets/Flower2.png")
#		rgb_im = im.convert('RGB')
#		Img_3d_array = np.asarray(rgb_im)
#		before_preprocess_data = np.empty((0, 3), dtype=np.uint8)
#		original_allocation = np.empty((0, 1))
#		data = np.empty((0, 3))
#		data_dic = {}
#		pixel_allocation = self.out_img/255
#		
#		
#		#	Compress data into identical colors, due to too many repeated colors
#		for i in range(Img_3d_array.shape[0]):
#			for j in range(Img_3d_array.shape[1]):
#				data_dic[ str(Img_3d_array[i,j]) ] = [Img_3d_array[i,j], pixel_allocation[i,j]]
#		
#		for i,j in data_dic.items():
#			before_preprocess_data = np.vstack((before_preprocess_data, j[0]))
#			data = np.vstack((data, j[0]))
#			original_allocation = np.vstack((original_allocation, j[1]))
#			
#		self.data = preprocessing.scale(data)
#		test_base.predefine_orig_clustering(self)
#
#
#
#	def output_alt_info(self):
#		db = self.ASC.db
#
#
#		new_cost = db['cf'].Final_calc_cost(db['W_matrix'])
#		CQ = db['cf'].cluster_quality(db)
#		Altness = db['cf'].alternative_quality(db)
#
#		out_str  = "--- " +			 					str(db['run_alternative_time'] 	)	+	 " seconds ---" +	'\n'
#		out_str += '\tq used : ' +			 			str(self.q 						)	+	 '\n'
#		out_str += '\torig num clusters :' +				str(self.orig_c_num 			)	+	 '\n'
#		out_str += '\talt num clusters :' +				str(self.c_num 					)	+	 '\n'
#		out_str += '\tsigma used : '+			 			str(self.sigma_used  			)	+	 '\n'
#		#out_str += '\tsigma_ratio : ' +			 		str(self.sigma_ratio 			)	+	 '\n'
#		out_str += '\tmedian of pairwise distance : ' + 	str(self.median_pair_dist 		)	+	 '\n'
#		out_str += '\tlambda used : '+			 		str(self.lambda_used 			)	+	 '\n'
#		#out_str += '\tlambda_ratio : ' +			 		str(self.lambda_ratio 			)	+	 '\n'
#		#out_str += '\tHSIC ratio : ' +			 		str(self.hsic_ratio 			)	+	 '\n'
#		out_str += '\tCost : ' +			 				str(new_cost 					)	+	 '\n'
#		out_str += '\tCQ : ' +			 				str(CQ 							)	+	 '\n'
#		out_str += '\tAltness : ' +			 			str(Altness 					)	+	 '\n\n'
#		out_str += '\tRun Hash : ' 				+ str(db['run_hash'] 	) + '\n'
#		out_str += '\t:::::::::: Cut and Paste :::::::\n'
#
#
#		out_str += '\tBest\tMNI\tCQ\tAlt\tCost\tTime\tNMI\tCost\tTime\n'
#		out_str += '\t' +  '-'
#		out_str += '\t' +  '-'
#		out_str += '\t' +  str(np.round(CQ, 3)  ) 
#		out_str += '\t' +  '-' 
#		out_str += '\t' +  str(np.round(new_cost, 3)) 
#		out_str += '\t' +  str(np.round(db['run_alternative_time'], 3)) 
#		out_str += '\t' + '-' + '\t' + str(np.round(new_cost,3)) + '\t' + str(np.round(db['run_alternative_time'], 3)) + '\n'
#
#
#		self.write_out(out_str)
#
#
#
#
#
#
#
#
#
#	def perform_default_run(self):
#		self.draw_original_allocation()
#		self.set_up_alternative()
#		self.calc_alt_cluster()
#		self.output_alt_info()
#		self.draw_alt_image()
#		self.save_result_matrices()
#
#
#	def run_with_W_0(self, technique='SM', pickle_count=0):
#		self.write_out(':::::::::::::   W0 of Flower with ' + technique + ' :::::::::::::\n')
#		debug_info = False
#		self.set_up_alternative()
#
#		self.ASC.set_values('W_opt_technique', technique)
#		self.ASC.set_values('init_W_from_pickle',True)
#		self.ASC.set_values('pickle_count',pickle_count)		
#
#		self.calc_alt_cluster()
#		self.output_alt_info()
#		self.save_result_matrices()
#		import pdb; pdb.set_trace()
#
#
#	def maintain_average(self, avg_dict):
#		db = self.ASC.db
#	
#		new_cost = db['cf'].Final_calc_cost(db['W_matrix'])
#		CQ = db['cf'].cluster_quality(db)
#		Altness = db['cf'].alternative_quality(db)
#		alt_orig = np.round(normalized_mutual_info_score(self.alt_allocation, self.orig_label),3)
#
#		avg_dict['Alt_V_Alt_NMI'].append(0)
#		avg_dict['Alt_Vs_Orig_NMI'].append(alt_orig)
#		avg_dict['Alt'].append(Altness)
#		avg_dict['CQ'].append(CQ)
#		avg_dict['Cost'].append(new_cost)
#		avg_dict['Time'].append(db['run_alternative_time'])
#
#
#	def random_initializations(self, n_inits, technique):
#		self.write_out(':::::::::::::   Random Initialization of Flower with ' + technique + ' :::::::::::::\n')
#		debug_info = False
#		avg_dict = {}
#		avg_dict['Alt_V_Alt_NMI'] = []
#		avg_dict['Alt_Vs_Orig_NMI'] = []
#		avg_dict['Alt'] = []
#		avg_dict['CQ'] = []
#		avg_dict['Cost'] = []
#		avg_dict['Time'] = []
#
#
#		for pinit in range(n_inits):
#			sys.stdout.write("\rCurrently %dth random run." % (pinit))
#			sys.stdout.flush()
#
#			self.set_up_alternative()
#			
#			self.ASC.set_values('W_opt_technique', technique)
#			self.ASC.set_values('init_W_from_pickle',True)
#			self.ASC.set_values('pickle_count',pinit)
#			self.calc_alt_cluster()
#
#			self.maintain_average(avg_dict)
#			#self.plot_result()
#
#
#		self.output_random_initialize(avg_dict, technique, 0)
#
#
#
#	def ran_single_from_pickle_init(self, technique, pinit):
#		self.write_out(':::::::::::::   Single from Pickle of Flower with ' + technique + ' , ' + str(pinit) + ' :::::::::::::\n')
#		debug_info = False
#		avg_dict = {}
#		avg_dict['Alt_V_Alt_NMI'] = []
#		avg_dict['Alt_Vs_Orig_NMI'] = []
#		avg_dict['Alt'] = []
#		avg_dict['CQ'] = []
#		avg_dict['Cost'] = []
#		avg_dict['Time'] = []
#
#
#		self.set_up_alternative()
#		
#		self.ASC.set_values('W_opt_technique', technique)
#		self.ASC.set_values('init_W_from_pickle',True)
#		self.ASC.set_values('pickle_count',pinit)
#		self.calc_alt_cluster()
#
#		self.maintain_average(avg_dict)
#		self.output_random_initialize(avg_dict, technique, 0)
#
#		self.draw_alt_image()
#
#	def perform_default_run_full(self, debug_info=False):
#		test_base.perform_default_run(self, debug_info)
#		D.db['cf'].test_2nd_order(D.db)

iL = imglib('./img/mario.png')


