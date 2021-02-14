
import numpy as np
from lib.path_tools import *
from lib.plot_clusters import *
from lib.create_gif import *

class kernel_herding_debug:
	def __init__(self, kh):
		self.kh = kh
		if self.kh.debug_mode == False: return
		if self.kh.data_name is None: 
			self.kh.debug_mode = False
			return

		ensure_path_exists('./results')
		ensure_path_exists('./results/' + self.kh.data_name)
		ensure_path_exists('./results/' + self.kh.data_name + '/sample_growth')
		remove_files('./results/' + self.kh.data_name + '/sample_growth/')

	def collect_result(self, n):
		merged_result = './results/' + self.kh.data_name + '/merged_results.gif'
		merged_resultF = './results/' + self.kh.data_name + '/merged_results_fast.gif'
		imgList = []
		p = './results/' + self.kh.data_name + '/sample_growth/'

		for i in range(n):
			f = p + str(i) + '.png'
			if file_exists(f):
				imgList.append(f)

		gif_from_img(imgList, merged_result, 2)
		gif_from_img(imgList, merged_resultF, 0.1)

	def save_results(self, oldError):
		if self.kh.debug_mode == False: return
		d = self.kh.d

		subSample = np.empty((0, d))
		subLabels = np.empty((0, 1))
		for i in self.kh.l:
			ẋ = self.kh.sSamples.subSamples[i]

			subSample = np.vstack((subSample, ẋ))
			subLabels = np.vstack((subLabels, self.kh.sSamples.Y_list[i][0:ẋ.shape[0], :]))

		title = str(subSample.shape[0]) + ' Samples'
		save_path = './results/' + self.kh.data_name + '/sample_growth/' + str(subSample.shape[0]) + '.png'

		cluster_plot(subSample, subLabels, title=title, save_path=save_path)

