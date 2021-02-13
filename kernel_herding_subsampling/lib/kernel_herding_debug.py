
import numpy as np

class kernel_herding_debug:
	def __init__(self, kh):
		self.kh = kh

	def save_results(self, oldError):
		if self.kh.debug_mode == False: return
		d = self.kh.d

		subSample = np.empty((0, d))
		subLabels = np.empty((0, 1))
		for i in self.kh.l:
			ẋ = self.kh.sSamples.subSamples[i]

			subSample = np.vstack((subSample, ẋ))
			subLabels = np.vstack((subLabels, self.kh.sSamples.Y_list[i][0:ẋ.shape[0], :]))

