#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np

class line_plot:
	def __init__(self, title_font=16, xfont=16, yfont=16):
		self.title_font = title_font
		self.xfont = xfont
		self.yfont = yfont

	def plot_line(self, X, Y, title, xlabel, ylabel, imgText=None, outpath=None):
		self.add_plot(X,Y)
		self.set_title(title, fontsize=self.title_font)
		self.set_xlabel(xlabel, fontsize=self.xfont)
		self.set_ylabel(ylabel, fontsize=self.yfont)
		self.add_text(X, Y, imgText)

		if outpath is None: plt.show()
		else: plt.savefig(outpath)

	def add_text(self, X, Y, textstr, α=0.05, β=0.95):
		if textstr is None: return
		mX = np.min(X)
		mY = np.min(Y)

		xLoc = α*(np.max(X) - mX) + mX
		yLoc = β*(np.max(Y) - mY) + mY

		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		plt.text(xLoc, yLoc, textstr, fontsize=14, verticalalignment='top', bbox=props)



	#plt.axis('tight');
	def set_title(self, title, fontsize=16):
		plt.title(title, fontsize=fontsize)

	def set_xlabel(self, xlabel, fontsize=16):
		plt.xlabel(xlabel, fontsize=fontsize)

	def set_ylabel(self, ylabel, fontsize=16):
		plt.ylabel(ylabel, fontsize=fontsize)

	def add_plot(self, X,Y, color='blue'):
		#color='blue'        # specify color by name
		#color='g'           # short color code (rgbcmyk)
		#color='0.75'        # Grayscale between 0 and 1
		#color='#FFDD44'     # Hex code (RRGGBB from 00 to FF)
		#color=(1.0,0.2,0.3) # RGB tuple, values 0 to 1
		#color='chartreuse'; # all HTML color names supported

		plt.plot(X,Y, color=color)

	def show(self):

		plt.show()



if __name__ == "__main__":
	textstr = '\n'.join(( r'$\mu=%.2f$' % (0.1, ), r'$\mathrm{median}=%.2f$' % (0, ), r'$\sigma=%.2f$' % (33, )))
	x = np.linspace(0, 10, 1000)
	y = np.sin(x)

	lp = line_plot()
	lp.plot_line(x, y, 'Title Here', 'X axis', 'Y axis', imgText=textstr)#, outpath)
