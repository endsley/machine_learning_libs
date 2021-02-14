#!/usr/bin/env python

import imageio		#conda install -c conda-forge imageio
import os
import sys


def gif_from_img(imgList, outpath, duration):
	frames = []
	for filename in imgList:
		if filename.endswith(".png"):
			frames.append(imageio.imread(filename))
		
	# Save them as frames into a gif 
	kargs = { 'duration': duration }
	imageio.mimsave(outpath, frames, 'GIF', **kargs)

if __name__ == "__main__":
	imgList = ['imgs/22.png', 'imgs/30.png', 'imgs/40.png', 'imgs/60.png']
	gif_from_img(imgList, 'combined.gif', 1)
