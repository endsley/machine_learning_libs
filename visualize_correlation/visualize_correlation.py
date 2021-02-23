#!/usr/bin/env python

import numpy as np
from line_plot import *


def gen_sample(ρ):
	mean1 = [0, 0, 0, 0, 0, 0]
	cov1 = [[1, ρ, 0, 0, 0, 0],
			[ρ, 1, 0, 0, 0, 0],
			[0, 0, 1, ρ, 0, 0],
			[0, 0, ρ, 1, 0, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1]]
	
	X1 = np.random.multivariate_normal(mean1, cov1)
	
	return X1


for ρ in np.arange(0.1, 1.1, 0.1):
	X = np.empty((0, 6))
	for i in range(300):
		x = gen_sample(ρ)
		x = np.atleast_2d(x)
		X = np.vstack((X, x))
	pth = './ρ_at_%.1f'%ρ + '.png'
	textstr = 'ρ = %.2f'%ρ  #'\n'.join(( r'$\mu=%.2f$' % (0.1, ), r'$\mathrm{median}=%.2f$' % (0, ), r'$\sigma=%.2f$' % (33, )))
	lp = line_plot()
	lp.plot_line(X[:,0], X[:,1], textstr, 'X axis', 'Y axis', imgText='', outpath=pth)
	lp.clear()
