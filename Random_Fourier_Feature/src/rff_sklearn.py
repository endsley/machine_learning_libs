#!/usr/bin/env python

import numpy as np
import sys
import sklearn.metrics
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)


X = np.array([[0, 0], [1, 1], [1, 0], [0, 1],[2,1]])

rbf_feature = RBFSampler(gamma=1, n_components=1200, random_state=None)
X_features = rbf_feature.fit_transform(X)

print(X_features.shape)
print(X_features.dot(X_features.T))




rbk = sklearn.metrics.pairwise.rbf_kernel(X, gamma=1)
print('\n')
print( rbk )
