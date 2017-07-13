import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle

###############################################################################################################
# LOADING DYNAMICAL SYSTEM
###############################################################################################################
dynamical_system = cPickle.load(open('../data/dynamical_system.pickle', 'rb'))
Mskew 	= dynamical_system['Mskew']
Msymm 	= dynamical_system['Msym']
x 		= dynamical_system['x']
dx 		= dynamical_system['dx']
times	= dynamical_system['times']

###############################################################################################################
# SIMULATION
###############################################################################################################
old_x = np.vstack(np.random.rand(x.shape[1]))
old_x = np.vstack(x[0])

dt = (times[1]-times[0])

Xt = [old_x.flatten()]

M = Mskew + Msymm

for i in range(2000):
	new_x = old_x + dt * (np.dot(Mskew, old_x))
	Xt.append(new_x.flatten())
	old_x = new_x

Xt = np.array(Xt)

plot(Xt)
show()

