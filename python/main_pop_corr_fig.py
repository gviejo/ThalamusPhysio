

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import time
import os, sys
import ipyparallel
import matplotlib.cm as cm

###############################################################################################################
# TO LOAD
###############################################################################################################
files = os.listdir("../data/corr_pop")

ywak = []
yrem = []
yrip = []

for f in files:
	data = cPickle.load(open("../data/corr_pop/"+f, 'rb'))
	theta_wake_corr = data['theta_wake_corr']
	rip_corr 		= data['rip_corr']
	theta_rem_corr 	= data['theta_rem_corr']
	index = np.tril_indices(len(rip_corr[0]))
	rip = np.vstack([rip_corr[0][index],rip_corr[1][index]]).transpose()
	wake = []
	for i in range(len(theta_wake_corr)):
		np.fill_diagonal(theta_wake_corr[i][1], 1.0)
		index = np.tril_indices(len(theta_wake_corr[i][0]))
		wake.append(np.vstack([theta_wake_corr[i][0][index],theta_wake_corr[i][1][index]]).transpose())
	wake = np.vstack(wake)
	rem = []
	for i in range(len(theta_rem_corr)):
		np.fill_diagonal(theta_rem_corr[i][1], 1.0)
		index = np.tril_indices(len(theta_rem_corr[i][0]))
		rem.append(np.vstack([theta_rem_corr[i][0][index],theta_rem_corr[i][1][index]]).transpose())
	rem = np.vstack(rem)

	
	# remove nan
	rem = rem[~np.isnan(rem[:,1])]
	wake = wake[~np.isnan(wake[:,1])]
	rip = rip[~np.isnan(rip[:,1])]

	# restrict to less than 3 second
	rem = rem[rem[:,0] <= 3.0,:]
	wake = wake[wake[:,0] <= 3.0,:]
	rip = rip[rip[:,0] <= 3.0,:]

	# average rip corr
	bins = np.arange(0.1, 3.0, 0.1)
	# bins[0] = 0.000001
	

	index_rip = np.digitize(rip[:,0], bins)
	index_wake = np.digitize(wake[:,0], bins)
	index_rem = np.digitize(rem[:,0], bins)
	mean_rip, mean_wake, mean_rem = ([], [], [])
	for i in range(1, len(bins)):
		mean_rip.append(np.mean(rip[index_rip == i,1]))
		mean_wake.append(np.mean(wake[index_wake == i,1]))
		mean_rem.append(np.mean(rem[index_rem == i,1]))

	# bad
	# 
	# 
	# plot(xt, mean_rip, 'o-', label = 'rip')
	# plot(xt, mean_wake, 'o-')
	# plot(xt, mean_rem, 'o-')
	# legend()
	# show()
	ywak.append(mean_wake)
	yrem.append(mean_rem)
	yrip.append(mean_rip)

bins[0] = 0.0
xt = bins[0:-1] + (bins[1] - bins[0])/2.

ywak = np.array(ywak)
yrem = np.array(yrem)
yrip = np.array(yrip)

meanywak = ywak[~np.isnan(ywak)[:,0]].mean(0)
meanyrem = yrem[~np.isnan(yrem)[:,0]].mean(0)
meanyrip = yrip[~np.isnan(yrip)[:,0]].mean(0)


plot(xt, meanywak, 'o-', label = 'theta(wake)')
plot(xt, meanyrem, 'o-', label = 'theta(rem)')
plot(xt, meanyrip, 'o-', label = 'ripple')

figure()
xt = list(xt[::-1]*-1.0)+[0.0]+list(xt)
meanywak = list(meanywak[::-1])+[1.0]+list(meanywak)
meanyrem = list(meanyrem[::-1])+[1.0]+list(meanyrem)
meanyrip = list(meanyrip[::-1])+[1.0]+list(meanyrip)

plot(xt, meanywak, 'o-', label = 'theta(wake)')
plot(xt, meanyrem, 'o-', label = 'theta(rem)')
plot(xt, meanyrip, 'o-', label = 'ripple')


legend()
xlabel('s')
ylabel('r')
show()

