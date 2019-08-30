#!/usr/bin/env python
'''
	File name: main_ripp_mod.py
	Author: Guillaume Viejo
	Date created: 16/08/2017    
	Python Version: 3.5.2


'''
import sys
import numpy as np
import pandas as pd
import scipy.io
from functions import *
# from pylab import *
# import ipyparallel
from multiprocessing import Pool
import os
import neuroseries as nts
from time import time
from pylab import *
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import _pickle as cPickle

def compute_similarity(imap, times):
	distance = pd.Series(index = times, data = np.nan)
	# maxdistance
	x = np.atleast_2d(imap[:,:,0].flatten()).T - imap[:,:,0].flatten()
	y = np.atleast_2d(imap[:,:,1].flatten()).T - imap[:,:,1].flatten()
	maxd = np.max(np.sqrt(x**2 + y**2))
	for i in range(len(times)):
		x = np.atleast_2d(imap[:,i,0]).T - imap[:,i,0]
		y = np.atleast_2d(imap[:,i,1]).T - imap[:,i,1]
		d = np.sqrt(x**2 + y**2)
		d = d[np.triu_indices_from(d,1)]
		distance.loc[times[i]] = np.mean(d/maxd)
	# distance = distance.rolling(window=10,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=1)
	return distance

def compute_stability(imap, times):
	x = np.power(imap[:,1:,0] - imap[:,0:-1,0], 2)
	y = np.power(imap[:,1:,1] - imap[:,0:-1,1], 2)
	d = np.sqrt(x + y)
	d = pd.DataFrame(index = times[0:-1] + np.diff(times)/2, data = d.T)
	# d = d.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=2)
	return d.mean(1)


path = '../figures/figures_articles_v4/figure1/nohd_isomap_50ms_mixed_swr_rnd/'
files = np.sort([f for f in os.listdir(path) if '.pickle' in f and 'Mouse' in f])

allsimi_nohd = {}
allstab_nohd = {}

for f in files[0:-1]:
	print(f)
	data = cPickle.load(open(path+f, 'rb'))

	times = data['times']
	
	swrstab = []
	swrsimi = []
	rndstab = []
	rndsimi = []

	for n in data['imaps'].keys():
		iswr = data['imaps'][n]['swr']		
		swrstab.append(compute_stability(iswr, times))
		swrsimi.append(compute_similarity(iswr, times))
		irnd = data['imaps'][n]['rnd']
		rndstab.append(compute_stability(irnd, times))
		rndsimi.append(compute_similarity(irnd, times))

	swrstab = pd.concat(swrstab, 1)
	swrsimi = pd.concat(swrsimi, 1)
	rndstab = pd.concat(rndstab, 1)
	rndsimi = pd.concat(rndsimi, 1)
	
	swrstab = (swrstab - rndstab) / rndstab
	swrsimi = (swrsimi - rndsimi) / rndsimi

	# swrstab = swrstab.apply(scipy.stats.zscore)
	# swrsimi = swrsimi.apply(scipy.stats.zscore)

	allstab_nohd[f.split(".")[0]] = swrstab.mean(1)
	allsimi_nohd[f.split(".")[0]] = swrsimi.mean(1)
	

allsimi_nohd = pd.DataFrame.from_dict(allsimi_nohd)
allstab_nohd = pd.DataFrame.from_dict(allstab_nohd)


figure()
subplot(121)
plot(allstab_nohd, alpha = 0.5)
plot(allstab_nohd.mean(1), color = 'black', linewidth = 2)
title("stability")
subplot(122)
plot(allsimi_nohd, alpha = 0.5)
plot(allsimi_nohd.mean(1), color = 'black', linewidth = 2)
title("similarity")
show()



sys.exit()

################################################################################################
# comparing with HD
################################################################################################
path = '../figures/figures_articles_v4/figure1/good_100ms_pickle/'
files = [f for f in os.listdir(path) if '.pickle' in f and 'Mouse' in f]

files.remove("Mouse17-130129.pickle")

allsimi_hd = {}
allstab_hd = {}

for f in files:
	data = cPickle.load(open(path+f, 'rb'))

	swrsimi = []
	swrstab = []

	for n in data['swr'].keys():		
		iswr		= data['swr'][n]['iswr']
		rip_tsd		= data['swr'][n]['rip_tsd']		
		times 		= data['swr'][n]['times']

		swrsimi.append(compute_similarity(iswr, times))
		swrstab.append(compute_stability(iswr, times))
	
	swrsimi = pd.concat(swrsimi, 1)
	swrstab = pd.concat(swrstab, 1)

	rndsimi = []
	rndstab	= []

	for n in data['rnd'].keys():
		irand 		= data['rnd'][n]['irand']

		rndsimi.append(compute_similarity(irand, times))
		rndstab.append(compute_stability(irand, times))

	rndsimi = pd.concat(rndsimi, 1)
	rndstab = pd.concat(rndstab, 1)
		
	allsimi_hd[f.split(".")[0]] = (swrsimi.mean(1) - rndsimi.mean(1)) / rndsimi.mean(1)
	allstab_hd[f.split(".")[0]] = (swrstab.mean(1) - rndstab.mean(1)) / rndstab.mean(1)

allsimi_hd = pd.DataFrame.from_dict(allsimi_hd)
allstab_hd = pd.DataFrame.from_dict(allstab_hd)


figure()

subplot(121)
plot(allsimi_hd.mean(1), label = 'hd')
plot(allsimi_nohd.mean(1), label = 'no-hd')
legend()
title("similarity")

subplot(122)
plot(allstab_hd.mean(1), label = 'hd')
plot(allstab_nohd.mean(1), label = 'no-hd')
legend()
title("stability")

show()