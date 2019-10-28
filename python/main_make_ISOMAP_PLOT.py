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
from pylab import *
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

def compute_stability(imap, times):
	x = np.power(imap[:,1:,0] - imap[:,0:-1,0], 2)
	y = np.power(imap[:,1:,1] - imap[:,0:-1,1], 2)
	d = np.sqrt(x + y)
	d = pd.DataFrame(index = times[0:-1] + np.diff(times)/2, data = d.T)
	# d = d.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1)
	return d.mean(1)

def compute_similarity(imap, times):
	distance = pd.Series(index = times, data = np.nan)
	for i in range(len(times)):
		x = np.atleast_2d(imap[:,i,0]).T - imap[:,i,0]
		y = np.atleast_2d(imap[:,i,1]).T - imap[:,i,1]
		d = np.sqrt(x**2 + y**2)
		d = d[np.triu_indices_from(d,1)]
		distance.loc[times[i]] = np.mean(d)
	# distance = distance.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1)
	return distance

######################################################################################################
# HD
######################################################################################################
path = '../figures/figures_articles_v4/figure1/hd_isomap_40ms_mixed_swr_rnd_wake/'
files = np.sort(os.listdir(path))

stabhd = {}
simihd = {}

for f in files:
	data = cPickle.load(open(path+f, 'rb'))
	times = data['times']
	stability = []
	similarity = []
	stabswr = []
	stabrnd	= []
	simiswr = []
	simirnd = []
	for n in data['imaps'].keys():
		iswr = data['imaps'][n]['swr']
		irnd = data['imaps'][n]['rnd']
		iwak = data['imaps'][n]['wak']
		
		stabswr.append(compute_stability(iswr, times))
		stabrnd.append(compute_stability(irnd, times))

		simiswr.append(compute_similarity(iswr, times))
		simirnd.append(compute_similarity(irnd, times))

	stabswr = pd.concat(stabswr, 1)
	stabrnd = pd.concat(stabrnd, 1)
	stability = (stabswr.mean(1) - stabrnd.mean(1)) / stabrnd.mean(1)
	# stability = stabswr.mean(1)

	simiswr = pd.concat(simiswr, 1)
	simirnd = pd.concat(simirnd, 1)
	similarity = (simiswr.mean(1) - simirnd.mean(1)) / simirnd.mean(1)
	# similarity = simiswr.mean(1)
	
	stabhd[f.split(".")[0]] = stability
	simihd[f.split(".")[0]] = similarity

stabhd = pd.DataFrame.from_dict(stabhd)
simihd = pd.DataFrame.from_dict(simihd)

# stabhd = stabhd.apply(scipy.stats.zscore)
# simihd = simihd.apply(scipy.stats.zscore)

######################################################################################################
# NO HD
######################################################################################################
path = '../figures/figures_articles_v4/figure1/nohd_isomap_20ms_mixed_swr_rnd_wake/'
files = np.sort(os.listdir(path))

stabnohd = {}
siminohd = {}

for f in files:
	data = cPickle.load(open(path+f, 'rb'))
	times = data['times']
	stability = []
	similarity = []
	stabswr = []
	stabrnd	= []
	simiswr = []
	simirnd = []
	for n in data['imaps'].keys():
		iswr = data['imaps'][n]['swr']
		irnd = data['imaps'][n]['rnd']
		iwak = data['imaps'][n]['wak']
		
		stabswr.append(compute_stability(iswr, times))
		stabrnd.append(compute_stability(irnd, times))

		simiswr.append(compute_similarity(iswr, times))
		simirnd.append(compute_similarity(irnd, times))


	stabswr = pd.concat(stabswr, 1)
	stabrnd = pd.concat(stabrnd, 1)
	stability = (stabswr.mean(1) - stabrnd.mean(1)) / stabrnd.mean(1)

	simiswr = pd.concat(simiswr, 1)
	simirnd = pd.concat(simirnd, 1)
	similarity = (simiswr.mean(1) - simirnd.mean(1)) / simirnd.mean(1)
	
	stabnohd[f.split(".")[0]] = stability
	siminohd[f.split(".")[0]] = similarity

stabnohd = pd.DataFrame.from_dict(stabnohd)
siminohd = pd.DataFrame.from_dict(siminohd)

stabnohd = stabnohd.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=2)
siminohd = siminohd.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=2)

figure()
subplot(221)
plot(stabhd, alpha = 0.5)
plot(stabhd.mean(1), color = 'black', linewidth = 2)
title("stability")
subplot(222)
plot(simihd, alpha = 0.5)
plot(simihd.mean(1), color = 'black', linewidth = 2)
title("similarity")
subplot(223)
plot(stabnohd, alpha = 0.5)
plot(stabnohd.mean(1), color = 'black', linewidth = 2)
subplot(224)
plot(siminohd, alpha = 0.5)
plot(siminohd.mean(1), color = 'black', linewidth = 2)

figure()
subplot(121)
plot(stabhd.mean(1), linewidth = 2, label = 'hd')
plot(stabnohd.mean(1), linewidth = 2, label = 'no-hd')
legend()
title("stability")
subplot(122)
plot(simihd.mean(1), linewidth = 2, label = 'hd')
plot(siminohd.mean(1), linewidth = 2, label = 'no-hd')
legend()
title("similarity")
show()




tosave = {	'stab':{'hd':stabhd,'nohd':stabnohd},
			'simi':{'hd':simihd, 'nohd':siminohd}			
			}

cPickle.dump(tosave, open('../figures/figures_articles_v4/figure2/STABILITY_ISOMAP.pickle', 'wb'))




sys.exit()
######################################################################################################
# HD 100 ms 0.75
######################################################################################################
path = '../figures/figures_articles_v4/figure1/good_100ms_pickle/'
# path = '../figures/figures_articles_v4/figure1/'
files = np.sort([f for f in os.listdir(path) if 'Mouse' in f and 'pickle' in f])

stabhd100 = {}
simihd100 = {}

for f in files[0:-1]:
	data = cPickle.load(open(path+f, 'rb'))
	stability = []
	similarity = []
	stabswr = []
	stabrnd	= []
	simiswr = []
	simirnd = []

	for n in data['swr'].keys():
		iswr = data['swr'][n]['iswr']
		times = data['swr'][n]['times']
		stabswr.append(compute_stability(iswr, times))
		simiswr.append(compute_similarity(iswr, times))

	stabswr = pd.concat(stabswr, 1)
	simiswr = pd.concat(simiswr, 1)

	for n in data['rnd'].keys():
		irnd = data['rnd'][n]['irand']			
		stabrnd.append(compute_stability(irnd, times))		
		simirnd.append(compute_similarity(irnd, times))
	
	stabrnd = pd.concat(stabrnd, 1)
	simirnd = pd.concat(simirnd, 1)

	stability = (stabswr.mean(1) - stabrnd.mean(1)) / stabrnd.mean(1)
	similarity = (simiswr.mean(1) - simirnd.mean(1)) / simirnd.mean(1)
		
	stabhd100[f.split(".")[0]] = stability
	simihd100[f.split(".")[0]] = similarity

stabhd100 = pd.DataFrame.from_dict(stabhd100)
simihd100 = pd.DataFrame.from_dict(simihd100)


######################################################################################################
# HD 50 ms 0.75
######################################################################################################
path = '../figures/figures_articles_v4/figure1/'
files = np.sort([f for f in os.listdir(path) if 'Mouse' in f and 'pickle' in f])

stabhd50 = {}
simihd50 = {}

for f in files[0:-1]:
	data = cPickle.load(open(path+f, 'rb'))
	stability = []
	similarity = []
	stabswr = []
	stabrnd	= []
	simiswr = []
	simirnd = []

	for n in data['swr'].keys():
		iswr = data['swr'][n]['iswr']
		times = data['swr'][n]['times']
		stabswr.append(compute_stability(iswr, times))
		simiswr.append(compute_similarity(iswr, times))

	stabswr = pd.concat(stabswr, 1)
	simiswr = pd.concat(simiswr, 1)

	for n in data['rnd'].keys():
		irnd = data['rnd'][n]['irand']			
		stabrnd.append(compute_stability(irnd, times))		
		simirnd.append(compute_similarity(irnd, times))
	
	stabrnd = pd.concat(stabrnd, 1)
	simirnd = pd.concat(simirnd, 1)

	stability = (stabswr.mean(1) - stabrnd.mean(1)) / stabrnd.mean(1)
	similarity = (simiswr.mean(1) - simirnd.mean(1)) / simirnd.mean(1)
		
	stabhd50[f.split(".")[0]] = stability
	simihd50[f.split(".")[0]] = similarity

stabhd50 = pd.DataFrame.from_dict(stabhd50)
simihd50 = pd.DataFrame.from_dict(simihd50)


figure()
subplot(221)
plot(stabhd.mean(1), label = '20 ms | 25 overlap | 1sd smooth')
legend()
title("stability")
subplot(222)
plot(simihd.mean(1), label = '20 ms | 25 overlap | 1sd smooth')
title("similarity")
subplot(223)
plot(stabhd50.mean(1), label = '50 ms | 75 overlap | 4sd smooth')
plot(stabhd100.mean(1), label = '100 ms | 75 overlap | 4sd smooth')
legend()
subplot(224)
plot(simihd50.mean(1), label = '50 ms | 75 overlap | 4sd smooth')
plot(simihd100.mean(1), label = '100 ms | 75 overlap | 4sd smooth')
show()