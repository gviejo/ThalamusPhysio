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


path = '../figures/figures_articles_v4/figure1/good_100ms_pickle/'
files = [f for f in os.listdir(path) if '.pickle' in f and 'Mouse' in f]

files.remove("Mouse17-130129.pickle")

radius = []
velocity = []
order = []



for f in files:
	data = cPickle.load(open(path+f, 'rb'))

	swrvel 			= []
	swrrad 			= []
	for n in data['swr'].keys():
		iwak		= data['swr'][n]['iwak']
		iswr		= data['swr'][n]['iswr']
		rip_tsd		= data['swr'][n]['rip_tsd']
		rip_spikes	= data['swr'][n]['rip_spikes']
		times 		= data['swr'][n]['times']
		wakangle	= data['swr'][n]['wakangle']
		neurons		= data['swr'][n]['neurons']
		tcurves		= data['swr'][n]['tcurves']

		normwak = np.sqrt(np.sum(np.power(iwak,2), 1))
		normswr = np.sqrt(np.sum(np.power(iswr, 2), -1))
		swrrad.append(normswr)
		angswr = np.arctan2(iswr[:,:,1], iswr[:,:,0])
		angswr = (angswr + 2*np.pi)%(2*np.pi)

		
		for i in range(len(angswr)):
			a = np.unwrap(angswr[i])
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std=1.0)
			swrvel.append(np.abs(np.diff(c.values))/0.1)
	
	swrvel = np.array(swrvel)
	swrrad = np.vstack(swrrad)

	rndvel 			= []
	rndrad			= []
	for n in data['rnd'].keys():
		irand 		= data['rnd'][n]['irand']
		iwak2 		= data['rnd'][n]['iwak2']
		normrnd = np.sqrt(np.sum(np.power(irand,2), -1))
		rndrad.append(normrnd)
		angrnd = np.arctan2(irand[:,:,1], irand[:,:,0])
		angrnd = (angrnd + 2*np.pi)%(2*np.pi)

		for i in range(len(angrnd)):
			a = np.unwrap(angrnd[i])
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std=1.0)
			rndvel.append(np.abs(np.diff(c.values))/0.1)

	rndvel = np.array(rndvel)
	rndrad = np.vstack(rndrad)

	

	order.append(f.split(".")[0])
	tmp = pd.Series(index = times, data = (swrrad.mean(0) - rndrad.mean(0))/(rndrad.mean(0)))
	tmp = tmp.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std = 1.0)
	radius.append(tmp)
	tmp = pd.Series(index = times[0:-1]+np.diff(times)/2, data = (swrvel.mean(0) - rndvel.mean(0))/rndvel.mean(0))
	tmp = tmp.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std = 1.0)
	velocity.append(tmp)

velocity = pd.concat(velocity, 1)
radius = pd.concat(radius, 1)
velocity.columns = pd.Index(order)
radius.columns = pd.Index(order)


tosave = {'velocity':velocity,
			'radius':radius}


# cPickle.dump(tosave, open('../figures/figures_articles_v4/figure1/RING_DECODING.pickle', 'wb'))