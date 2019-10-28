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


####################################################################################
# HD 
####################################################################################
path = '../figures/figures_articles_v4/figure1/hd_isomap_30ms_mixed_swr_rnd_wake/'
files = [f for f in os.listdir(path) if '.pickle' in f and 'Mouse' in f]

files.remove("Mouse17-130129.pickle")

radius = []
velocity = []
order = []



for f in files:
	data = cPickle.load(open(path+f, 'rb'))

	swrvel 			= []
	swrrad 			= []
	rndvel 			= []
	rndrad 			= []
	times 			= data['times']
	for n in data['imaps'].keys():
		iwak		= data['imaps'][n]['wak']
		iswr		= data['imaps'][n]['swr']
		irnd 		= data['imaps'][n]['rnd']

		normwak = np.sqrt(np.sum(np.power(iwak,2), 1))
		normswr = np.sqrt(np.sum(np.power(iswr, 2), -1))
		normrnd = np.sqrt(np.sum(np.power(irnd,2), -1))

		swrrad.append((normswr.mean(0) - normrnd.mean(0))/normrnd.mean(0))
		# swrrad.append(normswr.mean(0))
		# rndrad.append(normrnd.mean(0))
		
		angswr = np.arctan2(iswr[:,:,1], iswr[:,:,0])
		angswr = (angswr + 2*np.pi)%(2*np.pi)
		angrnd = np.arctan2(irnd[:,:,1], irnd[:,:,0])
		angrnd = (angrnd + 2*np.pi)%(2*np.pi)

		tmp1 = []
		for i in range(len(angswr)):
			a = np.unwrap(angswr[i])
			b = pd.Series(index = times, data = a)
			c = b
			c = b.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=0.5)
			tmp1.append(np.abs(np.diff(c.values))/0.02)

		tmp2 = []
		for i in range(len(angrnd)):
			a = np.unwrap(angrnd[i])
			b = pd.Series(index = times, data = a)
			c = b
			c = b.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=0.5)
			tmp2.append(np.abs(np.diff(c.values))/0.02)
	
		tmp1 = np.array(tmp1)
		tmp2 = np.array(tmp2)
		
		swrvel.append((tmp1.mean(0) - tmp2.mean(0))/tmp2.mean(0))
		# swrvel.append(tmp1.mean(0))
		# rndvel.append(tmp2.mean(0))

	swrvel = np.array(swrvel)
	swrrad = np.array(swrrad)
	# rndvel = np.array(rndvel)
	# rndrad = np.array(rndrad)

	order.append(f.split(".")[0])
	tmp = pd.DataFrame(index = times, data = swrrad.T)
	# tmp = pd.Series(index = times, data = (swrrad.mean(0) - rndrad.mean(0))/(rndrad.mean(0)))
	tmp = tmp.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std = 0.5)
	radius.append(tmp.mean(1))
	# radius.append(tmp)
	tmp = pd.DataFrame(index = times[0:-1]+np.diff(times)/2, data = swrvel.T)
	# tmp = pd.Series(index = times[0:-1]+np.diff(times)/2, data = (swrvel.mean(0) - rndvel.mean(0))/rndvel.mean(0))
	tmp = tmp.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std = 0.5)
	velocity.append(tmp.mean(1))
	# velocity.append(tmp)

velocity = pd.concat(velocity, 1)
radius = pd.concat(radius, 1)
velocity.columns = pd.Index(order)
radius.columns = pd.Index(order)


tosave = {'velocity':velocity,
			'radius':radius}

figure()
subplot(121)
plot(velocity, linewidth = 1, alpha = 0.5, color = 'grey')
plot(velocity.mean(1), linewidth = 4, alpha = 1)
subplot(122)
plot(radius, linewidth = 1, alpha = 0.5, color = 'grey')
plot(radius.mean(1), linewidth = 4, alpha = 1)
show()

# cPickle.dump(tosave, open('../figures/figures_articles_v4/figure1/RING_DECODING_30ms.pickle', 'wb'))










sys.exit()

####################################################################################
# NO HD 
####################################################################################
path = '../figures/figures_articles_v4/figure1/nohd_isomap_20ms_mixed_swr_rnd_wake/'
files = [f for f in os.listdir(path) if '.pickle' in f and 'Mouse' in f]

# files.remove("Mouse17-130129.pickle")

radius = []
velocity = []
order = []



for f in files:
	data = cPickle.load(open(path+f, 'rb'))

	swrvel 			= []
	swrrad 			= []
	rndvel 			= []
	rndrad 			= []
	times 			= data['times']
	for n in data['imaps'].keys():
		iwak		= data['imaps'][n]['wak']
		iswr		= data['imaps'][n]['swr']
		irnd 		= data['imaps'][n]['rnd']

		normwak = np.sqrt(np.sum(np.power(iwak,2), 1))
		normswr = np.sqrt(np.sum(np.power(iswr, 2), -1))
		normrnd = np.sqrt(np.sum(np.power(irnd,2), -1))

		# swrrad.append((normswr.mean(0) - normrnd.mean(0))/normrnd.mean(0))
		swrrad.append(normswr.mean(0))
		rndrad.append(normrnd.mean(0))
		
		angswr = np.arctan2(iswr[:,:,1], iswr[:,:,0])
		angswr = (angswr + 2*np.pi)%(2*np.pi)
		angrnd = np.arctan2(irnd[:,:,1], irnd[:,:,0])
		angrnd = (angrnd + 2*np.pi)%(2*np.pi)

		tmp1 = []
		for i in range(len(angswr)):
			a = np.unwrap(angswr[i])
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std=1)
			tmp1.append(np.abs(np.diff(c.values))/0.02)

		tmp2 = []
		for i in range(len(angrnd)):
			a = np.unwrap(angrnd[i])
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std=1)
			tmp2.append(np.abs(np.diff(c.values))/0.02)
	
		tmp1 = np.array(tmp1)
		tmp2 = np.array(tmp2)
		
		# swrvel.append((tmp1.mean(0) - tmp2.mean(0))/tmp2.mean(0))
		swrvel.append(tmp1.mean(0))
		rndvel.append(tmp2.mean(0))

	swrvel = np.array(swrvel)
	swrrad = np.array(swrrad)
	rndvel = np.array(rndvel)
	rndrad = np.array(rndrad)

	order.append(f.split(".")[0])
	# tmp = pd.DataFrame(index = times, data = swrrad.T)
	tmp = pd.Series(index = times, data = (swrrad.mean(0) - rndrad.mean(0))/(rndrad.mean(0)))
	tmp = tmp.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std = 1)
	# radius.append(tmp.mean(1))
	radius.append(tmp)
	# tmp = pd.DataFrame(index = times[0:-1]+np.diff(times)/2, data = swrvel.T)
	tmp = pd.Series(index = times[0:-1]+np.diff(times)/2, data = (swrvel.mean(0) - rndvel.mean(0))/rndvel.mean(0))
	tmp = tmp.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std = 1)
	# velocity.append(tmp.mean(1))
	velocity.append(tmp)

velocity = pd.concat(velocity, 1)
radius = pd.concat(radius, 1)
velocity.columns = pd.Index(order)
radius.columns = pd.Index(order)


tosave = {'velocity':velocity,
			'radius':radius}

figure()
subplot(121)
plot(velocity, linewidth = 1, alpha = 0.5, color = 'grey')
plot(velocity.mean(1), linewidth = 4, alpha = 1)
subplot(122)
plot(radius, linewidth = 1, alpha = 0.5, color = 'grey')
plot(radius.mean(1), linewidth = 4, alpha = 1)
show()
