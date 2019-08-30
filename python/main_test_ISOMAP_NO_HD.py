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



data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {ep:pd.DataFrame() for ep in ['wak', 'rem', 'sws']}


# session = 'Mouse17/Mouse17-130130'
session = 'Mouse12/Mouse12-120806'
# session = 'Mouse32/Mouse32-140822'
# session = 'Mouse12/Mouse12-120807'

generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
shankStructure 	= loadShankStructure(generalinfo)
if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
else:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')	
wake_ep 		= loadEpoch(data_directory+session, 'wake')
sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
sws_ep 			= loadEpoch(data_directory+session, 'sws')
rem_ep 			= loadEpoch(data_directory+session, 'rem')
sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
sws_ep 			= sleep_ep.intersect(sws_ep)	
rem_ep 			= sleep_ep.intersect(rem_ep)
rip_ep,rip_tsd 	= loadRipples(data_directory+session)
rip_ep			= sws_ep.intersect(rip_ep)	
rip_tsd 		= rip_tsd.restrict(sws_ep)
speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)
hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])


spikes 			= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}
neurons 		= np.sort(list(spikes.keys()))


####################################################################################################################
# binning data
####################################################################################################################
allrates		= {}

n_ex = 200
tmp = nts.Ts(rip_tsd.as_series().sample(n_ex, replace = False).sort_index()).index.values
rip_tsd = pd.Series(index = tmp, data = np.nan)
# rip_tsd = rip_tsd.iloc[0:200]

bins_size = [200,10,100]

####################################################################################################################
# BIN SWR
####################################################################################################################
@jit(nopython=True)
def histo(spk, obins):
	n = len(obins)
	count = np.zeros(n)
	for i in range(n):
		count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
	return count


bin_size = bins_size[2]	
left_bound = np.arange(-500-bin_size/2, 500 - bin_size/4,bin_size/4)
obins = np.vstack((left_bound, left_bound+bin_size)).T

times = obins[:,0]+(np.diff(obins)/2).flatten()
tmp = []

rip_spikes = {}
# sys.exit()
tmp2 = rip_tsd.index.values/1e3
for i, t in enumerate(tmp2):
	print(i, t)
	tbins = t + obins
	spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)
	rip_spikes[i] = {}
	for j in neurons:
		spks = spikes[j].as_units('ms').index.values
		spike_counts[j] = histo(spks, tbins)
		nspks = spks - t
		rip_spikes[i][j] = nspks[np.logical_and((spks-t)>=-500, (spks-t)<=500)]
	tmp.append(np.sqrt(spike_counts/(bins_size[-1])))

allrates['swr'] = tmp



####################################################################################################################
# BIN RANDOM
####################################################################################################################
tmp = []
n_ex = 200
rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, np.maximum(1,n_ex//len(sws_ep))) for i in sws_ep.index])))
tmp3 = rnd_tsd.index.values/1000

for i, t in enumerate(tmp3):
	print(i, t)
	tbins = t + obins
	spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)	
	for j in neurons:
		spks = spikes[j].as_units('ms').index.values	
		spike_counts[j] = histo(spks, tbins)
		nspks = spks - t
	
	tmp.append(np.sqrt(spike_counts/(bins_size[-1])))
	
allrates['rnd'] = tmp



####################################################################################################################
# SMOOTHING
####################################################################################################################

tmp3 = []
for rates in allrates['swr']:
	tmp3.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=1).values)
tmp3 = np.vstack(tmp3)

tmp2 = []
for rates in allrates['rnd']:
	tmp2.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=1).values)
tmp2 = np.vstack(tmp2)


####################################################################################################################
# ISOMAP SWR
####################################################################################################################
n = len(tmp3)
print(tmp3.shape)
print(tmp2.shape)
tmp = np.vstack((tmp3, tmp2))

imap = Isomap(n_neighbors = 400, n_components = 2, n_jobs = -1).fit_transform(tmp)

iswr = imap[0:n]
irnd = imap[n:]


iswr = iswr.reshape(len(rip_tsd),len(times),2)
irnd = irnd.reshape(len(rnd_tsd),len(times),2)


distance = pd.DataFrame(index = times, columns = ['swr', 'rnd'])
for c, dat in zip(['swr', 'rnd'], [iswr, irnd]):
	for i in range(len(times)):
		x = np.atleast_2d(dat[:,i,0]).T - dat[:,i,0]
		y = np.atleast_2d(dat[:,i,1]).T - dat[:,i,1]
		d = np.sqrt(x**2 + y**2)
		d = d[np.triu_indices_from(d,1)]	
		distance.loc[times[i],c] = np.mean(d)





colors = np.hstack((np.linspace(0, 1, int(len(times)/2)), np.ones(1), np.linspace(0, 1, int(len(times)/2))[::-1]))

figure()
scatter(irnd[:,:,0].flatten(), irnd[:,:,1].flatten(), alpha = 0.5, linewidth = 0, color = 'grey')
for n in range(len(iswr)):
	scatter(iswr[n,:,0], iswr[n,:,1], c = colors, alpha = 0.5, linewidth = 0)
legend()

figure()
plot(distance)

show()

