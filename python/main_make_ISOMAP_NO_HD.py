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

####################################################################################################################
# FUNCTIONS
####################################################################################################################
@jit(nopython=True)
def histo(spk, obins):
	n = len(obins)
	count = np.zeros(n)
	for i in range(n):
		count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
	return count


data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')


for session in datasets:
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]	
	if np.sum(hd_info == 0)>10:		
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
		
		spikes 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}
		neurons 		= np.sort(list(spikes.keys()))

		print(session, len(neurons))
	
		bin_size = 50
		# left_bound = np.arange(-500-bin_size/2, 500 - bin_size/4,bin_size/4) # 75% overlap
		left_bound = np.arange(-1000-bin_size/2, 1000 - bin_size/2, bin_size/2) # 50% overlap
		obins = np.vstack((left_bound, left_bound+bin_size)).T
		times = obins[:,0]+(np.diff(obins)/2).flatten()

		# cutting times between -500 to 500
		times = times[np.logical_and(times>=-500, times<=500)]

		# datatosave = {'times':times, 'swr':{}, 'rnd':{}, 'bin_size':bin_size}
		datatosave = {'times':times, 'imaps':{}, 'bin_size':bin_size}

		n_ex = 50
		n_rip = len(rip_tsd)
		n_loop = n_rip//n_ex
		idx = np.random.randint(0, n_loop, n_rip)

		# for i in range(n_loop):
		for i in range(10):
			print(i, '/', n_loop)
			####################################################################################################################
			# SWR
			####################################################################################################################						
			# BINNING
			tmp = rip_tsd.index.values[idx == i]
			subrip_tsd = pd.Series(index = tmp, data = np.nan)			
			rates_swr = []
			tmp2 = subrip_tsd.index.values/1e3
			for j, t in enumerate(tmp2):				
				tbins = t + obins
				spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)
				for k in neurons:
					spks = spikes[k].as_units('ms').index.values
					spike_counts[k] = histo(spks, tbins)
					
				rates_swr.append(np.sqrt(spike_counts/(bin_size)))

			####################################################################################################################
			# RANDOM
			####################################################################################################################			
			# BINNING
			rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[j,'start']+500000, sws_ep.loc[j,'end']+500000, np.maximum(1,n_ex//len(sws_ep))) for j in sws_ep.index])))
			if len(rnd_tsd) > n_ex:
				rnd_tsd = rnd_tsd[0:n_ex]
			rates_rnd = []
			tmp3 = rnd_tsd.index.values/1000
			for j, t in enumerate(tmp3):				
				tbins = t + obins
				spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)	
				for k in neurons:
					spks = spikes[k].as_units('ms').index.values	
					spike_counts[k] = histo(spks, tbins)
			
				rates_rnd.append(np.sqrt(spike_counts/(bin_size)))


			# SMOOTHING
			tmp3 = []
			for rates in rates_swr:
				tmp3.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=1).loc[-500:500].values)
			tmp3 = np.vstack(tmp3)
			tmp3 = tmp3.astype(np.float32)
			#SMOOTHING
			tmp2 = []
			for rates in rates_rnd:				
				tmp2.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=1).loc[-500:500].values)
			tmp2 = np.vstack(tmp2)
			tmp2 = tmp2.astype(np.float32)

			n = len(tmp3)
			tmp = np.vstack((tmp3, tmp2))

			# ISOMAP			
			imap = Isomap(n_neighbors = 100, n_components = 2).fit_transform(tmp)



			iswr = imap[0:n].reshape(len(subrip_tsd),len(times),2)
			irnd = imap[n:].reshape(len(rnd_tsd),len(times),2)

			datatosave['imaps'][i] = {'swr':iswr, 'rnd':irnd}

			

		# n_loop = 10
		# n_ex = 100
		# for i in range(n_loop):
			
		# 	# ISOMAP
		# 	tmp2 = tmp2.astype(np.float32)
		# 	imap = Isomap(n_neighbors = 100, n_components = 2).fit_transform(tmp2)			
			
		# 	datatosave['rnd'][i] = imap.reshape(len(rnd_tsd),len(times),2)
			
		####################################################################################################################
		# SAVING
		####################################################################################################################

		cPickle.dump(datatosave, open('../figures/figures_articles_v4/figure1/nohd_isomap_50ms_mixed_swr_rnd/'+session.split("/")[1]+'.pickle', 'wb'))

			