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
import _pickle as cPickle

####################################################################################################################
# FUNCTIONS
####################################################################################################################


dview = Pool(1)

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')



for session in datasets:
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]	
	if np.sum(hd_info == 1)>10:		
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
		
		spikes 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
		neurons 		= np.sort(list(spikes.keys()))

		print(session, len(neurons))
	
		bin_size = 10
		# left_bound = np.arange(-1000-bin_size/2, 1000 - bin_size/4,bin_size/4) # 75% overlap
		# left_bound = np.arange(-1000-bin_size/2, 1000 - bin_size/2, bin_size/2) # 50% overlap
		left_bound = np.arange(-1000-bin_size+3*bin_size/4, 1000 - 3*bin_size/4,3*bin_size/4) # 25% overlap
		obins = np.vstack((left_bound, left_bound+bin_size)).T
		times = obins[:,0]+(np.diff(obins)/2).flatten()

		# cutting times between -500 to 500
		times = times[np.logical_and(times>=-500, times<=500)]

		# datatosave = {'times':times, 'swr':{}, 'rnd':{}, 'bin_size':bin_size}
		datatosave = {'times':times, 'imaps':{}, 'bin_size':bin_size}

		n_ex = 5
		n_rip = len(rip_tsd)
		n_loop = n_rip//n_ex
		idx = np.random.randint(0, n_loop, n_rip)		

		####################################################################################################################
		# WAKE
		####################################################################################################################						
		bin_size_wake = 400
		wake_ep = wake_ep.intersect(nts.IntervalSet(start=wake_ep.loc[0,'start'], end = wake_ep.loc[0,'start']+15*60*1e6))
		bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size_wake, bin_size_wake)
		spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
		for i in neurons:
			spks = spikes[i].as_units('ms').index.values
			spike_counts[i], _ = np.histogram(spks, bins)
		rates_wak = np.sqrt(spike_counts/(bin_size_wake))		

		args = []
		# for i in range(n_loop):
		for i in range(10):
			args.append([spikes, rip_tsd, idx, obins, neurons, bin_size, sws_ep, n_ex, times, rates_wak, i])
		print(n_loop)
		result = dview.starmap_async(compute_isomap, args).get()
		
		
		for i in range(len(result)):		
			datatosave['imaps'][i] = {'swr':result[i][0], 'rnd':result[i][1], 'wak':result[i][2]}
			
			
		####################################################################################################################
		# SAVING
		####################################################################################################################

		cPickle.dump(datatosave, open('../figures/figures_articles_v4/figure1/hd_isomap_'+str(bin_size)+'ms_mixed_swr_rnd_wake/'+session.split("/")[1]+'.pickle', 'wb'))

			