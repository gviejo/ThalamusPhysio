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
from multiprocessing import Pool
import os
import neuroseries as nts
from time import time
from pylab import *
from numba import jit



@jit(nopython=True)
def quickBin(spikelist, ts, bins, index):
	rates = np.zeros((len(ts), len(bins)-1, len(index)))
	for i, t in enumerate(ts):
		tbins = t + bins
		for j in range(len(spikelist)):					
			a, _ = np.histogram(spikelist[j], tbins)
			rates[i,:,j] = a
	return rates


data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

datatosave = {'hd-real':{},
				'hd-rnd':{},
				'nohd-real':{},
				'nohd-rnd':{}}



bin_size = 10 # ms
bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2
times = bins[0:-1] + np.diff(bins)/2
n_ex = 1000


for session in datasets:
	print(session)
	generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
	shankStructure 	= loadShankStructure(generalinfo)
	if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
	else:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
	spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
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
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])

	####################################################################################################################
	# binning data
	####################################################################################################################	
	spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
	spikesnohd 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}
	hdneurons		= np.sort(list(spikeshd.keys()))
	nohdneurons		= np.sort(list(spikesnohd.keys()))
				
	####################################################################################################################
	# HD NEURONS
	####################################################################################################################	
	if len(spikeshd) >=5:
		ts = rip_tsd.as_units('ms').index.values
		rates = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)

		datatosave['hd-real'][session] = {	'num_steps': len(times), 
											'dt': np.array(bin_size*1e-3),
											'all_data': rates.astype(np.int32),
											'data_dim': len(hdneurons),
											'train_percentage':np.array(1),
											'train_data':None,
											'train_truth':None,
											'train_ext_input':None,
											'valid_data':None,
											'valid_truth':None,
											'valid_ext_input':None,
											'valid_train':None
											}

		# random		
		rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, n_ex//len(sws_ep)) for i in sws_ep.index])))
		ts = rnd_tsd.as_units('ms').index.values
		rates2 = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)			
		
		datatosave['hd-rnd'][session] = {	'num_steps': len(times), 
											'dt': np.array(bin_size*1e-3),
											'all_data': rates2.astype(np.int32),
											'data_dim': len(hdneurons),
											'train_percentage':np.array(1),
											'train_data':None,
											'train_truth':None,
											'train_ext_input':None,
											'valid_data':None,
											'valid_truth':None,
											'valid_ext_input':None,
											'valid_train':None											
											}

	####################################################################################################################
	# NO HD NEURONS
	####################################################################################################################	
	if len(spikesnohd) >=5:
		ts = rip_tsd.as_units('ms').index.values
		rates = quickBin([spikesnohd[j].as_units('ms').index.values for j in nohdneurons], ts, bins, nohdneurons)	

		datatosave['nohd-real'][session] = {'num_steps': len(times), 
											'dt': np.array(bin_size*1e-3),
											'all_data': rates.astype(np.int32),
											'data_dim': len(nohdneurons),
											'train_percentage':np.array(1),
											'train_data':None,
											'train_truth':None,
											'train_ext_input':None,
											'valid_data':None,
											'valid_truth':None,
											'valid_ext_input':None,
											'valid_train':None											
											}

		# random		
		rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, n_ex//len(sws_ep)) for i in sws_ep.index])))
		ts = rnd_tsd.as_units('ms').index.values
		rates2 = quickBin([spikesnohd[j].as_units('ms').index.values for j in nohdneurons], ts, bins, nohdneurons)			

		datatosave['nohd-rnd'][session] = {'num_steps': len(times), 
											'dt': np.array(bin_size*1e-3),
											'all_data': rates2.astype(np.int32),
											'data_dim': len(nohdneurons),
											'train_percentage':np.array(1),
											'train_data':None,
											'train_truth':None,
											'train_ext_input':None,
											'valid_data':None,
											'valid_truth':None,
											'valid_ext_input':None,
											'valid_train':None											
											}

import _pickle as pickle
pickle.dump(datatosave, open("/home/guillaume/SWR_factors/data/swr_spike_count_all.pickle", "wb"))