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
start_init = time()
data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = dict(zip(['wake', 'rem', 'sws'],[{} for _ in range(3)]))

# clients = ipyparallel.Client()	
# dview = clients.direct_view()
dview = Pool(8)

bin_size = 5.0
bins = np.arange(2, 1000.0, bin_size) # 5 ms bin
x = bins[0:-1] + bin_size/2.0

allisi = {ep:pd.DataFrame(index = x) for ep in ['wake', 'rem', 'sws']}

for session in datasets:	
		start = time()
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
		for ep,k in zip([wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws']):					
			spikes_ep 		= {n:spikes[n].restrict(ep) for n in spikes.keys() if len(spikes[n].restrict(ep))}
			for n in spikes_ep.keys():
				neuron = session.split("/")[1]+"_"+str(n)
				spike_times = spikes_ep[n].as_units('ms').index.values
				dif = np.diff(spike_times)
				dif = dif[np.logical_and(dif >= 2.0, dif < 1000.0)]												
				a, b = np.histogram(dif, bins, density = True)
				# firing rate
				fr = (len(spike_times)/float(ep.tot_length('s')))
				tmp = np.exp(-fr*(x/1000.0))
				tmp = tmp/tmp.sum()
				isi = a*bin_size - tmp
				allisi[k][neuron] = isi
				
		print(session, time() - start)
	
	

allisi['wake'].to_hdf('/mnt/DataGuillaume/MergedData/ISI_ALL.h5', key = 'wake', mode = 'w')
allisi['rem'].to_hdf('/mnt/DataGuillaume/MergedData/ISI_ALL.h5', key = 'rem', mode = 'a')
allisi['sws'].to_hdf('/mnt/DataGuillaume/MergedData/ISI_ALL.h5', key = 'sws', mode = 'a')


sys.exit()

firing_rate.to_hdf('/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5', key='firing_rate', mode='w')


import _pickle as cPickle
cPickle.dump(datatosave, open('/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.pickle', 'wb'))


print(time() - start_init)

