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


datatosave = {ep:pd.DataFrame() for ep in ['wak', 'rem', 'sws']}

# clients = ipyparallel.Client()	
# dview = clients.direct_view()
dview = Pool(8)

# firing_rate = pd.DataFrame(columns = ['wake', 'rem', 'sws'])

sys.exit()
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
		Hcorr_ep 		= {}
		for ep,k in zip([wake_ep, rem_ep, sws_ep], ['wak', 'rem', 'sws']):					
			spikes_ep 		= {n:spikes[n].restrict(ep) for n in spikes.keys() if len(spikes[n].restrict(ep))}
			spikes_list 	= [spikes_ep[i].as_units('ms').index.values for i in spikes_ep.keys()]
			Hcorr = dview.map_async(autocorr, spikes_list).get()			
			# normalizing by nomber of spikes			
			for n,i in zip(spikes_ep.keys(), range(len(spikes_ep.keys()))):				
				datatosave[k][session.split("/")[1]+"_"+str(n)] = Hcorr[i]/(len(spikes_list[i])/float(ep.tot_length('s')))				

				# neuron = session.split("/")[1]+"_"+str(n)
				# firing_rate.loc[neuron, k] = (len(spikes_list[i])/float(ep.tot_length('s')))
			
		print(session, time() - start)
	
	


# firing_rate.to_hdf('/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5', key='firing_rate', mode='w')

store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG_SMALLBINS.h5", 'w')
store_autocorr.put('wak', datatosave['wak'])
store_autocorr.put('rem', datatosave['rem'])
store_autocorr.put('sws', datatosave['sws'])
store_autocorr.close()


print(time() - start_init)

