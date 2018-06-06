#!/usr/bin/env python
'''
    File name: main_ripp_mod.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2

Sharp-waves ripples modulation 
Used to make figure 1

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
dview = Pool(24)


for session in datasets:
	# if 'Mouse12' in session:
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
		for ep,k in zip([wake_ep, rem_ep, sws_ep], ['wake', 'rem', 'sws']):					
			spikes_ep 		= {n:spikes[n].restrict(ep) for n in spikes.keys() if len(spikes[n].restrict(ep))}
			spikes_list 	= [spikes_ep[i].as_units('ms').index.values for i in spikes_ep.keys()]
			Hcorr = dview.map_async(autocorr, spikes_list).get()

			# normalizing by nomber of spikes
			
			for n,i in zip(spikes_ep.keys(), range(len(spikes_ep.keys()))):
				datatosave[k][session.split("/")[1]+"_"+str(n)] = Hcorr[i]/(len(spikes_list[i])/float(ep.tot_length('s')))

			
		print(session, time() - start)
	
	
	

import _pickle as cPickle
cPickle.dump(datatosave, open('/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.pickle', 'wb'))


print(time() - start_init)

