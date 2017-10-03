#!/usr/bin/env python
'''
    File name: main_ripp_mod.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2

Sharp-waves ripples modulation 
Used to make figure 1

'''

import numpy as np
import pandas as pd
import scipy.io
from functions import *
# from pylab import *
import ipyparallel
import os, sys
import neuroseries as nts
import time

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {}

clients = ipyparallel.Client()	
dview = clients.direct_view()

for session in datasets:	
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

	spikes_sws 		= {n:spikes[n].restrict(sws_ep) for n in spikes.keys() if len(spikes[n].restrict(sws_ep))}

	# plotEpoch(wake_ep, sleep_ep, rem_ep, sws_ep, rip_ep, {0:spikes[27]})	
	
	def cross_correlation(tsd):
		spike_tsd, rip_tsd = tsd
		import numpy as np
		from functions import xcrossCorr_fast
		bin_size 	= 5 # ms 
		nb_bins 	= 200
		confInt 	= 0.95
		nb_iter 	= 1000
		jitter  	= 150 # ms					
		H0, Hm, HeI, HeS, Hstd, times = xcrossCorr_fast(rip_tsd, spike_tsd, bin_size, nb_bins, nb_iter, jitter, confInt)		
		
		return (H0 - Hm)/Hstd


	spikes_list = [spikes_sws[i].as_units('ms').index.values for i in spikes_sws.keys()]
	
	Hcorr = dview.map_sync(cross_correlation, zip(spikes_list, [rip_tsd.as_units('ms').index.values for i in spikes_sws.keys()]))
		
	datatosave[session] = {}
	for n,i in zip(spikes_sws.keys(), range(len(spikes_sws.keys()))):
		datatosave[session][session.split("/")[1]+"_"+str(n)] = Hcorr[i]

	print(session)		
	

import _pickle as cPickle
cPickle.dump(datatosave, open('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', 'wb'))




