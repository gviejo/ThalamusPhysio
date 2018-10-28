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
# import ipyparallel
from multiprocessing import Pool
import os, sys
import neuroseries as nts
import time

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {}

# clients = ipyparallel.Client()	
# dview = clients.direct_view()
dview = Pool(8)


for session in datasets:	
	generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
	shankStructure 	= loadShankStructure(generalinfo)
	if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
	else:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
	spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		

	spind_ep_hpc = np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".evt.spd.hpc")[:,0]
	spind_ep_hpc = spind_ep_hpc.reshape(len(spind_ep_hpc)//2,2)
	spind_ep_hpc = nts.IntervalSet(spind_ep_hpc[:,0], spind_ep_hpc[:,1], time_units = 'ms')

	spikes_spd 		= {n:spikes[n].restrict(spind_ep_hpc) for n in spikes.keys() if len(spikes[n].restrict(spind_ep_hpc))}
	
	spikes_list = [spikes_spd[i].as_units('ms').index.values for i in spikes_spd.keys()]
	# for tsd in spikes_list:
	# 	cross_correlation((tsd, tsd))	
	Hcorr = dview.map_async(autocorr, spikes_list).get()
		
	for n,i in zip(spikes_spd.keys(), range(len(spikes_spd.keys()))):
		datatosave[session.split("/")[1]+"_"+str(n)] = Hcorr[i]

	print(session)		
	

import _pickle as cPickle
cPickle.dump(datatosave, open('/mnt/DataGuillaume/MergedData/AUTOCORR_SPD.pickle', 'wb'))




