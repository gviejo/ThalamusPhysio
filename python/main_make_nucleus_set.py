#!/usr/bin/env python

'''
	File name: main_make_nucleus_set.py
	Author: Guillaume Viejo
	Date created: 28/09/2017    
	Python Version: 3.5.2


to create dict with the neurons belonging to each nuclei according to the mapping

'''

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import sys
from scipy.ndimage import gaussian_filter	
import os

###############################################################################################################
# PARAMETERS
###############################################################################################################

mouses 	= ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']

path 	= '../data/maps/mapping_session_shank_nucleus_paxino.txt'

data 	= loadMappingNucleus(path)
nuclei 	= np.unique(np.hstack([np.unique(data[m].values) for m in data.keys()]))
# need different way of accesing values
# neuron -> nucleus
# nucleus -> pool of neuron

data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

neuron_to_nucleus = dict()
nucleus_to_neuron = dict(zip(nuclei,[[] for _ in range(len(nuclei))]))
neuron_to_shank = dict()
neuron_to_channel = dict()

df_mapping_nucleus = pd.DataFrame(columns = ['session', 'shank', 'channel', 'nucleus', 'hd'])


for m in mouses:
	sessions 		= [s.split("/")[1] for s in datasets if m in s]
	for s in sessions:				
		generalinfo 		= scipy.io.loadmat(data_directory+m+"/"+s+'/Analysis/GeneralInfo.mat')
		shankStructure 		= loadShankStructure(generalinfo)
		spikes,shank		= loadSpikeData(data_directory+m+"/"+s+'/Analysis/SpikeData.mat', shankStructure['thalamus'])				
		hd_info 			= scipy.io.loadmat(data_directory+m+'/'+s+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		hd_info_neuron		= np.array([hd_info[n] for n in spikes.keys()])
		shankIndex 			= np.array([shank[n] for n in spikes.keys()]).flatten()
		channelIndex 		= scipy.io.loadmat(data_directory+m+'/'+s+'/Analysis/SpikeWaveF.mat')['maxIx'][0]-1
		channelIndex		= channelIndex[list(spikes.keys())]

		if np.max(shankIndex) > 8 : sys.exit("Invalid shank index for thalamus" + s)				
		shank_to_neurons 	= {k:np.array(list(spikes.keys()))[shankIndex == k] for k in np.unique(shankIndex)}				
		for k in shank_to_neurons:			
			for n in shank_to_neurons[k]:				
				neuron_to_nucleus[s+'_'+str(n)] = data[m].loc[sessions.index(s), 7-k]
				nucleus_to_neuron[data[m].loc[sessions.index(s), 7-k]].append(s+'_'+str(n))
				neuron_to_shank[s+'_'+str(n)] = (sessions.index(s),k)
				neuron_to_channel[s+'_'+str(n)] = channelIndex[n]
				df_mapping_nucleus.loc[s+'_'+str(n),'session'] = sessions.index(s)
				df_mapping_nucleus.loc[s+'_'+str(n),'shank'] = 7-k
				df_mapping_nucleus.loc[s+'_'+str(n),'channel'] = channelIndex[n]
				df_mapping_nucleus.loc[s+'_'+str(n),'nucleus'] = data[m].loc[sessions.index(s), 7-k]
				df_mapping_nucleus.loc[s+'_'+str(n),'hd'] = hd_info_neuron[n]

mapping_nucleus = {
					'neuron_to_nucleus': neuron_to_nucleus,
					'nucleus_to_neuron': nucleus_to_neuron,
					'neuron_to_shank':neuron_to_shank,
					'neuron_to_channel':neuron_to_channel
					}

cPickle.dump(mapping_nucleus, open("../data/maps/mapping_nucleus_allen.pickle", 'wb'))

df_mapping_nucleus.to_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5", key='mapping', mode='w')
