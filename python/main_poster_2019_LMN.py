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


session = 'Mouse32/Mouse32-140822'

ad_ahv = []
ad_tcurves = []

for session in datasets:
# for session in ['Mouse32/Mouse32-140822']:
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



		spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
		neurons 		= np.sort(list(spikeshd.keys()))


		####################################################################################################################
		# HEAD DIRECTION INFO
		####################################################################################################################
		spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
		neurons 		= np.sort(list(spikeshd.keys()))
		# sys.exit()
		position 		= pd.read_csv(data_directory+session+"/"+session.split("/")[1] + ".csv", delimiter = ',', header = None, index_col = [0])
		angle 			= nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')
		tcurves 		= computeAngularTuningCurves(spikeshd, angle, wake_ep, nb_bins = 61, frequency = 1/0.0256)
		neurons 		= tcurves.idxmax().sort_values().index.values
		frate 			= pd.Series(index = tcurves.keys(), data = [len(spikes[k].restrict(wake_ep))/wake_ep.tot_length('s') for k in tcurves.keys()])
		# tcurves 		= tcurves/frate

		####################################################################################################################
		# ANGULAR VELOCITY
		####################################################################################################################
		nb_bins 		= 61
		bin_size		= np.mean(np.diff(angle.as_units('s').index.values))
		tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
		tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
		tmp2 			= nts.Tsd(tmp2)
		tmp4			= np.diff(tmp2.values)/np.diff(tmp2.as_units('s').index.values)	
		velocity 		= nts.Tsd(t=tmp2.index.values[1:], d = tmp4)
		velocity 		= velocity.restrict(wake_ep)
		bins 			= np.linspace(-np.pi, np.pi, nb_bins)
		idx 			= bins[0:-1]+np.diff(bins)/2
		velo_curves		= pd.DataFrame(index = idx, columns = neurons)

		for k in neurons:
			spks 		= spikes[k].restrict(wake_ep)	
			speed_spike = velocity.realign(spks)
			spike_count, bin_edges = np.histogram(speed_spike, bins)
			occupancy, _ = np.histogram(velocity.restrict(wake_ep), bins)
			spike_count = spike_count/(occupancy+1)
			velo_curves[k] = spike_count/bin_size
			# normalizing by firing rate 
			# velo_curves[k] = velo_curves[k]/(len(spikes[k].restrict(wake_ep))/wake_ep.tot_length('s'))

		tcurves.columns = pd.Index([session.split('/')[1]+'_'+str(n) for n in tcurves.columns])

		velo_curves.columns = pd.Index([session.split('/')[1]+'_'+str(n) for n in velo_curves.columns])

		ad_tcurves.append(tcurves)
		ad_ahv.append(velo_curves)
		



ad_tcurves = pd.concat(ad_tcurves, 1)
ad_ahv = pd.concat(ad_ahv, 1)

####################################################################################################################
# SAVING DATA
####################################################################################################################
datatosave = {
	"tcurves"	: ad_tcurves,
	"ahvcurves"	: ad_ahv
	}

import _pickle as cPickle
cPickle.dump(datatosave, open('../../LMNphysio/figures/figures_poster_2019/Data_AD.pickle', 'wb'))



