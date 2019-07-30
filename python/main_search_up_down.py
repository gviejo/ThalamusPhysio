#!/usr/bin/env python
'''

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
from Wavelets import MyMorlet as Morlet

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {}
spikes_spindle_phase = {'hpc':{}, 'thl':{}}

session = 'Mouse32/Mouse32-140822'

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


#################################################################################################
#DETECTION UP/DOWN States
#################################################################################################
print("up/down states")

store 			= pd.HDFStore(data_directory+'/phase_spindles/'+session.split("/")[1]+".lfp")
phase_hpc 		= nts.Tsd(store['phase_hpc_spindles'])
phase_thl 		= nts.Tsd(store['phase_thl_spindles'])
lfp_hpc 		= nts.Tsd(store['lfp_hpc'])
lfp_thl			= nts.TsdFrame(store['lfp_thl'])
store.close()		

# bins of 5000 us
bins 			= np.floor(np.arange(lfp_hpc.start_time(), lfp_hpc.end_time()+5000, 5000))
total_value 	= nts.Tsd(bins[0:-1]+(bins[1]-bins[0])/2, np.zeros(len(bins)-1)).restrict(sws_ep)
# each shank
for s in shankStructure['thalamus']:		
	neuron_index = np.where(shank == s)[0]
	if len(neuron_index):
		tmp = {i:spikes[i] for i in neuron_index}
		frate			= getFiringRate(tmp, bins)		
		frate 			= frate.as_series()
		# ISI of 50 ms		
		down_ep 		= nts.IntervalSet(frate[frate==0.0].index.values[0:-1], frate[frate==0.0].index.values[1:])
		down_ep 		= down_ep.drop_long_intervals(5001).merge_close_intervals(0.0).intersect(sws_ep).drop_short_intervals(50000)
		down_ep 		= down_ep.merge_close_intervals(50000)
		down_candidates = nts.Tsd((down_ep['start'] + (down_ep['end'] - down_ep['start'])/2.).values, np.zeros(len(down_ep)))
		# Smooth frate 
		tmp 			= gaussFilt(frate.values, (2,))
		frate 			= nts.Tsd(frate.index.values, tmp).restrict(sws_ep)
		threshold	 	= np.percentile(frate, 20)
		# frate < threshold around down candidates		
		boole			= nts.Tsd((frate.as_series() <= threshold)*1.0)
		# realigner bool sur les candidates
		boole 			= boole.realign(down_candidates)
		down_ep 		= down_ep.iloc[np.where(boole)[0]]		
		# add +1 in total value / pandas is weird # TODO
		for i in total_value.restrict(down_ep).index.values:
			total_value.loc[i] += 1
# at least 3 shank 
down_ep 		= nts.IntervalSet(total_value[total_value>=3.0].index.values[0:-1], total_value[total_value>=3.0].index.values[1:])
down_ep 		= down_ep.drop_long_intervals(5001).merge_close_intervals(0.0)
down_ep 		= down_ep.merge_close_intervals(50000).drop_long_intervals(400000)
up_ep 			= nts.IntervalSet(down_ep['end'][0:-1], down_ep['start'][1:]).intersect(sws_ep)
up_ep 			= up_ep.drop_short_intervals(500000)
