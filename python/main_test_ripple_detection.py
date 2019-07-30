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
from pylab import *


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

###############################################################################################
# LOADING LFP
###############################################################################################
path = data_directory+session+'/'+session.split('/')[1]+'.eeg'
lfp = loadLFP(path, n_channels=69, channel=64, frequency=1250.0, precision='int16')

frequency = 1250.0
low_cut = 100
high_cut = 300
windowLength = 51
low_thresFactor = 5
high_thresFactor = 5
minRipLen = 20 # ms
maxRipLen = 200 # ms
minInterRippleInterval = 20 # ms
limit_peak = 20


signal = butter_bandpass_filter(lfp.values, low_cut, high_cut, frequency, order = 4)

# sys.exit()

squared_signal = np.square(signal)

window = np.ones(windowLength)/windowLength

nSS = scipy.signal.filtfilt(window, 1, squared_signal)

# Removing point above 100000
nSS = pd.Series(index = lfp.index.values, data = nSS)
nSS = nSS[nSS<60000]

nSS = (nSS - np.mean(nSS))/np.std(nSS)

signal = pd.Series(index = lfp.index.values, data = signal)



figure()
ax = subplot(211)
plot(nSS)
subplot(212,sharex = ax)
plot(lfp)
show()


######################################################l##################################
# Round1 : Detecting Ripple Periods by thresholding normalized signal
thresholded = np.where(nSS > low_thresFactor, 1,0)
start = np.where(np.diff(thresholded) > 0)[0]
stop = np.where(np.diff(thresholded) < 0)[0]
if len(stop) == len(start)-1:
	start = start[0:]
if len(stop)-1 == len(start):
	stop = stop[1:]



################################################################################################
# Round 2 : Excluding ripples whose length < minRipLen and greater than Maximum Ripple Length
if len(start):
	l = (nSS.index.values[stop] - nSS.index.values[start])/1000 # from us to ms
	idx = np.logical_and(l > minRipLen, l < maxRipLen)
else:	
	print("Detection by threshold failed!")
	sys.exit()

rip_ep = nts.IntervalSet(start = nSS.index.values[start[idx]], end = nSS.index.values[stop[idx]])

####################################################################################################################
# Round 3 : Merging ripples if inter-ripple period is too short
rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval/1000, time_units = 's')



#####################################################################################################################
# Round 4: Discard Ripples with a peak power < high_thresFactor and > limit_peak
rip_max = []
rip_tsd = []
for s, e in rip_ep.values:
	tmp = nSS.loc[s:e]
	rip_tsd.append(tmp.idxmax())
	rip_max.append(tmp.max())

rip_max = np.array(rip_max)
rip_tsd = np.array(rip_tsd)

tokeep = np.logical_and(rip_max > high_thresFactor, rip_max < limit_peak)

rip_ep = rip_ep[tokeep].reset_index(drop=True)
rip_tsd = nts.Tsd(t = rip_tsd[tokeep], d = rip_max[tokeep])


# t1, t2 = (6002729000,6003713000)
t1, t2 = (6002700000,6003713000)
figure()
ax = subplot(211)
plot(lfp.loc[t1:t2])
plot(signal.loc[t1:t2])
plot(lfp.restrict(rip_ep).loc[t1:t2], '.')
subplot(212,sharex = ax)
plot(nSS.loc[t1:t2])
axhline(low_thresFactor)
show()




###########################################################################################################
# Writing for neuroscope

rip_ep			= sws_ep.intersect(rip_ep)	
rip_tsd 		= rip_tsd.restrict(sws_ep)

start = rip_ep.as_units('ms')['start'].values
peaks = rip_tsd.as_units('ms').index.values
ends = rip_ep.as_units('ms')['end'].values

datatowrite = np.vstack((start,peaks,ends)).T.flatten()

n = len(rip_ep)

texttowrite = np.vstack(((np.repeat(np.array(['PyRip start 1']), n)), 
						(np.repeat(np.array(['PyRip peak 1']), n)),
						(np.repeat(np.array(['PyRip stop 1']), n))
							)).T.flatten()

evt_file = data_directory+session+'/'+session.split('/')[1]+'.evt.py.rip'
f = open(evt_file, 'w')
for t, n in zip(datatowrite, texttowrite):
	f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
f.close()		




