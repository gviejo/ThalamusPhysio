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

# session = 'Mouse32/Mouse32-140822'
session = 'Mouse17/Mouse17-130130'


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
print(session)
bin_size = 10000 # us
rates = []
for e in sws_ep.index:
	ep = sws_ep.loc[[e]]
	bins = np.arange(ep.iloc[0,0], ep.iloc[0,1], bin_size)
	r = pd.DataFrame(index = (bins[0:-1] + np.diff(bins)/2).astype('int'), columns = np.sort(list(spikes.keys())))
	for n in spikes.keys():
		r[n] = np.histogram(spikes[n].restrict(ep).index.values, bins)[0]
	rates.append(r)
rates = pd.concat(rates, 0)
a = 6214234000
b = 6218234000

a = 6232840000
b = a+4000*1000

a = 3681837000
b = a + 2000*1000

a = 2362542000
b = a + 2000*1000

ab = nts.IntervalSet(start = a, end = b)

total = rates.sum(1)

total2 = total.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)

idx = total2[total2<np.percentile(total2,20)].index.values

tmp = [[idx[0]]]
for i in range(1,len(idx)):
	if (idx[i] - idx[i-1]) > bin_size:
		tmp.append([idx[i]])
	elif (idx[i] - idx[i-1]) == bin_size:
		tmp[-1].append(idx[i])

down_ep = np.array([[e[0],e[-1]] for e in tmp if len(e) > 1])
down_ep = nts.IntervalSet(start = down_ep[:,0], end = down_ep[:,1])



down_ep = down_ep.drop_short_intervals(bin_size)
down_ep = down_ep.merge_close_intervals(bin_size*2)
down_ep = down_ep.drop_short_intervals(30000)
down_ep = down_ep.drop_long_intervals(500000)

down_ep = down_ep.reset_index(drop=True)


up_ep 	= nts.IntervalSet(down_ep['end'][0:-1], down_ep['start'][1:])
up_ep = sws_ep.intersect(up_ep)




###########################################################################################################
# Writing for neuroscope

start = down_ep.as_units('ms')['start'].values
ends = down_ep.as_units('ms')['end'].values

datatowrite = np.vstack((start,ends)).T.flatten()

n = len(down_ep)

texttowrite = np.vstack(((np.repeat(np.array(['PyDown start 1']), n)), 
						(np.repeat(np.array(['PyDown stop 1']), n))
							)).T.flatten()

evt_file = data_directory+session+'/'+session.split('/')[1]+'.evt.py.dow'
f = open(evt_file, 'w')
for t, n in zip(datatowrite, texttowrite):
	f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
f.close()		


start = up_ep.as_units('ms')['start'].values
ends = up_ep.as_units('ms')['end'].values

datatowrite = np.vstack((start,ends)).T.flatten()

n = len(up_ep)

texttowrite = np.vstack(((np.repeat(np.array(['PyUp start 1']), n)), 
						(np.repeat(np.array(['PyUp stop 1']), n))
							)).T.flatten()

evt_file = data_directory+session+'/'+session.split('/')[1]+'.evt.py.upp'
f = open(evt_file, 'w')
for t, n in zip(datatowrite, texttowrite):
	f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
f.close()		
