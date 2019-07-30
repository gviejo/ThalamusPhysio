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

def computeAccelTuningCurves(spikes, accel, ep, bin_size, nb_bins = 20):
	accel 		= accel.restrict(ep)
	bins 		= np.linspace(accel.min(), accel.max(), nb_bins)
	idx 		= bins[0:-1]+np.diff(bins)/2
	accel_curves = pd.DataFrame(index = idx,columns = np.arange(len(spikes)))
	for k in spikes:
		spks 	= spikes[k]
		spks 	= spks.restrict(ep)
		accel_spike = accel.realign(spks)
		spike_count, bin_edges = np.histogram(accel_spike, bins)
		occupancy, _ = np.histogram(accel, bins)
		spike_count = spike_count/(occupancy+1)
		accel_curves[k] = spike_count/bin_size

	return accel_curves

def computeSpeedTuningCurves(spikes, speed, ep, bin_size, nb_bins = 10, max_speed = 100):
	speed 		= speed.restrict(ep)
	bins 		= np.linspace(0, max_speed, nb_bins)
	idx 		= bins[0:-1]+np.diff(bins)/2
	speed_curves = pd.DataFrame(index = idx,columns = np.arange(len(spikes)))
	for k in spikes:
		spks 	= spikes[k]
		spks 	= spks.restrict(ep)
		speed_spike = speed.realign(spks)
		spike_count, bin_edges = np.histogram(speed_spike, bins)
		occupancy, _ = np.histogram(speed, bins)
		spike_count = spike_count/(occupancy+1)
		speed_curves[k] = spike_count/bin_size

	return speed_curves




data_directory = '/mnt/DataGuillaume/MergedData/'

datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')


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

	neurons_name 	= [session.split("/")[1]+"_"+str(n) for n in spikes.keys()]
	sys.exit()
	tmp 			= scipy.io.loadmat(data_directory+session+'/Analysis/linspeed.mat')['speed']
	lin_speed 		= nts.Tsd(t = tmp[:,0], d = tmp[:,1], time_units = 's')

	bin_size 		= 1/39

	# downsample speed
	bin_size 	= 0.1
	time_bins 	= np.arange(lin_speed.index[0], lin_speed.index[-1]+bin_size*1e6, bin_size*1e6)
	index 		= np.digitize(lin_speed.index.values, time_bins)
	tmp 		= lin_speed.groupby(index).mean()
	tmp.index 	= time_bins[np.unique(index)-1]+(bin_size*1e6)/2

	accel 		= nts.Tsd(t = lin_speed.index.values[0:-1]+ bin_size/2, d = np.diff(lin_speed.values))

	accel_curves 	= computeAccelTuningCurves(spikes, accel, wake_ep, bin_size)
	accel_curves.columns = neurons_name

	speed_curves 	= computeSpeedTuningCurves(spikes, lin_speed, wake_ep, bin_size)
	speed_curves.columns = neurons_name

	speed_curves_shifted = {}
	for t in np.linspace(-2.0, 2.0, 20):
		tmp = lin_speed.copy()
		tmp.index += int(t*1e6)
		speed_curves_shifted[t] = computeSpeedTuningCurves(spikes, tmp , wake_ep, bin_size)
		speed_curves_shifted[t].columns = neurons_name

	accel_curves_shifted = {}
	for t in np.linspace(-2.0, 2.0, 20):
		tmp = accel.copy()
		tmp.index += int(t*1e6)
		accel_curves_shifted[t] = computeAccelTuningCurves(spikes, tmp , wake_ep, bin_size)
		accel_curves_shifted[t].columns = neurons_name


	store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")
	auto_wak	= store_autocorr['wake'][neurons_name]
	auto_rem 	= store_autocorr['rem'][neurons_name]
	auto_sws	= store_autocorr['sws'][neurons_name]
	store_autocorr.close()
	auto_wak.loc[0] = 0.0
	auto_rem.loc[0] = 0.0
	auto_sws.loc[0] = 0.0
	auto_wak	= auto_wak[0:100]
	auto_rem	= auto_rem[0:100]
	auto_sws	= auto_sws[0:100]
	auto_wak	= auto_wak.rolling(window=40, win_type='gaussian', center=True, min_periods=1).mean(std=5.0)
	auto_rem	= auto_rem.rolling(window=40, win_type='gaussian', center=True, min_periods=1).mean(std=5.0)
	auto_sws	= auto_sws.rolling(window=40, win_type='gaussian', center=True, min_periods=1).mean(std=5.0)
	auto_wak	= auto_wak[1:50]
	auto_rem	= auto_rem[1:50]
	auto_sws	= auto_sws[1:50]


	cells = set([7, 32, 33, 34, 35, 38, 39])
	rest = set(np.arange(len(spikes))) - cells
	order = list(cells) + list(rest)

	set1 = [session.split("/")[1]+"_"+str(n) for n in cells]
	set2 = [session.split("/")[1]+"_"+str(n) for n in rest]




from pylab import *



figure()
for i ,k in enumerate(speed_curves.columns[order]):
	subplot(5,8,i+1)
	if order[i] in cells:
		plot(speed_curves[k], color = 'red')
	else:
		plot(speed_curves[k], color = 'grey')
	
figure()
subplot(131)
plot(auto_wak[set1], alpha = 0.8, color = 'red')
plot(auto_wak[set2], alpha = 0.8, color = 'grey')
ylim(0, 5)
subplot(132)
plot(auto_rem[set1], alpha = 0.8, color = 'red')
plot(auto_rem[set2], alpha = 0.8, color = 'grey')
subplot(133)
plot(auto_sws[set1], alpha = 0.8, color = 'red')
plot(auto_sws[set2], alpha = 0.8, color = 'grey')


figure()
for i ,k in enumerate(accel_curves.columns[order]):
	subplot(5,8,i+1)	
	plot(accel_curves[k])


from matplotlib import cm
times = np.sort(list(speed_curves_shifted.keys()))
norm = matplotlib.colors.Normalize(vmin=times[0], vmax=times[-1])
rgba_color = cm.jet(norm(400),bytes=True) 


figure()
for i, n in enumerate(order):
	subplot(5,8,i+1)
	name = session.split("/")[1]+"_"+str(n)
	plot(speed_curves[name], color = 'black')
	for t in np.sort(list(speed_curves_shifted.keys())):
		plot(speed_curves_shifted[t][name], color = cm.jet(norm(t)), linewidth = 0.8, alpha = 0.7)
		

# i = 33
# n = order[i]
# name = session.split("/")[1]+"_"+str(n)
# for i, t in enumerate(np.sort(list(speed_curves_shifted.keys()))):
# 	subplot(4,5,i+1)
# 	plot(speed_curves[name], color = 'black')
# 	plot(speed_curves_shifted[t][name], label = str(t))
# 	title(t)



figure()
for i, n in enumerate(order):
	subplot(5,8,i+1)
	name = session.split("/")[1]+"_"+str(n)
	plot(accel_curves[name], color = 'black')
	for t in np.sort(list(accel_curves_shifted.keys())):
		plot(accel_curves_shifted[t][name])

show()



