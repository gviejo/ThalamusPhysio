#!/usr/bin/env python
'''
    File name: main_make_Thetainfo.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2

Theta modulation, returns angle 


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
from scipy.ndimage.filters import gaussian_filter
import _pickle as cPickle
from pycircstat import mean as circmean

def computePlaceFields(spikes, position, ep, nb_bins = 100, frequency = 120.0):
	place_fields = {}
	position_tsd = position.restrict(ep)
	xpos = position_tsd.iloc[:,0]
	ypos = position_tsd.iloc[:,1]
	xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
	ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)	
	for n in spikes:
		position_spike = position_tsd.realign(spikes[n].restrict(ep))
		spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
		occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
		mean_spike_count = spike_count/(occupancy+1)
		place_field = mean_spike_count*frequency    
		place_fields[n] = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
		
	extent = (xbins[0], xbins[-1], ybins[0], ybins[-1]) # USEFUL FOR MATPLOTLIB
	return place_fields, extent


data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

session = 'Mouse12/Mouse12-120809'

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

print(session, len(neurons))

####################################################################################################################
# HEAD DIRECTION INFO
####################################################################################################################
spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
position 		= pd.read_csv(data_directory+session+"/"+session.split("/")[1] + ".csv", delimiter = ',', header = None, index_col = [0])
angle 			= nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')
tcurves 		= computeAngularTuningCurves(spikeshd, angle, wake_ep, nb_bins = 60, frequency = 1/0.0256)
hdneurons 		= tcurves.idxmax().sort_values().index.values

####################################################################################################################
# 
####################################################################################################################
spikesnohd 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}


####################################################################################################################
# PHASE SPIKE NO HD
####################################################################################################################
n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')
lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
lfp_hpc 		= downsample(lfp_hpc.restrict(wake_ep), 1, 5)
lfp_filt_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 5, 15, fs/5, 2))	
power	 		= nts.Tsd(lfp_filt_hpc.index.values, np.abs(lfp_filt_hpc.values))
enveloppe,dummy	= getPeaksandTroughs(power, 5)

index 			= (enveloppe.as_series() > np.percentile(enveloppe, 10)).values*1.0
start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
tmp 			= np.where(end_cand != start_cand)
start_cand 		= enveloppe.index.values[start_cand[tmp]]
end_cand	 	= enveloppe.index.values[end_cand[tmp]]
theta_ep		= nts.IntervalSet(start_cand, end_cand)
theta_ep		= theta_ep.drop_short_intervals(300000)
theta_ep 	 	= theta_ep.merge_close_intervals(30000).drop_short_intervals(1000000)

phase 			= getPhase(lfp_hpc, 5, 15, 16, fs/5.)	

phase 			= phase.restrict(theta_ep)
phase 			= phase.as_series()
tmp  			= phase.values + (2*np.pi)
tmp 			= tmp % (2*np.pi) 
phase 			= nts.Tsd(t = phase.index.values, d = tmp)

spikes_phase	= {n:phase.realign(spikesnohd[n].restrict(theta_ep), align = 'closest') for n in spikesnohd.keys()}

####################################################################################################################
# PHASE ANGLE RELATION
####################################################################################################################
nb_bins 		= 19
bins 			= np.linspace(0, 2*np.pi, nb_bins)

idx 			= bins[0:-1]+np.diff(bins)/2
tuning_curves 	= pd.DataFrame(index = idx, columns = spikes_phase.keys())
angle 			= angle[~angle.index.duplicated(keep='first')]

for k in spikes_phase:
	spk_phase 		= spikes_phase[k].restrict(theta_ep)
	spk_angle 		= angle.restrict(theta_ep).realign(spk_phase)
	idx 			= np.digitize(spk_angle.values, bins)-1
	angle_phase 	= np.zeros(nb_bins-1)
	for i in np.unique(idx):
		# angle_phase[i] 	= circmean(spk_phase[idx==i])
		angle_phase[i] 	= np.mean(np.cos(spk_phase[idx==i].values))
	occupancy, _ 	= np.histogram(angle, bins)
	tuning_curves[k] = angle_phase

figure()
for i, n in enumerate(spikesnohd.keys()):
	subplot(6,8,i+1)
	plot(tuning_curves[n])
	# tmp = tuning_curves[n].sort_values()
	# plot(tmp.values, tmp.index.values)
	# ylim(0, 2*np.pi)
show()


# angle 			= angle.restrict(ep)
# # # Smoothing the angle here
# # tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
# # tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
# # angle			= nts.Tsd(tmp2%(2*np.pi))
# for k in spikes:
# 	spks 			= spikes[k]
# 	true_ep 		= nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), 
# 								end = np.minimum(angle.index[-1], spks.index[-1]))		
# 	spks 			= spks.restrict(true_ep)		
# 	angle_spike 	= angle.restrict(true_ep).realign(spks)
# 	spike_count, bin_edges = np.histogram(angle_spike, bins)
# 	occupancy, _ 	= np.histogram(angle, bins)
# 	spike_count 	= spike_count/occupancy
# 	tuning_curves[k] = spike_count*frequency

# plot(lfp_hpc.restrict(theta_ep))
# plot(lfp_filt_hpc.restrict(theta_ep))
# plot(phase.restrict(theta_ep).as_series()*100)