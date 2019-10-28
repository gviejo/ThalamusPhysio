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

session = 'Mouse12/Mouse12-120815'

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
# spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
# position 		= pd.read_csv(data_directory+session+"/"+session.split("/")[1] + ".csv", delimiter = ',', header = None, index_col = [0])
# angle 			= nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')
# tcurves 		= computeAngularTuningCurves(spikeshd, angle, wake_ep, nb_bins = 60, frequency = 1/0.0256)
# neurons 		= tcurves.idxmax().sort_values().index.values

####################################################################################################################
# POSITION X Y 
####################################################################################################################
position 		= pd.read_csv(data_directory+session+"/"+session.split("/")[1] + "_XY.csv", delimiter = ',', header = None, index_col = [0])
position		= nts.TsdFrame(t = position.index.values, d = position.values, time_units = 's')


spikesnohd 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}

placefield, extent = computePlaceFields(spikesnohd, position, wake_ep, nb_bins = 40, frequency = 1/0.0256)

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
# PHASE POSITION
####################################################################################################################
nb_bins = 15
frequency = 1/0.0256
phase_fields = {}
position_tsd = position.restrict(theta_ep)
xpos = position_tsd.iloc[:,0]
ypos = position_tsd.iloc[:,1]
xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)	
for n in spikesnohd:
	position_spike = position_tsd.realign(spikes[n].restrict(theta_ep))
	yindex = np.digitize(position_spike.iloc[:,1].values, ybins)
	xindex = np.digitize(position_spike.iloc[:,0].values, xbins)
	index = np.vstack((yindex, xindex)).T
	tmp = np.zeros((len(ybins)-1, len(xbins)-1))
	for i in range(tmp.shape[0]):
		for j in range(tmp.shape[1]):
			spk = spikes_phase[n].values[np.logical_and(yindex == i, xindex == j)]
			if len(spk):
				tmp[i,j] = circmean(spk)#, low = -np.pi, high = np.pi)
	
	occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])

	phase_fields[n] = tmp/occupancy

	# sys.exit()
	# spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
	# occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
	# mean_spike_count = spike_count/(occupancy+1)
	# place_field = mean_spike_count*frequency    
	# place_fields[n] = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
	
extent = (xbins[0], xbins[-1], ybins[0], ybins[-1]) # USEFUL FOR MATPLOTLIB




####################################################################################################################
# PLOT
####################################################################################################################
figure()
for i, n in enumerate(placefield.keys()):
	subplot(5,9,i+1)
	tmp = placefield[n].values
	tmp = gaussian_filter(tmp, 1)
	imshow(tmp, extent = extent, cmap = 'jet')


figure()
for i, n in enumerate(phase_fields.keys()):
	subplot(5,9,i+1)
	tmp = phase_fields[n]
	# tmp = gaussian_filter(tmp, 2)
	imshow(tmp, extent = extent, cmap = 'hsv', interpolation =  'bilinear', vmin = 0, vmax = 2*np.pi)#, vmin = 0, vmax = 2*np.pi)
	colorbar()
show()