#!/usr/bin/env python
'''
	File name: main_spindle_feedback.py
	Author: Guillaume Viejo
	Date created: 20/09/2017    
	Python Version: 3.5.2

to search for feedback of hippocampal spindles to thalamic spindles

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

# clients = ipyparallel.Client()
# print(clients.ids)
# dview = clients.direct_view()

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {}

allkappa = []
meankappa = []

session_rip_in_thl_spind_mod = []
session_rip_in_thl_spind_phase = {}
session_rip_in_hpc_spind_mod = []
session_rip_in_hpc_spind_phase = {}
phase = {}

for session in datasets:
	print("session"+session)	
	generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
	shankStructure 	= loadShankStructure(generalinfo)	
	if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
	else:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
	spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
	sws_ep 			= loadEpoch(data_directory+session, 'sws')
	spikes 			= {n:spikes[n] for n in spikes.keys() if len(spikes[n].restrict(sws_ep))}
	n_neuron 		= len(spikes)
	n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')
	lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
	lfp_hpc 		= downsample(lfp_hpc, 1, 5)
	thl_channels 	= list(np.sort([shank_to_channel[k][0] for k in shankStructure['thalamus']]))		
	lfp_thl 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, thl_channels, float(fs), 'int16')	
	lfp_thl 		= downsample(lfp_thl, 1, 5)
##################################################################################################
# LOAD THALAMIC SPINDLES
##################################################################################################		
	tmp 			= np.genfromtxt("/mnt/DataGuillaume/MergedData/"+session+"/"+session.split("/")[1]+".evt.spd.thl")[:,0]
	tmp 			= tmp.reshape(len(tmp)//2,2)
	spind_thl_ep 	= nts.IntervalSet(tmp[:,0], tmp[:,1], time_units = 'ms')
##################################################################################################
# LOAD HIPP SPINDLES
##################################################################################################		
	tmp 			= np.genfromtxt("/mnt/DataGuillaume/MergedData/"+session+"/"+session.split("/")[1]+".evt.spd.hpc")[:,0]
	tmp 			= tmp.reshape(len(tmp)//2,2)
	spind_hpc_ep 	= nts.IntervalSet(tmp[:,0], tmp[:,1], time_units = 'ms')	
##################################################################################################
# PHASE KAPPA
##################################################################################################		
	spind_ep 		= spind_hpc_ep.intersect(spind_thl_ep).drop_short_intervals(0.0)
	phase_hpc 		= getPhase(lfp_hpc, 8, 18, 16, fs/5., power = False)
	phase_hpc		= phase_hpc.restrict(sws_ep)
	phase_thl 		= getPhase(lfp_thl, 8, 18, 16, fs/5., power = False)
	phase_thl		= phase_thl.restrict(sws_ep)

	spind_mod1 		= computePhaseModulation(phase_hpc, spikes, spind_hpc_ep)
	spind_mod2 		= computePhaseModulation(phase_thl[0], spikes, spind_thl_ep)
	spind_mod3 		= computePhaseModulation(phase_hpc, spikes, spind_ep)
	spind_mod4 		= computePhaseModulation(phase_thl[0], spikes, spind_ep)
	
	kappa 			= np.vstack([spind_mod1[:,2], spind_mod3[:,2], spind_mod2[:,2], spind_mod4[:,2]]).transpose()

	kappa[np.isnan(kappa)] = 0.0

	allkappa.append(kappa)
	meankappa.append(kappa.mean(0))

##################################################################################################
# Phase of RIPPLES in SPINDLES
##################################################################################################		
	rip_ep,rip_tsd 	= loadRipples(data_directory+session)
	rip_ep			= sws_ep.intersect(rip_ep)	
	rip_tsd 		= rip_tsd.restrict(sws_ep)		

	rip_in_thl_spind_mod, rip_in_thl_spind_phase = computePhaseModulation(phase_thl, {0:rip_tsd}, spind_thl_ep, True)
	rip_in_hpc_spind_mod, rip_in_hpc_spind_phase = computePhaseModulation(phase_hpc, {0:rip_tsd}, spind_hpc_ep, True)

	session_rip_in_thl_spind_mod.append(rip_in_thl_spind_mod)
	session_rip_in_hpc_spind_mod.append(rip_in_hpc_spind_mod)
	session_rip_in_thl_spind_phase[session] = rip_in_thl_spind_phase
	session_rip_in_hpc_spind_phase[session] = rip_in_hpc_spind_phase


##################################################################################################
# STORING
##################################################################################################		
	store 			= pd.HDFStore("../data/phase_spindles/"+session.split("/")[1]+".lfp")
	store['lfp_thl'] = lfp_thl.as_dataframe()
	store['lfp_hpc'] = lfp_hpc.as_series()
	store.close()	
	
	store_spike 	= pd.HDFStore("../data/spikes_thalamus/"+session.split("/")[1]+".spk")
	for n in spikes.keys(): store_spike[str(n)] = spikes[n].as_series()
	store_spike.close()






sys.exit()

allkappa = np.vstack(allkappa)	

a = allkappa[np.sum(allkappa > 0.0, 1) == 4]
meankappa = np.array(meankappa)

plot(allkappa.transpose(), 'o-')

plot(meankappa.transpose())
xticks([0,1,2,3], ['Hpc phase(spind Hpc)', 'Hpc phase(spind Hpc^THL)', 'Thl phase(spind THL)', 'THL phase (spind HPC^THL)'])
ylabel(" mean kappa per session")

show()

figure()
subplot(211)
for session in datasets:
	hist, bin_edges = np.histogram(session_rip_in_hpc_spind_phase[session][0].values, 100, density = True)
	y = gaussFilt(hist, (5,))
	x = bin_edges[0:-1] + (bin_edges[1] - bin_edges[0])/2
	kappa = session_rip_in_hpc_spind_mod[np.where(datasets == session)[0][0]][0][-1]
	print(kappa)
	plot(x, y, '-', linewidth = kappa*2)
title("Ripples phase in hippocampal spindles per session")	
subplot(212)
for session in datasets:
	hist, bin_edges = np.histogram(session_rip_in_thl_spind_phase[session][0].values, 100, density = True)
	y = gaussFilt(hist, (5,))
	x = bin_edges[0:-1] + (bin_edges[1] - bin_edges[0])/2
	kappa = session_rip_in_thl_spind_mod[np.where(datasets == session)[0][0]][0][-1]
	print(kappa)
	plot(x, y, '-', linewidth = kappa*2)
title("Ripples phase in thalamic spindles per session")		
show()

