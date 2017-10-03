#!/usr/bin/env python
'''
	File name: main_make_Spindleinfo.py
	Author: Guillaume Viejo
	Date created: 16/08/2017    
	Python Version: 3.5.2

Spindle modulation, return angle

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


for session in datasets:
	print("session"+session)
	start = time.time()

	generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
	shankStructure 	= loadShankStructure(generalinfo)	
	if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
	else:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
	print("Hpc channel ", hpc_channel)
	spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
	# wake_ep 		= loadEpoch(data_directory+session, 'wake')
	sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
	sws_ep 			= loadEpoch(data_directory+session, 'sws')
	# rem_ep 			= loadEpoch(data_directory+session, 'rem')
	sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
	sws_ep 			= sleep_ep.intersect(sws_ep)	
	# rem_ep 			= sleep_ep.intersect(rem_ep)
	# speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)	
	# speed_ep 		= nts.IntervalSet(speed[speed>2.5].index.values[0:-1], speed[speed>2.5].index.values[1:]).drop_long_intervals(26000).merge_close_intervals(50000)
	# wake_ep 		= wake_ep.intersect(speed_ep)
	# good_ep 		= loadTheta(data_directory+session+'/Analysis/ThetaInfo.mat') # TODO
	# to match main_make_SWRinfo.py
	spikes 			= {n:spikes[n] for n in spikes.keys() if len(spikes[n].restrict(sws_ep))}
	n_neuron 		= len(spikes)
	n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')
	# lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
	# lfp_hpc 		= downsample(lfp_hpc, 1, 5)

	store 			= pd.HDFStore("../data/phase_spindles/"+session.split("/")[1]+".lfp")
	phase_hpc 		= nts.Tsd(store['phase_hpc_spindles'])
	phase_thl 		= nts.Tsd(store['phase_thl_spindles'])
	lfp_hpc 		= nts.Tsd(store['lfp_hpc'])
	lfp_thl			= nts.TsdFrame(store['lfp_thl'])
	store.close()		
	
##################################################################################################
# DETECTION UP/DOWN States
##################################################################################################
	# print("up/down states")
	# # bins of 5000 us
	# bins 			= np.floor(np.arange(lfp_hpc.start_time(), lfp_hpc.end_time()+5000, 5000))
	# total_value 	= nts.Tsd(bins[0:-1]+(bins[1]-bins[0])/2, np.zeros(len(bins)-1)).restrict(sws_ep)
	# # each shank
	# for s in shankStructure['thalamus']:		
	# 	neuron_index = np.where(shank == s)[0]
	# 	if len(neuron_index):
	# 		tmp = {i:spikes[i] for i in neuron_index}
	# 		frate			= getFiringRate(tmp, bins)		
	# 		# ISI of 50 ms		
	# 		down_ep 		= nts.IntervalSet(frate[frate==0.0].index.values[0:-1], frate[frate==0.0].index.values[1:])
	# 		down_ep 		= down_ep.drop_long_intervals(5001).merge_close_intervals(0.0).intersect(sws_ep).drop_short_intervals(50000)
	# 		down_ep 		= down_ep.merge_close_intervals(50000)
	# 		down_candidates = nts.Tsd((down_ep['start'] + (down_ep['end'] - down_ep['start'])/2.).values, np.zeros(len(down_ep)))
	# 		# Smooth frate 
	# 		tmp 			= gaussFilt(frate.values, (2,))
	# 		frate 			= nts.Tsd(frate.index.values, tmp).restrict(sws_ep)
	# 		threshold	 	= np.percentile(frate, 20)
	# 		# frate < threshold around down candidates		
	# 		boole			= nts.Tsd((frate <= threshold)*1.0)
	# 		# realigner bool sur les candidates
	# 		boole 			= boole.realign(down_candidates)
	# 		down_ep 		= down_ep.iloc[np.where(boole)[0]]		
	# 		# add +1 in total value / pandas is weird # TODO
	# 		for i in total_value.restrict(down_ep).index.values:
	# 			total_value.loc[i] += 1
	# # at least 3 shank 
	# down_ep 		= nts.IntervalSet(total_value[total_value>=3.0].index.values[0:-1], total_value[total_value>=3.0].index.values[1:])
	# down_ep 		= down_ep.drop_long_intervals(5001).merge_close_intervals(0.0)
	# down_ep 		= down_ep.merge_close_intervals(50000).drop_long_intervals(400000)
	# up_ep 			= nts.IntervalSet(down_ep['end'][0:-1], down_ep['start'][1:]).intersect(sws_ep)
	# up_ep 			= up_ep.drop_short_intervals(500000)

##################################################################################################
# DETECTION Spindles in thalamus
##################################################################################################	
	# print("spindles")
	# thl_channels 	= list(np.sort([ch for k in shankStructure['thalamus'] for ch in shank_to_channel[k]]))	
	thl_channels 	= list(np.sort([shank_to_channel[k][0] for k in shankStructure['thalamus']]))		
	# lfp_thl 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, thl_channels, float(fs), 'int16')	
	# lfp_thl 		= downsample(lfp_thl, 1, 5)	
	# filter of each shank	
	lfp_filt 		= nts.TsdFrame(lfp_thl.index.values, np.zeros(lfp_thl.shape))
	for i in lfp_thl.keys():
		lfp_filt[i] = butter_bandpass_filter(lfp_thl[i].values, 8, 18, fs/5, 2)
	lfp_filt		= lfp_filt.restrict(sws_ep)
	lfp_mean 		= nts.Tsd(lfp_filt.mean(1))	
	power	 		= nts.Tsd(lfp_mean.index.values, np.abs(lfp_mean.values))
	enveloppe,dummy	= getPeaksandTroughs(power, 5)	
	index 			= (enveloppe > np.percentile(enveloppe, 50)).values*1.0
	start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
	end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
	if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
	if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
	tmp 			= np.where(end_cand != start_cand)
	start_cand 		= enveloppe.index.values[start_cand[tmp]]
	end_cand	 	= enveloppe.index.values[end_cand[tmp]]
	spind_ep_thl	= nts.IntervalSet(start_cand, end_cand)
	spind_ep_thl	= spind_ep_thl.drop_short_intervals(200000).drop_long_intervals(3000000)
	#count number of cycle, shoulb be superior to five peaks and troughs
	peaks, troughs 	= getPeaksandTroughs(lfp_mean, 5)
	index 		 	= np.zeros(len(spind_ep_thl))
	for i in range(len(spind_ep_thl)):
		n_peaks = len(peaks.restrict(nts.IntervalSet(spind_ep_thl.iloc[i]['start'], spind_ep_thl.iloc[i]['end'])))
		n_troughs = len(troughs.restrict(nts.IntervalSet(spind_ep_thl.iloc[i]['start'], spind_ep_thl.iloc[i]['end'])))		
		if n_peaks >= 4 and n_troughs >= 4 : index[i] = 1
	spind_ep_thl 	= spind_ep_thl[index==1]

	writeNeuroscopeEvents("/mnt/DataGuillaume/MergedData/"+session+"/"+session.split("/")[1]+".evt.spd.thl", spind_ep_thl, "Spindles")

	# phase, pwr		= getPhase(lfp_hpc, 8, 18, 16, fs/5., power = True)
	# phase 			= phase.restrict(sws_ep)
	phase 			= phase_hpc

	spikes_spind	= {n:spikes[n].restrict(spind_ep_thl) for n in spikes.keys()}
	spikes_phase	= {n:phase.realign(spikes_spind[n], align = 'closest') for n in spikes_spind.keys()}

	# spind_thl_mod 		= np.ones((n_neuron,3))*np.nan
	spind_thl_mod 		= {}
	for n in range(len(spikes_phase.keys())):
		neuron = list(spikes_phase.keys())[n]
		ph = spikes_phase[neuron]
		mu, kappa, pval = getCircularMean(ph.values)
		spind_thl_mod[session.split("/")[1]+"_"+str(neuron)] = np.array([mu, pval, kappa])


##################################################################################################
# DETECTION Spindles in hippocampus
##################################################################################################	
	#filter of each shank		
	lfp_filt_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 8, 18, fs/5, 2))
	lfp_filt_hpc	= lfp_filt_hpc.restrict(sws_ep)	
	power	 		= nts.Tsd(lfp_filt_hpc.index.values, np.abs(lfp_filt_hpc.values))
	enveloppe,dummy	= getPeaksandTroughs(power, 5)	
	index 			= (enveloppe > np.percentile(enveloppe, 50)).values*1.0
	start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
	end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
	if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
	if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
	tmp 			= np.where(end_cand != start_cand)
	start_cand 		= enveloppe.index.values[start_cand[tmp]]
	end_cand	 	= enveloppe.index.values[end_cand[tmp]]
	spind_ep_hpc	= nts.IntervalSet(start_cand, end_cand)
	spind_ep_hpc	= spind_ep_hpc.drop_short_intervals(200000).drop_long_intervals(3000000)
	#count number of cycle, shoulb be superior to six peaks and troughs
	peaks, troughs 	= getPeaksandTroughs(lfp_filt_hpc, 5)
	index 		 	= np.zeros(len(spind_ep_hpc))
	for i in range(len(spind_ep_hpc)):
		n_peaks = len(peaks.restrict(nts.IntervalSet(spind_ep_hpc.iloc[i]['start'], spind_ep_hpc.iloc[i]['end'])))
		n_troughs = len(troughs.restrict(nts.IntervalSet(spind_ep_hpc.iloc[i]['start'], spind_ep_hpc.iloc[i]['end'])))
		if n_peaks >= 4 and n_troughs >= 4 : index[i] = 1
	spind_ep_hpc 	= spind_ep_hpc[index==1]

	writeNeuroscopeEvents("/mnt/DataGuillaume/MergedData/"+session+"/"+session.split("/")[1]+".evt.spd.hpc", spind_ep_hpc, "Spindles")

	# phase, pwr		= getPhase(lfp_hpc, 8, 18, 16, fs/5., power = True)
	# phase 			= phase.restrict(sws_ep)
	phase 			= phase_hpc

	spikes_spind	= {n:spikes[n].restrict(spind_ep_hpc) for n in spikes.keys()}
	spikes_phase	= {n:phase.realign(spikes_spind[n], align = 'closest') for n in spikes_spind.keys()}

	# spind_hpc_mod 	= np.ones((n_neuron,3))*np.nan
	spind_hpc_mod 	= {}
	for n in range(len(spikes_phase.keys())):
		neuron = list(spikes_phase.keys())[n]
		ph = spikes_phase[neuron]
		mu, kappa, pval = getCircularMean(ph.values)
		spind_hpc_mod[session.split("/")[1]+"_"+str(neuron)] = np.array([mu, pval, kappa])
	
##################################################################################################
# DETECTION Spindles in thalamus FOR EACH SHANK
##################################################################################################		
	# # thl_channels 	= list(np.sort([ch for k in shankStructure['thalamus'] for ch in shank_to_channel[k]]))	
	# thl_channels 	= list(np.sort([shank_to_channel[k][0] for k in shankStructure['thalamus']]))		
	# lfp_thl 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, thl_channels, float(fs), 'int16')	
	# lfp_thl 		= downsample(lfp_thl, 1, 5)
	# # filter of each shank	
	# lfp_filt 		= nts.TsdFrame(lfp_thl.index.values, np.zeros(lfp_thl.shape))
	# for i in lfp_thl.keys():
	# 	lfp_filt[i] = butter_bandpass_filter(lfp_thl[i].values, 8, 18, fs/5, 2)
	# lfp_filt		= lfp_filt.restrict(sws_ep)
	# # lfp_mean 		= nts.Tsd(lfp_filt.mean(1))	
	# # power	 		= nts.Tsd(lfp_mean.index.values, np.abs(lfp_mean.values))
	# power 			= nts.TsdFrame(lfp_filt.as_dataframe().abs())
	# enveloppe,dummy	= getPeaksandTroughs(power, 5)	
	
	# spind_shank_ep 	= {}	
	# for i in range(enveloppe.shape[1]):
	# 	index 			= (enveloppe[i].dropna() > enveloppe[i].dropna().quantile(0.40)).values*1.0	
	# 	start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
	# 	end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
	# 	if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
	# 	if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
	# 	tmp 			= np.where(end_cand != start_cand)
	# 	start_cand 		= enveloppe.index.values[start_cand[tmp]]
	# 	end_cand	 	= enveloppe.index.values[end_cand[tmp]]
	# 	spind_ep_thl	= nts.IntervalSet(start_cand, end_cand)		
	# 	spind_ep_thl	= spind_ep_thl.drop_short_intervals(50000).drop_long_intervals(3000000)
	# 	spind_shank_ep[i] = spind_ep_thl
	

# ##################################################################################################
# # STORING
# ##################################################################################################		
# 	store 			= pd.HDFStore("../data/phase_spindles/"+session.split("/")[1]+".lfp")
# 	store['lfp_thl'] = lfp_thl.as_dataframe()
# 	store['lfp_hpc'] = lfp_hpc.as_series()
# 	store['phase_hpc_spindles'] = phase_hpc.as_series()
# 	store['phase_thl_spindles'] = phase_thl.as_dataframe()
# 	store.close()	
	
# 	store_spike 	= pd.HDFStore("../data/spikes_thalamus/"+session.split("/")[1]+".spk")
# 	for n in spikes.keys(): store_spike[str(n)] = spikes[n].as_series()
# 	store_spike.close()
	


	
	# sys.exit()

	stop = time.time()
	print(stop - start, ' s')		
	datatosave[session] = {	'thl':spind_thl_mod,
							'hpc':spind_hpc_mod}




# datatosave = dview.map_sync(make_theta_info, datasets)

import _pickle as cPickle
cPickle.dump(datatosave, open('/mnt/DataGuillaume/MergedData/SPINDLE_mod.pickle', 'wb'))

# zz = {}
# for s in range(8):
# 	zz[s] = []
# 	for session in datatosave.keys():
# 		zz[s].append(datatosave[session][s])
# 	zz[s] = np.vstack(zz[s])



	
# figure()
# for s in range(8):
# 	subplot(3,3,s+1)
# 	theta_mod_toplot = z['wake'][:,0]
# 	phi_toplot = zz[s][:,0]
# 	force = z['wake'][:,2] + zz[s][:,2] 
# 	x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
# 	y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
# 	scatter(x, y, s = force*10., label = 'theta wake')
# 	theta_mod_toplot = z['rem'][:,0]
	
# 	force = z['rem'][:,2] + zz[s][:,2] 
# 	x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
# 	y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
# 	scatter(x, y, s = force*10., label = 'theta rem')
# 	xlabel('Theta phase (rad)')
# 	ylabel('Spindle phase (rad)')
# 	title("Shank "+str(s))
# show()