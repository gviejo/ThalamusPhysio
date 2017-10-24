#!/usr/bin/env python
'''
    File name: main_make_Thetainfo.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2

Theta modulation, returns angle 
Used to make figure 1
# TODO ASK ADRIEN ABOUT THE RESTRICTION BY SLEEP_EP
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
spikes_theta_phase = {'wake':{},'rem':{}}

for session in datasets:	
	print(session)
	start = time.time()

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
	speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)	
	speed_ep 		= nts.IntervalSet(speed[speed>2.5].index.values[0:-1], speed[speed>2.5].index.values[1:]).drop_long_intervals(26000).merge_close_intervals(50000)
	wake_ep 		= wake_ep.intersect(speed_ep).drop_short_intervals(3000000)
	# to match main_make_SWRinfo.py
	spikes 			= {n:spikes[n] for n in spikes.keys() if len(spikes[n].restrict(sws_ep))}
	n_neuron 		= len(spikes)
	n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')	
	lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
	lfp_hpc 		= downsample(lfp_hpc, 1, 5)

##################################################################################################
# DETECTION THETA
##################################################################################################		
	lfp_filt_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 5, 15, fs/5, 2))	
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
	good_ep			= nts.IntervalSet(start_cand, end_cand)
	good_ep			= good_ep.drop_short_intervals(300000)

	theta_wake_ep 	= wake_ep.intersect(good_ep).merge_close_intervals(30000).drop_short_intervals(1000000)
	theta_rem_ep 	= rem_ep.intersect(good_ep).merge_close_intervals(30000).drop_short_intervals(1000000)
	

	writeNeuroscopeEvents("/mnt/DataGuillaume/MergedData/"+session+"/"+session.split("/")[1]+".wake.evt.theta", theta_wake_ep, "Theta")
	writeNeuroscopeEvents("/mnt/DataGuillaume/MergedData/"+session+"/"+session.split("/")[1]+".rem.evt.theta", theta_rem_ep, "Theta")
	
	
	phase 			= getPhase(lfp_hpc, 6, 14, 16, fs/5.)	
	ep 				= { 'wake'	: theta_wake_ep,
						'rem'	: theta_rem_ep}
	theta_mod 		= {}
	


	for e in ep.keys():		
		spikes_phase	= {n:phase.realign(spikes[n], align = 'closest') for n in spikes.keys()}

		# theta_mod[e] 	= np.ones((n_neuron,3))*np.nan
		theta_mod[e] 	= {}
		for n in range(len(spikes_phase.keys())):			
			neuron = list(spikes_phase.keys())[n]
			ph = spikes_phase[neuron].restrict(ep[e])
			mu, kappa, pval = getCircularMean(ph.values)
			theta_mod[e][session.split("/")[1]+"_"+str(neuron)] = np.array([mu, pval, kappa])
			spikes_theta_phase[e][session.split("/")[1]+"_"+str(neuron)] = ph.values

	
	stop = time.time()
	print(stop - start, ' s')		
	datatosave[session] = theta_mod



import _pickle as cPickle
cPickle.dump(datatosave, open('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', 'wb'))
cPickle.dump(spikes_theta_phase, open('/mnt/DataGuillaume/MergedData/SPIKE_THETA_PHASE.pickle', 'wb'))

