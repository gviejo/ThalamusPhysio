

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
import _pickle as cPickle
import time
import os, sys
import ipyparallel
import neuroseries as nts
from numba import jit

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

@jit(nopython=True)
def histo(spk, obins):
	n = len(obins)
	count = np.zeros(n)
	for i in range(n):
		count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
	return count

allspeed = []

for session in datasets:
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	if np.sum(hd_info)>5:
		
		############################################################################################################
		# LOADING DATA
		############################################################################################################		
		
		generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
		shankStructure 	= loadShankStructure(generalinfo)	
		if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
			hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
		else:
			hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
		spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])
		# restrict spikes to hd neurons
		hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])		
		spikes			= {k:spikes[k] for k in np.where(hd_info_neuron)[0]}
		wake_ep 		= loadEpoch(data_directory+session, 'wake')
		sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
		sws_ep 			= loadEpoch(data_directory+session, 'sws')
		rem_ep 			= loadEpoch(data_directory+session, 'rem')
		sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
		sws_ep 			= sleep_ep.intersect(sws_ep)	
		rem_ep 			= sleep_ep.intersect(rem_ep)
		# speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)	
		# speed_ep 		= nts.IntervalSet(speed[speed>2.5].index.values[0:-1], speed[speed>2.5].index.values[1:]).drop_long_intervals(26000).merge_close_intervals(50000)
		# wake_ep 		= wake_ep.intersect(speed_ep).drop_short_intervals(3000000)	
		n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')	
		try:
			ang = np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+'.ang')
		except:
			pass
		if len(ang.shape) > 1:		
			print(session)
			
			angle = nts.Tsd(t = ang[:,0], d = ang[:,1], time_units = 's').restrict(wake_ep).as_series()
			angle = angle[angle>0.0]
			angle = nts.Tsd(angle)		

			############################################################################################################
			# TUNING CURVES
			############################################################################################################
			tuning_curves 		= computeAngularTuningCurves(spikes, angle, wake_ep, nb_bins = 61, frequency = 1/0.0256)
			neuron_order = tuning_curves.columns.values

			# for i,n in enumerate(tuning_curves.columns.values):
			# 	subplot(4,5,i+1, projection = 'polar')
			# 	plot(tuning_curves[n])
			# show()

			############################################################################################################
			# BINING ACTIVITY AROUND RIPPLES
			############################################################################################################
			rip_ep,rip_tsd 	= loadRipples(data_directory+session)
			rip_ep			= sws_ep.intersect(rip_ep)	
			rip_tsd 		= rip_tsd.restrict(sws_ep)	

			bin_size = 40 # ms
			bins = np.arange(-1000, 1000+0.25*bin_size, 0.25*bin_size) - bin_size/2
			obins = np.vstack((bins,bins+bin_size)).T
			times = obins[:,0] + (np.diff(obins)/2).flatten()

			swr_angle 		= pd.DataFrame(index = times, columns = np.arange(len(rip_tsd)))
			swr_spike 		= pd.DataFrame(index = times, columns = np.arange(len(rip_tsd)))

			for i, t in enumerate(rip_tsd.as_units('ms').index.values):
				spike_counts 	= pd.DataFrame(index = times, columns = neuron_order)
				# tbins = t + bins
				tbins = t + obins
				for k in neuron_order:
					# spike_counts[k], _ = np.histogram(spikes[k].as_units('ms').index.values, tbins)
					spike_counts[k] = histo(spikes[k].as_units('ms').index.values, tbins)
				swr_spike.loc[:,i] = spike_counts.sum(1)
				tcurves_array = tuning_curves.values
				spike_counts_array = spike_counts.values

				proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))
				part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
				part2 = np.histogram(angle, np.linspace(0, 2*np.pi, 61), weights = np.ones_like(angle)/float(len(angle)))[0]
				for j in range(len(proba_angle)):
					part3 = np.prod(tcurves_array**spike_counts_array[j], 1)
					p = part1 * part2 * part3
					proba_angle[j] = p/p.sum() # Normalization process here

				proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)	
				proba_angle = proba_angle.astype('float')		

				angle = pd.Series(index = proba_angle.index.values, data = proba_angle.idxmax(1).values)
				# setting up some conditions here
				# angle[spike_counts.sum(1)<2] = np.nan		
				swr_angle[i] = angle

			tmp = pd.DataFrame(data = np.unwrap(swr_angle.values, axis=0), index = swr_angle.index)
			# tmp = tmp.rolling(window=20,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
			

			speed = np.abs(np.diff(tmp, axis = 0))
			speed = pd.DataFrame(index = times[0:-1], data = speed)

			# meanspeed = pd.Series(index = times[0:-1], data = speed.mean(1))
			allspeed.append(speed)
					
			# sys.exit()
allspeed = pd.concat(allspeed, 1)