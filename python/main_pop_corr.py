

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

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')


clients = ipyparallel.Client()
print(clients.ids)
dview = clients.direct_view()

session = datasets[0]

def compute_population_correlation(session):
	data_directory = '/mnt/DataGuillaume/MergedData/'
	import numpy as np	
	import scipy.io	
	import scipy.stats		
	import _pickle as cPickle
	import time
	import os, sys	
	import neuroseries as nts
	from functions import loadShankStructure, loadSpikeData, loadEpoch, loadSpeed, loadXML, loadRipples, loadLFP, downsample, getPeaksandTroughs, butter_bandpass_filter
# for session in datasets:
	print(session)
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
	n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')	

	rip_ep,rip_tsd 	= loadRipples(data_directory+session)
	rip_ep			= sws_ep.intersect(rip_ep)	
	rip_tsd 		= rip_tsd.restrict(sws_ep)

	theta_rem_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".rem.evt.theta")[:,0]
	theta_rem_ep 		= theta_rem_ep.reshape(len(theta_rem_ep)//2,2)
	theta_rem_ep 		= nts.IntervalSet(theta_rem_ep[:,0], theta_rem_ep[:,1], time_units = 'ms')

	theta_wake_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".wake.evt.theta")[:,0]
	theta_wake_ep 		= theta_wake_ep.reshape(len(theta_wake_ep)//2,2)
	theta_wake_ep 		= nts.IntervalSet(theta_wake_ep[:,0], theta_wake_ep[:,1], time_units = 'ms')


	lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
	lfp_hpc 		= downsample(lfp_hpc, 1, 5)
	lfp_filt_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 5, 15, fs/5, 2))		
	peaks, troughs	= getPeaksandTroughs(lfp_filt_hpc, 10)
	
	troughs_wake	= troughs.restrict(theta_wake_ep)
	troughs_rem		= troughs.restrict(theta_rem_ep)

	
	if len(theta_wake_ep) and len(theta_rem_ep):		
		###############################################################################################################
		# POPULATION CORRELATION FOR EACH RIPPLES
		###############################################################################################################
		n_ripples = len(rip_ep)
		n_neurons = len(spikes)
		# population activity vector
		pop = np.zeros( (n_ripples, n_neurons) )
		keys_neurons = list(spikes.keys())			
		for i in range(n_ripples):									
			start, stop = rip_ep.iloc[i]
			for j in range(n_neurons): 
				pop[i,j] = np.sum(np.logical_and(spikes[keys_neurons[j]].index.values >= start, spikes[keys_neurons[j]].index.values <= stop))

		# normalize by each ripple length
		pop = pop/np.vstack((rip_ep.as_units('s')['end'] - rip_ep.as_units('s')['start']).values)		
		# divide by mean firing rate
		pop = pop / (pop.mean(0)+1e-12)
		
		# rip_corr = np.zeros(n_ripples-1)

		#matrix of distance between ripples	in second
		interval_mat = np.vstack(rip_tsd.as_units('s').index.values) - rip_tsd.as_units('s').index.values
		# interval_index = np.logical_and(interval_mat < 5.0, interval_mat > 0.0)
		corr = np.ones(interval_mat.shape)
		index = np.tril_indices(n_ripples,-1)
		for i, j in zip(index[0], index[1]):	
			corr[i,j] = scipy.stats.pearsonr(pop[i], pop[j])[0]
			corr[j,i] = corr[i,j]

		rip_corr = (interval_mat, corr)
		allrip_corr = np.vstack((rip_corr[0][index], rip_corr[1][index])).transpose()

		###############################################################################################################
		# POPULATION CORRELATION FOR EACH THETA CYCLE OF REM
		###############################################################################################################
		# population activity vector
		pop = []
		for i in theta_rem_ep.index:
			start, stop = theta_rem_ep.loc[i]		
			index = np.logical_and(troughs_rem.index.values>start, troughs_rem.index.values<stop)
			n_cycle = index.sum()-1
			subpop = np.zeros( (n_cycle, n_neurons) )
			for j in range(n_cycle):
				cstart,cstop = troughs_rem[index].index.values[j:j+2]				
				for k in range(n_neurons):
					subpop[j,k] = float(np.sum(np.logical_and(spikes[k].index.values >= cstart, spikes[k].index.values <= cstop)))
			# normalized by each theta length
			dure = troughs_rem[index].as_units('s').index.values[1:] - troughs_rem[index].as_units('s').index.values[0:-1]
			subpop = subpop/np.vstack(dure)
			subpop = subpop/(subpop.mean(0)+1e-12)
			pop.append(subpop)
				

		# correlation
		# compute all time interval for each ep of theta
		theta_rem_corr = []
		alltheta_rem_corr = []
		for i in theta_rem_ep.index:
			start, stop = theta_rem_ep.loc[i]		
			index = np.logical_and(troughs_rem.index.values>start, troughs_rem.index.values<stop)
			# matrix of distance between creux
			interval_mat = np.vstack(troughs_rem[index].as_units('s').index.values[0:-1]) - troughs_rem[index].as_units('s').index.values[0:-1]
			corr = np.ones(interval_mat.shape)
			subpop = pop[i]
			xx = np.tril_indices(interval_mat.shape[0],-1)
			for j, k in zip(xx[0], xx[1]):	
				corr[j,k] = scipy.stats.pearsonr(subpop[j], subpop[k])[0]
				corr[k,j] = corr[j,k]	
			
			theta_rem_corr.append((interval_mat,corr))
			
			alltheta_rem_corr.append(np.vstack((theta_rem_corr[i][0][xx], theta_rem_corr[i][1][xx])).transpose())

		alltheta_rem_corr = np.vstack(alltheta_rem_corr)

		###############################################################################################################
		# POPULATION CORRELATION FOR EACH THETA CYCLE OF WAKE
		###############################################################################################################
		# population activity vector
		pop = []
		for i in theta_wake_ep.index:
			start, stop = theta_wake_ep.loc[i]		
			index = np.logical_and(troughs_wake.index.values>start, troughs_wake.index.values<stop)
			n_cycle = index.sum()-1
			subpop = np.zeros( (n_cycle, n_neurons) )
			for j in range(n_cycle):
				cstart,cstop = troughs_wake[index].index.values[j:j+2]				
				for k in range(n_neurons):
					subpop[j,k] = float(np.sum(np.logical_and(spikes[k].index.values >= cstart, spikes[k].index.values <= cstop)))
			# normalized by each theta length
			dure = troughs_wake[index].as_units('s').index.values[1:] - troughs_wake[index].as_units('s').index.values[0:-1]
			subpop = subpop/np.vstack(dure)
			subpop = subpop/(subpop.mean(0)+1e-12)
			pop.append(subpop)
				

		# correlation
		# compute all time interval for each ep of theta
		theta_wake_corr = []
		alltheta_wake_corr = []
		for i in theta_wake_ep.index:
			start, stop = theta_wake_ep.loc[i]		
			index = np.logical_and(troughs_wake.index.values>start, troughs_wake.index.values<stop)
			# matrix of distance between creux
			interval_mat = np.vstack(troughs_wake[index].as_units('s').index.values[0:-1]) - troughs_wake[index].as_units('s').index.values[0:-1]
			corr = np.ones(interval_mat.shape)
			subpop = pop[i]
			xx = np.tril_indices(interval_mat.shape[0],-1)
			for j, k in zip(xx[0], xx[1]):	
				corr[j,k] = scipy.stats.pearsonr(subpop[j], subpop[k])[0]
				corr[k,j] = corr[j,k]	
			
			theta_wake_corr.append((interval_mat,corr))
			
			alltheta_wake_corr.append(np.vstack((theta_wake_corr[i][0][xx], theta_wake_corr[i][1][xx])).transpose())

		alltheta_wake_corr = np.vstack(alltheta_wake_corr)




		# print(time.clock() - start_time, "seconds")
		# print('\n')

		tosave =	 {	'rip_corr':rip_corr,
						'theta_wake_corr':theta_wake_corr,
						'theta_rem_corr':theta_rem_corr
				}
		cPickle.dump(tosave, open('../data/corr_pop/'+session.split("/")[1]+".pickle", 'wb'))

		return session
	else:
		return 0

a = dview.map_sync(compute_population_correlation, datasets)
	


###############################################################################################################
# PLOT
###############################################################################################################
last = np.max([np.max(allrip_corr[:,0]),np.max(alltheta_corr[:,0])])
bins = np.arange(0.0, last, 0.2)
# average rip corr
index_rip = np.digitize(allrip_corr[:,0], bins)
mean_ripcorr = np.array([np.mean(allrip_corr[index_rip == i,1]) for i in np.unique(index_rip)[0:30]])
# average theta corr
index_theta = np.digitize(alltheta_corr[:,0], bins)
mean_thetacorr = np.array([np.mean(alltheta_corr[index_theta == i,1]) for i in np.unique(index_theta)[0:30]])


xt = list(bins[0:30][::-1]*-1.0)+list(bins[0:30])
ytheta = list(mean_thetacorr[0:30][::-1])+list(mean_thetacorr[0:30])
yrip = list(mean_ripcorr[0:30][::-1])+list(mean_ripcorr[0:30])
plot(xt, ytheta, 'o-', label = 'theta')
plot(xt, yrip, 'o-', label = 'ripple')
legend()
xlabel('s')
ylabel('r')
show()



