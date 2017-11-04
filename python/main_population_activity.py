

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


# clients = ipyparallel.Client()
# print(clients.ids)
# dview = clients.direct_view()

session = datasets[0]

# def compute_population_correlation(session):
# 	data_directory = '/mnt/DataGuillaume/MergedData/'
# 	import numpy as np	
# 	import scipy.io	
# 	import scipy.stats		
# 	import _pickle as cPickle
# 	import time
# 	import os, sys	
# 	import neuroseries as nts
# 	from functions import loadShankStructure, loadSpikeData, loadEpoch, loadSpeed, loadXML, loadRipples, loadLFP, downsample, getPeaksandTroughs, butter_bandpass_filter
for session in datasets:
# for session in ['Mouse12/Mouse12-120810']:
	start_time = time.clock()
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
	rip_tsd 		= rip_tsd.restrict(rip_ep)

	theta_rem_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".rem.evt.theta")[:,0]
	theta_rem_ep 		= theta_rem_ep.reshape(len(theta_rem_ep)//2,2)
	theta_rem_ep 		= nts.IntervalSet(theta_rem_ep[:,0], theta_rem_ep[:,1], time_units = 'ms')

	theta_wake_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".wake.evt.theta")[:,0]
	theta_wake_ep 		= theta_wake_ep.reshape(len(theta_wake_ep)//2,2)
	theta_wake_ep 		= nts.IntervalSet(theta_wake_ep[:,0], theta_wake_ep[:,1], time_units = 'ms')


	# lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
	# lfp_hpc 		= downsample(lfp_hpc, 1, 5)
	# lfp_filt_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 5, 15, fs/5, 2))		
	# peaks, troughs	= getPeaksandTroughs(lfp_filt_hpc, 10)
	# power	 		= nts.Tsd(lfp_filt_hpc.index.values, np.abs(lfp_filt_hpc.values))
	# enveloppe,dummy	= getPeaksandTroughs(power, 5)	
	# index 			= (enveloppe > np.percentile(enveloppe, 30)).values*1.0
	# start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
	# end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
	# if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
	# if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
	# tmp 			= np.where(end_cand != start_cand)
	# start_cand 		= enveloppe.index.values[start_cand[tmp]]
	# end_cand	 	= enveloppe.index.values[end_cand[tmp]]
	# good_ep			= nts.IntervalSet(start_cand, end_cand)
	# good_ep			= good_ep.drop_short_intervals(300000)

	# theta_wake_ep 	= wake_ep.intersect(good_ep).merge_close_intervals(30000).drop_short_intervals(1000000)
	# theta_rem_ep 	= rem_ep.intersect(good_ep).merge_close_intervals(30000).drop_short_intervals(1000000)
	
	# troughs_wake	= troughs.restrict(theta_wake_ep)
	# troughs_rem		= troughs.restrict(theta_rem_ep)

	

	# from pylab import figure, plot, legend, show
	# figure()
	# plot([	wake_ep['start'].iloc[0], wake_ep['end'].iloc[0]], np.zeros(2), '-', color = 'blue', label = 'wake')
	# [plot([	wake_ep['start'].iloc[i], wake_ep['end'].iloc[i]], np.zeros(2), '-', color = 'blue') for i in range(len(wake_ep))]
	# plot([sleep_ep['start'].iloc[0], sleep_ep['end'].iloc[0]], np.zeros(2), '-', color = 'green', label = 'sleep')
	# [plot([sleep_ep['start'].iloc[i], sleep_ep['end'].iloc[i]], np.zeros(2), '-', color = 'green') for i in range(len(sleep_ep))]	
	# plot([rem_ep['start'].iloc[0], rem_ep['end'].iloc[0]],  np.zeros(2)+0.1, '-', color = 'orange', label = 'rem')
	# [plot([rem_ep['start'].iloc[i], rem_ep['end'].iloc[i]], np.zeros(2)+0.1, '-', color = 'orange') for i in range(len(rem_ep))]

	# plot([sws_ep['start'].iloc[0], sws_ep['end'].iloc[0]],  np.zeros(2)+0.1, '-', color = 'red', label = 'sws')
	# [plot([sws_ep['start'].iloc[i], sws_ep['end'].iloc[i]], np.zeros(2)+0.1, '-', color = 'red') for i in range(len(sws_ep))]	

	# plot([theta_rem_ep['start'].iloc[0], theta_rem_ep['end'].iloc[0]],  np.zeros(2)+0.2, '-', color = 'black', label = 'theta_rem')
	# [plot([theta_rem_ep['start'].iloc[i], theta_rem_ep['end'].iloc[i]], np.zeros(2)+0.2, '-', color = 'black') for i in range(len(theta_rem_ep))]	
	# plot([theta_wake_ep['start'].iloc[0], theta_wake_ep['end'].iloc[0]],  np.zeros(2)+0.2, '-', color = 'purple', label = 'theta_wake')
	# [plot([theta_wake_ep['start'].iloc[i], theta_wake_ep['end'].iloc[i]], np.zeros(2)+0.2, '-', color = 'purple') for i in range(len(theta_wake_ep))]	

	# legend()
	# show()
	store 			= pd.HDFStore("../data/population_activity_50ms/"+session.split("/")[1]+".h5")
	# store 			= pd.HDFStore("../data/population_activity_"+session.split("/")[1]+"_special_one_for_testing.h5")
	n_neurons = len(spikes)
	keys_neurons = list(spikes.keys())					
	if len(theta_wake_ep) and len(theta_rem_ep) and len(sleep_ep) == 2:
		# ###############################################################################################################
		# # SWR
		# ###############################################################################################################				
		# # population activity vector				
		# n_ripples = len(rip_ep)
		# 
		
		# rip_pop = np.zeros((n_ripples,n_neurons))
	
		# for i in range(n_ripples):						
		# 	start, stop = rip_ep.iloc[i]
		# 	start -= 50000
		# 	stop += 50000			
		# 	for j in range(n_neurons): 
		# 		rip_pop[i,j] = np.sum(np.logical_and(spikes[keys_neurons[j]].index.values >= start, spikes[keys_neurons[j]].index.values <= stop))

		# # normalize by each ripple length
		# duree = (rip_ep.as_units('s')['end'] - rip_ep.as_units('s')['start']).values
		# rip_pop = rip_pop/np.vstack(duree)
		# # divide by mean firing rate
		# # rip_pop = rip_pop / (rip_pop.mean(0)+1e-12)
		# rip_pop = pd.DataFrame(index = (rip_ep.as_units('s')['start'].values + duree)*1000*1000, columns = keys_neurons, data = rip_pop)

		# store.put('rip', rip_pop)
		# store.put('rip_ep', pd.DataFrame(rip_ep))

		# ###############################################################################################################
		# # THETA REM
		# ###############################################################################################################
		# # population activity vector		
		# rem_pop = np.zeros((len(troughs_rem) - len(theta_rem_ep), n_neurons))
		# count = 0
		# index_timing = np.zeros(rem_pop.shape[0])
		# for i in theta_rem_ep.index:			
		# 	index = np.logical_and(troughs_rem.index.values>=theta_rem_ep.loc[i]['start'], troughs_rem.index.values<=theta_rem_ep.loc[i]['end'])
		# 	n_cycle = index.sum()-1
		# 	timestep = troughs_rem.index.values[index]			
		# 	for j in range(len(timestep)-1):
		# 		start, stop = timestep[j:j+2]
		# 		duree = stop-start
		# 		for k in range(n_neurons):
		# 			rem_pop[count,k] = float(np.sum(np.logical_and(spikes[keys_neurons[k]].index.values >= start, spikes[keys_neurons[k]].index.values <= stop)))/(duree/1000/1000)
		# 		index_timing[count] = start + (duree)/2					
		# 		count += 1
				
		# rem_pop = pd.DataFrame(index = index_timing, columns = keys_neurons, data = rem_pop)

		# store.put('rem', rem_pop)
		# store.put('theta_rem_ep', pd.DataFrame(theta_rem_ep))

		# ###############################################################################################################
		# # THETA WAKE
		# ###############################################################################################################
		# # population activity vector		
		# wake_pop = np.zeros((len(troughs_wake) - len(theta_wake_ep), n_neurons))
		# count = 0
		# index_timing = np.zeros(wake_pop.shape[0])
		# for i in theta_wake_ep.index:			
		# 	index = np.logical_and(troughs_wake.index.values>=theta_wake_ep.loc[i]['start'], troughs_wake.index.values<=theta_wake_ep.loc[i]['end'])
		# 	n_cycle = index.sum()-1
		# 	timestep = troughs_wake.index.values[index]			
		# 	for j in range(len(timestep)-1):
		# 		start, stop = timestep[j:j+2]
		# 		duree = stop-start
		# 		for k in range(n_neurons):
		# 			wake_pop[count,k] = float(np.sum(np.logical_and(spikes[keys_neurons[k]].index.values >= start, spikes[keys_neurons[k]].index.values <= stop)))/(duree/1000/1000)
		# 		index_timing[count] = start + (duree)/2					
		# 		count += 1
				
		# wake_pop = pd.DataFrame(index = index_timing, columns = keys_neurons, data = wake_pop)

		# store.put('wake', wake_pop)
		# store.put('theta_wake_ep', pd.DataFrame(theta_wake_ep))
			
		###############################################################################################################
		# BINNING ALL WAKE 100 ms
		###############################################################################################################
		# population activity vector			
		wake_ep 		= loadEpoch(data_directory+session, 'wake')
		bins 			= np.arange(wake_ep['start'][0], wake_ep['end'][0], 50000)
		all_wake_pop 	= np.zeros((len(bins)-1, n_neurons))
		for n in range(n_neurons):
			all_wake_pop[:,n] = np.histogram(spikes[n].index.values, bins)[0]
		all_wake_pop /= 0.050
		all_wake_pop = pd.DataFrame(index = bins[0:-1] + 25000, columns = keys_neurons, data = all_wake_pop)

		store.put('allwake', all_wake_pop)
			
		###############################################################################################################
		# BINNING ALL SLEEP 100 ms
		###############################################################################################################
		# population activity vector			
		
		all_sleep_pop = []
		for e in range(len(sleep_ep)):
			bins = np.arange(sleep_ep['start'][e], sleep_ep['end'][e], 50000)
			sleep_pop 	= np.zeros((len(bins)-1, n_neurons))
			for n in range(n_neurons):				
				sleep_pop[:,n] = np.histogram(spikes[n].index.values, bins)[0]
			sleep_pop /= 0.050
			sleep_pop = pd.DataFrame(index = bins[0:-1] + 25000, columns = keys_neurons, data = sleep_pop)
			all_sleep_pop.append(sleep_pop)		
		store.put('presleep', all_sleep_pop[0])
		store.put('postsleep', all_sleep_pop[1])

	store.close()

	# return 

# a = dview.map_sync(compute_population_correlation, datasets)