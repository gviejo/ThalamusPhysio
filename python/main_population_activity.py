

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

# datasets = [s for s in datasets if 'Mouse17' in s]

# clients = ipyparallel.Client()
# print(clients.ids)
# dview = clients.direct_view()


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
# for session in [datasets[2]]:
	start_time = time.clock()
	print(session)
	generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
	shankStructure 	= loadShankStructure(generalinfo)	
	if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
	else:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
	spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
	hd_info 			= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	hd_info_neuron		= np.array([hd_info[n] for n in spikes.keys()])
	
	if np.sum(1-hd_info_neuron) > 5: 
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
		# rip_ep			= sws_ep.intersect(rip_ep)	
		# rip_tsd 		= rip_tsd.restrict(rip_ep)

		# theta_rem_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".rem.evt.theta")[:,0]
		# theta_rem_ep 		= theta_rem_ep.reshape(len(theta_rem_ep)//2,2)
		# theta_rem_ep 		= nts.IntervalSet(theta_rem_ep[:,0], theta_rem_ep[:,1], time_units = 'ms')

		# theta_wake_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".wake.evt.theta")[:,0]
		# theta_wake_ep 		= theta_wake_ep.reshape(len(theta_wake_ep)//2,2)
		# theta_wake_ep 		= nts.IntervalSet(theta_wake_ep[:,0], theta_wake_ep[:,1], time_units = 'ms')
		
		store_lfp 		= pd.HDFStore("/mnt/DataGuillaume/phase_spindles/"+session.split("/")[1]+".lfp")
		lfp_hpc 		= nts.Tsd(store_lfp['lfp_hpc'])
		store_lfp.close()
		lfp_filt_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 6, 14, fs/5, 2))		
		peaks, troughs	= getPeaksandTroughs(lfp_filt_hpc, 10)
		power	 		= nts.Tsd(lfp_filt_hpc.index.values, np.abs(lfp_filt_hpc.values))
		enveloppe,dummy	= getPeaksandTroughs(power, 5)	
		index 			= (enveloppe > np.percentile(enveloppe, 20)).values*1.0
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
		
		troughs_wake	= troughs.restrict(theta_wake_ep)
		troughs_rem		= troughs.restrict(theta_rem_ep)

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
		# convert spikes in a giant table
		keys_neurons = np.array(list(spikes.keys()))[hd_info_neuron==0]
		spike_table = pd.concat([spikes[k].fillna(1) for k in keys_neurons], axis = 1)
		spike_table.columns = keys_neurons
		n_neurons = len(keys_neurons)

		

		store 			= pd.HDFStore("/mnt/DataGuillaume/population_activity_nohd/"+session.split("/")[1]+".h5")

		if len(sleep_ep) == 2:
			###############################################################################################################
			# SWR
			###############################################################################################################				
			# population activity vector				
			n_ripples = len(rip_ep)		
			# rip_pop = np.zeros((n_ripples,n_neurons))	
			rip_pop = pd.DataFrame(index = rip_tsd.index.values, columns = keys_neurons)

			for i, t in enumerate(rip_tsd.index.values):						
				start, stop = rip_ep.iloc[i]
				rip_pop.iloc[i] = spike_table.loc[start:stop].fillna(0).sum(0)

			frate = rip_pop.sum()/rip_ep.tot_length('s')
			rip_pop = rip_pop.divide(pd.Series(index = rip_pop.index, data = (rip_ep['end'] - rip_ep['start']).values), 0)
			rip_pop = rip_pop/frate		
			rip_pop = rip_pop.fillna(0)
			store.put('rip', rip_pop)
			store.put('rip_ep', pd.DataFrame(rip_ep))

			###############################################################################################################
			# THETA REM
			###############################################################################################################
			rem_pop = pd.DataFrame(columns = keys_neurons)
			length_ev = []
			for start_ep, end_ep in zip(theta_rem_ep['start'], theta_rem_ep['end']):
				tr_in_ep = troughs_rem.as_series().loc[start_ep:end_ep].index.values
				if np.max((tr_in_ep[1:] - tr_in_ep[0:-1])/1000.) > 400: sys.exit()
				for start, stop in zip(tr_in_ep[0:-1], tr_in_ep[1:]):
					length_ev.append([(stop-start)/1000/1000])
					tmp = spike_table.loc[start:stop].fillna(0).sum(0)				
					rem_pop = rem_pop.append(tmp.to_frame().T.rename(index={0:start + (stop-start)/2}))

			frate = rem_pop.sum()/theta_rem_ep.tot_length('s')
			rem_pop = rem_pop.divide(pd.Series(index = rem_pop.index, data = np.array(length_ev).flatten()), 0)
			rem_pop = rem_pop/frate		
			rem_pop = rem_pop.fillna(0)
			store.put('rem', rem_pop)
			store.put('theta_rem_ep', pd.DataFrame(theta_rem_ep))

			###############################################################################################################
			# THETA WAKE
			###############################################################################################################
			wake_pop = pd.DataFrame(columns = keys_neurons)
			length_ev = []
			for start_ep, end_ep in zip(theta_wake_ep['start'], theta_wake_ep['end']):
				tr_in_ep = troughs_wake.as_series().loc[start_ep:end_ep].index.values
				if np.max((tr_in_ep[1:] - tr_in_ep[0:-1])/1000.) > 400: sys.exit()
				for start, stop in zip(tr_in_ep[0:-1], tr_in_ep[1:]):
					length_ev.append([(stop-start)/1000/1000])
					tmp = spike_table.loc[start:stop].fillna(0).sum(0)				
					wake_pop = wake_pop.append(tmp.to_frame().T.rename(index={0:start + (stop-start)/2}))
			
			frate = wake_pop.sum()/theta_wake_ep.tot_length('s')
			wake_pop = wake_pop.divide(pd.Series(index = wake_pop.index, data = np.array(length_ev).flatten()), 0)
			wake_pop = wake_pop/frate
			wake_pop = wake_pop.fillna(0)
			store.put('wake', wake_pop)
			store.put('theta_wake_ep', pd.DataFrame(theta_wake_ep))
				
		# 	###############################################################################################################
		# 	# BINNING ALL WAKE 100 ms
		# 	###############################################################################################################
		# 	# population activity vector			
		# 	bin_size = 5000
		# 	wake_ep 		= loadEpoch(data_directory+session, 'wake')
		# 	bins 			= np.arange(wake_ep['start'][0], wake_ep['end'][0], bin_size)
		# 	all_wake_pop 	= np.zeros((len(bins)-1, n_neurons))
		# 	for n in range(n_neurons):			
		# 		all_wake_pop[:,n] = np.histogram(spikes[n].index.values, bins)[0]
		# 	all_wake_pop /= (bin_size/1000)
		# 	all_wake_pop = pd.DataFrame(index = bins[0:-1] + bin_size//2, columns = keys_neurons, data = all_wake_pop)
			

		# 	store.put('allwake', all_wake_pop)
				
		# 	###############################################################################################################
		# 	# BINNING ALL SLEEP 100 ms
		# 	###############################################################################################################
		# 	# population activity vector			
			
		# 	all_sleep_pop = []
		# 	for e in range(len(sleep_ep)):
		# 		bins = np.arange(sleep_ep['start'][e], sleep_ep['end'][e], bin_size)
		# 		sleep_pop 	= np.zeros((len(bins)-1, n_neurons))
		# 		for n in range(n_neurons):				
		# 			sleep_pop[:,n] = np.histogram(spikes[n].index.values, bins)[0]
		# 		sleep_pop /= (bin_size/1000)
		# 		sleep_pop = pd.DataFrame(index = bins[0:-1] + bin_size//2, columns = keys_neurons, data = sleep_pop)
		# 		all_sleep_pop.append(sleep_pop)		
		# 	store.put('presleep', all_sleep_pop[0])
		# 	store.put('postsleep', all_sleep_pop[1])

		store.close()

	
	

	# return 

# a = dview.map_sync(compute_population_correlation, datasets)