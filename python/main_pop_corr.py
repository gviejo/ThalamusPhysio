

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
import scipy.stats		

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')


clients = ipyparallel.Client()
print(clients.ids)
dview = clients.direct_view()

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
# 	import pandas as pd
# 	from functions import loadShankStructure, loadSpikeData, loadEpoch, loadSpeed, loadXML, loadRipples, loadLFP, downsample, getPeaksandTroughs, butter_bandpass_filter
# for session in datasets:
for session in ['Mouse20/Mouse20-130516']:	
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
	hd_info 			= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	hd_info_neuron		= np.array([hd_info[n] for n in spikes.keys()])		
	
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
	store 			= pd.HDFStore("../data/population_activity/"+session.split("/")[1]+".h5")
	rip_pop 		= store['rip']
	rem_pop 		= store['rem']
	wak_pop 		= store['wake']
	all_pop 		= store['allwake']
	store.close()
	
	rip_pop = rip_pop[np.where(hd_info_neuron == 0)[0]]
	rem_pop = rem_pop[np.where(hd_info_neuron == 0)[0]]
	wak_pop = wak_pop[np.where(hd_info_neuron == 0)[0]]
	
	###############################################################################################################
	# POPULATION CORRELATION FOR EACH RIPPLES
	###############################################################################################################
	#matrix of distance between ripples	in second	
	interval_mat = np.vstack(nts.TsdFrame(rip_pop).as_units('s').index.values) - nts.TsdFrame(rip_pop).as_units('s').index.values	
	rip_corr = np.ones(interval_mat.shape)*np.nan
	index = np.where(np.logical_and(interval_mat < 3.0, interval_mat >= 0.0))		
	for i, j in zip(index[0], index[1]):					
		rip_corr[i,j] = scipy.stats.pearsonr(rip_pop.iloc[i].values, rip_pop.iloc[j].values)[0]		
		rip_corr[j,i] = rip_corr[i,j]

	allrip_corr = pd.DataFrame(index = interval_mat[index], data = rip_corr[index])
	rip_corr = pd.DataFrame(index = rip_pop.index.values, data = rip_corr, columns = rip_pop.index.values)		
	
	###############################################################################################################
	# POPULATION CORRELATION FOR EACH THETA CYCLE OF REM
	###############################################################################################################
	# compute all time interval for each ep of theta
	interval_mat = np.vstack(nts.TsdFrame(rem_pop).as_units('s').index.values) - nts.TsdFrame(rem_pop).as_units('s').index.values
	rem_corr = np.ones(interval_mat.shape)*np.nan
	index = np.where(np.logical_and(interval_mat < 3.0, interval_mat >= 0.0))
	for i, j in zip(index[0], index[1]):
		rem_corr[i,j] = scipy.stats.pearsonr(rem_pop.iloc[i].values, rem_pop.iloc[j].values)[0]
		rem_corr[j,i] = rem_corr[i,j]
	
	allrem_corr = pd.DataFrame(index = interval_mat[index], data = rem_corr[index])
	rem_corr = pd.DataFrame(index = rem_pop.index.values, data = rem_corr, columns = rem_pop.index.values)		

	###############################################################################################################
	# POPULATION CORRELATION FOR EACH THETA CYCLE OF WAKE
	###############################################################################################################
	# compute all time interval for each ep of theta
	interval_mat = np.vstack(nts.TsdFrame(wak_pop).as_units('s').index.values) - nts.TsdFrame(wak_pop).as_units('s').index.values
	wak_corr = np.ones(interval_mat.shape)*np.nan
	index = np.where(np.logical_and(interval_mat < 3.0, interval_mat >= 0.0))
	for i, j in zip(index[0], index[1]):
		wak_corr[i,j] = scipy.stats.pearsonr(wak_pop.iloc[i].values, wak_pop.iloc[j].values)[0]
		wak_corr[j,i] = wak_corr[i,j]

	allwak_corr = pd.DataFrame(index = interval_mat[index], data = wak_corr[index])
	wak_corr = pd.DataFrame(index = wak_pop.index.values, data = wak_corr, columns = wak_pop.index.values)		
	
	###############################################################################################################
	# STORING
	###############################################################################################################
	store 			= pd.HDFStore("../data/corr_pop_no_hd/"+session.split("/")[1]+".h5")
	store.put('rip_corr', rip_corr)
	store.put('allrip_corr', allrip_corr)
	store.put('wak_corr', wak_corr)
	store.put('allwak_corr', allwak_corr)
	store.put('rem_corr', rem_corr)
	store.put('allrem_corr', allrem_corr)	
	store.close()
	print(time.clock() - start_time, "seconds")
	# # return 

# a = dview.map_sync(compute_population_correlation, datasets)
	


# ###############################################################################################################
# # PLOT
# ###############################################################################################################
# last = np.max([np.max(allrip_corr[:,0]),np.max(alltheta_corr[:,0])])
# bins = np.arange(0.0, last, 0.2)
# # average rip corr
# index_rip = np.digitize(allrip_corr[:,0], bins)
# mean_ripcorr = np.array([np.mean(allrip_corr[index_rip == i,1]) for i in np.unique(index_rip)[0:30]])
# # average theta corr
# index_theta = np.digitize(alltheta_corr[:,0], bins)
# mean_thetacorr = np.array([np.mean(alltheta_corr[index_theta == i,1]) for i in np.unique(index_theta)[0:30]])


# xt = list(bins[0:30][::-1]*-1.0)+list(bins[0:30])
# ytheta = list(mean_thetacorr[0:30][::-1])+list(mean_thetacorr[0:30])
# yrip = list(mean_ripcorr[0:30][::-1])+list(mean_ripcorr[0:30])
# plot(xt, ytheta, 'o-', label = 'theta')
# plot(xt, yrip, 'o-', label = 'ripple')
# legend()
# xlabel('s')
# ylabel('r')
# show()



