#!/usr/bin/env python


import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import neuroseries as nts


data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
session 		= 'Mouse12/Mouse12-120810'
neurons 		= [session.split("/")[1]+"_"+str(u) for u in [38,37,40]]

#############################################################################################################
# GENERAL INFO
#############################################################################################################
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
allneurons 		= [session.split("/")[1]+"_"+str(list(spikes.keys())[i]) for i in spikes.keys()]
n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')
theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta_mod_rem 	= pd.DataFrame(index = theta_ses['rem'], columns = ['phase', 'pvalue', 'kappa'], data = theta_mod['rem'])
theta_mod_rem 	= theta_mod_rem.loc[allneurons]
theta_mod_rem 	= theta_mod_rem.sort_values('phase')

#############################################################################################################
# THETA EPISODE
#############################################################################################################
theta_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".rem.evt.theta")[:,0]
theta_ep 		= theta_ep.reshape(len(theta_ep)//2,2)
theta_ep 		= nts.IntervalSet(theta_ep[:,0], theta_ep[:,1], time_units = 'ms')

#############################################################################################################
# TO SAVE
#############################################################################################################
path_snippet 	= "../figures/figures_articles/figure1/"
#saveh5 			= pd.HDFStore(path_snippet+'snippet_'+session.split("/")[1]+'.h5')

#############################################################################################################
# SWR EPISODE
#############################################################################################################
lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
lfp_filt_rip_band_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 100, 300, fs, 2))	

rip_ep,rip_tsd 	= loadRipples(data_directory+session)
spike_rip_count = np.zeros((3,len(rip_ep)))
for e in rip_ep.index:
	start, stop = rip_ep.loc[e]
	for n in neurons:
		spike_rip_count[neurons.index(n),e] = len(spikes[int(n.split("_")[1])].loc[start:stop])

spike_rip_count = pd.DataFrame(index = rip_tsd.index, data = spike_rip_count.transpose())


time_bet_rip = rip_tsd.as_units('s').index.values[1:] - rip_tsd.as_units('s').index.values[0:-1]

start_rip, end_rip = (92150000,92950000)

H0 = []
Hm = []
Hstd = []
bin_size 	= 5 # ms 
nb_bins 	= 200
confInt 	= 0.95
nb_iter 	= 1000
jitter  	= 150 # ms					
for n in neurons:
	a, b, c, d, e, f = 	xcrossCorr_fast(rip_tsd.as_units('ms').index.values, spikes[int(n.split("_")[1])].as_units('ms').index.values, bin_size, nb_bins, nb_iter, jitter, confInt)
	H0.append(a)
	Hm.append(b)
	Hstd.append(e)

H0 		= pd.DataFrame(index = f, data = np.array(H0).transpose(), columns = neurons)
Hm 		= pd.DataFrame(index = f, data = np.array(Hm).transpose(), columns = neurons)
Hstd 	= pd.DataFrame(data = np.array(Hstd), index = neurons)



#saveh5.put('H0', H0)
#saveh5.put('Hm', Hm)
#saveh5.put('Hstd', Hstd)
#saveh5.put('lfp_filt_hpc_swr', lfp_filt_rip_band_hpc.loc[start_rip:end_rip].as_series())
#saveh5.put('lfp_hpc_swr', lfp_hpc.loc[start_rip:end_rip].as_series())
# for n in neurons:
	#saveh5.put('spike_swr'+n, spikes[int(n.split("_")[1])].loc[start_rip:end_rip].as_series())
# index = [np.where(x == rip_tsd.index.values)[0][0] for x in rip_tsd.loc[start_rip:end_rip].index.values]
#saveh5.put('swr_ep', pd.DataFrame(rip_ep.loc[index]))




figure()
ax1 = subplot(211)
plot(lfp_hpc.loc[start_rip:end_rip])
plot(lfp_filt_rip_band_hpc.loc[start_rip:end_rip])
index = [np.where(x == rip_tsd.index.values)[0][0] for x in rip_tsd.loc[start_rip:end_rip].index.values]
for i in index:
	start3, end3 = rip_ep.loc[i]
	plot(lfp_hpc.loc[start3:end3])

ax2 = subplot(212, sharex = ax1)
count = 0
for n in theta_mod_rem.index:	
	xt = spikes[int(n.split("_")[1])].loc[start_rip:end_rip].index.values
	if len(xt):			
		if n in neurons:			
			plot(xt, np.ones(len(xt))*count, '|', markersize = 5, markeredgewidth = 5)
		else:
			plot(xt, np.ones(len(xt))*count, '|', markersize = 2, markeredgewidth = 2)
		count+=1



#############################################################################################################
# THETA
#############################################################################################################
store 			= pd.HDFStore("../data/phase_spindles/"+session.split("/")[1]+".lfp")
lfp_hpc 		= nts.Tsd(store['lfp_hpc'])
store.close()		

lfp_filt_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 5, 15, fs/5, 2))	
power	 		= nts.Tsd(lfp_filt_hpc.index.values, np.abs(lfp_filt_hpc.values))

phase 			= getPhase(lfp_hpc, 6, 14, 16, fs/5.)	
spikes_phase	= {n:phase.realign(spikes[n], align = 'closest').restrict(theta_ep) for n in spikes.keys()}
sys.exit()
peaks, troughs 	= getPeaksandTroughs(lfp_filt_hpc, 10)
lfp_filt_hpc 	= lfp_filt_hpc.restrict(theta_ep)
lfp_hpc 		= lfp_hpc.restrict(theta_ep)
power 			= power.restrict(theta_ep)
peaks 			= peaks.restrict(theta_ep)
# search for the theta ep for which the phase is the closest of the mean phase
spikes_per_cycle = []
ep_pk = []
for e in theta_ep.index:
	start, stop = theta_ep.loc[e]
	peaks_in_ep = peaks.loc[start:stop]
	for i in range(len(peaks_in_ep)-1):
		t1,t2 = peaks_in_ep.index[[0,1]]
		tmp = []
		ep_pk.append([e,i])
		for n in neurons:
			tmp.append(len(spikes_phase[int(n.split("_")[1])].loc[t1:t2]))
		spikes_per_cycle.append(tmp)
spikes_per_cycle = np.array(spikes_per_cycle)
# plot((spikes_per_cycle > 0).sum(1))
# show()
best = (spikes_per_cycle > 0).sum(1) == 3
ep_pk = np.array(ep_pk)[best]
# ep_to_plot = np.unique(ep_pk[262:283,0])[0]

ep = 57
# for ep in np.unique(ep_pk[:,0]):
print(ep)
# start,end = theta_ep.loc[ep]
start,end = (5843120000,5844075000)

#saveh5.put('lfp_filt_hpc_theta', lfp_filt_hpc.loc[start:end].as_series())
#saveh5.put('lfp_hpc_theta', lfp_hpc.loc[start:end].as_series())
# for n in neurons:
	# #saveh5.put('spike_theta'+n, spikes[int(n.split("_")[1])].loc[start:end].as_series())
	#saveh5.put('phase_spike_theta_'+n, spikes_phase[int(n.split("_")[1])].as_series())
#saveh5.close()




figure()
ax1 = subplot(311)
plot(lfp_filt_hpc.loc[start:end])
plot(lfp_hpc.loc[start:end])

ax2 = subplot(312, sharex = ax1)
count = 0
for n in theta_mod_rem.index:	
	xt = spikes[int(n.split("_")[1])].loc[start:end].index.values
	if len(xt):			
		if n in neurons:			
			plot(xt, np.ones(len(xt))*count, '|', markersize = 5, markeredgewidth = 5)
		else:
			plot(xt, np.ones(len(xt))*count, '|', markersize = 2, markeredgewidth = 2)
		count+=1


for n in neurons:
	subplot(3,3,neurons.index(n)+1+6, projection = 'polar')
	tmp = spikes_phase[int(n.split("_")[1])].values
	tmp += 2*np.pi
	tmp %= 2*np.pi
	hist(tmp,10)
show()