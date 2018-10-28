#!/usr/bin/env python


import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
import sys, os
sys.path.append("../")
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import neuroseries as nts


data_directory 	= '/mnt/DataGuillaume/MergedData/'
path_snippet 	= "../../figures/figures_articles/figure2/"
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
session 		= 'Mouse12/Mouse12-120810'
# session 		= 'Mouse12/Mouse12-120809'
# session 		= [s for s in datasets if 'Mouse12' in s][4]
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
rip_ep,rip_tsd 	= loadRipples(data_directory+session)

swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
# filtering swr_mod
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
swr = swr.loc[-500:500]
#############################################################################################################
# THETA EPISODE
#############################################################################################################
theta_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".rem.evt.theta")[:,0]
theta_ep 		= theta_ep.reshape(len(theta_ep)//2,2)
theta_ep 		= nts.IntervalSet(theta_ep[:,0], theta_ep[:,1], time_units = 'ms')

#############################################################################################################
# PLOTTING SWR MODULATION 
#############################################################################################################
figure(figsize = (10,100))
best_neurons = theta_mod_rem.index[(theta_mod_rem['kappa']>np.median(theta_mod_rem['kappa'])).values]
for i, n in enumerate(best_neurons):
	subplot(len(best_neurons),1,i+1)
	plot(swr[n], label = str(theta_mod_rem.loc[n,'phase']))
	ylim(swr[best_neurons].min().min(), swr[best_neurons].max().max())
	title(n)
	legend()

savefig("search_examples.pdf", dpi = 900, facecolor = 'white')
os.system("evince search_examples.pdf &")

sys.exit()
#############################################################################################################
# SWR
#############################################################################################################
spikes_in_swr = {}
for n in neurons:
	sp = spikes[int(n.split("_")[1])]
	ts_for_n = pd.DataFrame()
	count = 0	
	for e in rip_tsd.index.values:
		if count == 1000: break
		sp_in_e = sp.loc[e-500000:e+500000]
		tmp = e - sp_in_e.index.values
		if len(tmp):
			for s in tmp:
				ts_for_n.loc[s/1000,count] = count
			count += 1
	spikes_in_swr[n] = ts_for_n.sort_index()


store = pd.HDFStore(path_snippet+'spikes_in_swr_'+session.split("/")[1]+'.h5', 'a')
for n in neurons:
	store[n] = spikes_in_swr[n]
store.close()


#############################################################################################################
# THETA
#############################################################################################################
store 			= pd.HDFStore("/mnt/DataGuillaume/phase_spindles/"+session.split("/")[1]+".lfp")
lfp_hpc 		= nts.Tsd(store['lfp_hpc'])
store.close()		
phase 			= getPhase(lfp_hpc, 6, 14, 16, fs/5.)
tmp = phase.values.copy()
tmp += 2*np.pi
tmp %= 2*np.pi
phase = nts.Tsd(pd.Series(index = phase.index, data = tmp))	
spikes_phase	= {n:phase.realign(spikes[n], align = 'closest').restrict(theta_ep) for n in spikes.keys()}
phase 			= phase.restrict(theta_ep)
lfp_filt_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 6, 14, fs/5, 2))	
lfp_filt_hpc 	= lfp_filt_hpc.restrict(theta_ep)
lfp_hpc 		= lfp_hpc.restrict(theta_ep)
peaks, troughs 	= getPeaksandTroughs(lfp_filt_hpc, 10)


start_phase = []
end_phase = []
for e in theta_ep.index:
	start, stop = theta_ep.loc[e]
	phase_in_ep = phase.loc[start:stop]
	pk, tr = getPeaksandTroughs(phase_in_ep, 10)
	if tr.index[0] < pk.index[0] and tr.index[-1] < pk.index[-1]:
		start_phase.append(tr.index.values)
		end_phase.append(pk.index.values)
	elif tr.index[0] < pk.index[0] and tr.index[-1] > pk.index[-1]:
		start_phase.append(tr.index.values[0:-1])
		end_phase.append(pk.index.values)
	elif tr.index[0] > pk.index[0] and tr.index[-1] < pk.index[-1]:
		start_phase.append(tr.index.values)
		end_phase.append(pk.index.values[1:])
	elif tr.index[0] > pk.index[0] and tr.index[-1] > pk.index[-1]:
		start_phase.append(tr.index.values[:-1])
		end_phase.append(pk.index.values[1:])
	if len(start_phase[-1]) != len(end_phase[-1]): sys.exit()

start_phase = np.hstack(start_phase)
end_phase = np.hstack(end_phase)
phase_ep = nts.IntervalSet(start = np.hstack(start_phase), end = np.hstack(end_phase))

phase_neurons = {}

for n in neurons:
	phase_for_n = pd.DataFrame(columns = np.arange(len(phase_ep)))
	sp = spikes_phase[int(n.split("_")[1])]
	for i in range(len(phase_ep)):
		tmp = sp.restrict(phase_ep.iloc[i:i+1])
		if len(tmp):
			for j in tmp:
				phase_for_n.loc[j,i] = i

	phase_neurons[n] = phase_for_n.copy()

# removing empty columns
for n in neurons:
	phase_neurons[n] = phase_neurons[n][np.where(~phase_neurons[n].isnull().all())[0]]

store = pd.HDFStore(path_snippet+'spikes_'+session.split("/")[1]+'.h5', 'a')
for n in neurons:
	store[n] = phase_neurons[n]
store.close()



sys.exit()


ax = subplot(211)
plot(lfp_hpc)
plot(lfp_filt_hpc)
# plot(peaks, 'o')
# plot(troughs, 'o')
subplot(212, sharex = ax)
plot(phase)
plot(phase_in_ep, 'o')



figure()
for n in neurons:
	subplot(1,3,neurons.index(n)+1, projection = 'polar')
	tmp = spikes_phase[int(n.split("_")[1])].values
	tmp += 2*np.pi
	tmp %= 2*np.pi
	hist(tmp,10)



figure()
ax1 = subplot(211)
# plot(lfp_hpc)
plot(lfp_filt_hpc)
for n in neurons:
	plot(lfp_filt_hpc.realign(spikes[int(n.split("_")[-1])]), '|', ms = 10, markeredgewidth = 10)


ax2 = subplot(212, sharex = ax1)
ct = 0
for n in neurons:
	i = int(n.split("_")[-1])
	plot(spikes[i].index.values, np.ones(len(spikes[i]))+ct, '|')
	ct+=1

show()





power	 		= nts.Tsd(lfp_filt_hpc.index.values, np.abs(lfp_filt_hpc.values))
power 			= power.restrict(theta_ep)
peaks 			= peaks.restrict(theta_ep)

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



# neurons = ['Mouse17-130130_27', 'Mouse17-130130_3', 'Mouse17-130130_8']
# neurons = ['Mouse12-120809_42', 'Mouse12-120809_39', 'Mouse12-120809_41']
# neurons = ['Mouse12-120810_37', 'Mouse12-120810_40', 'Mouse12-120810_43']
#############################################################################################################
# SWR EPISODE
#############################################################################################################
lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
lfp_filt_rip_band_hpc	= nts.Tsd(lfp_hpc.index.values, butter_bandpass_filter(lfp_hpc, 100, 300, fs, 2))	

rip_ep,rip_tsd 	= loadRipples(data_directory+session)
spike_rip_count = np.zeros((len(neurons),len(rip_ep)))
for e in rip_ep.index:
	start, stop = rip_ep.loc[e]
	for n in neurons:
		spike_rip_count[neurons.index(n),e] = len(spikes[int(n.split("_")[1])].loc[start:stop])

spike_rip_count = pd.DataFrame(index = rip_tsd.index, data = spike_rip_count.transpose(), columns = neurons)


#############################################################################################################
# SPIKES + SWR LFP
#############################################################################################################
figure()
for i, n in enumerate(neurons):
	subplot(3,1,i+1)
	plot(swr[n], label = str(i))
	legend()	

figure()
ax1 = subplot(311)
plot(lfp_filt_rip_band_hpc)
[axvline(t+500000, color = 'blue') for t in rip_tsd.index.values]
[axvline(t-500000, color = 'green') for t in rip_tsd.index.values]
[axvline(t , color = 'red') for t in rip_tsd.index.values]

ax2 = subplot(312, sharex = ax1)
ct = 0
for n in neurons:
	i = int(n.split("_")[-1])
	plot(spikes[i].index.values, np.ones(len(spikes[i]))+ct, '|')
	ct+=1

ax3 = subplot(313, sharex = ax1)
plot(spike_rip_count)

show()
sys.exit()


best_rip_ep = nts.IntervalSet(start = [1.13281e+10, 1.13399e+10, 1.13501e+10, 1.13812e+10],
								end = [1.13305e+10, 1.13411e+10, 1.13511e+10, 1.13819e+10])







#############################################################################################################
# TO SAVE
#############################################################################################################
path_snippet 	= "../../figures/figures_articles/figure2/"
#saveh5 			= pd.HDFStore(path_snippet+'snippet_'+session.split("/")[1]+'.h5')



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

sys.exit()

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


