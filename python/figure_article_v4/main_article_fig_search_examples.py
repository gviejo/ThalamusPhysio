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
path_snippet 	= "../../figures/figures_articles_v4/figure1/"
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
session 		= 'Mouse12/Mouse12-120810'
# session 		= 'Mouse12/Mouse12-120809'
# session 		= [s for s in datasets if 'Mouse12' in s][4]
neurons 		= [session.split("/")[1]+"_"+str(u) for u in [23, 19, 40]]

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
# speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)	
# speed_ep 		= nts.IntervalSet(speed[speed>2.5].index.values[0:-1], speed[speed>2.5].index.values[1:]).drop_long_intervals(26000).merge_close_intervals(50000)
# wake_ep 		= wake_ep.intersect(speed_ep).drop_short_intervals(3000000)
# to match main_make_SWRinfo.py
spikes 			= {n:spikes[n] for n in spikes.keys() if len(spikes[n].restrict(sws_ep))}
n_neuron 		= len(spikes)
allneurons 		= [session.split("/")[1]+"_"+str(list(spikes.keys())[i]) for i in spikes.keys()]
hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])
n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')
rip_ep,rip_tsd 	= loadRipples(data_directory+session)

swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
# filtering swr_mod
# swr 					= pd.DataFrame(	columns = swr_ses, 
# 										index = times,
# 										data = gaussFilt(swr_mod, (5,)).transpose())
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = swr_mod.T)

swr = swr.loc[-500:500]


#############################################################################################################
# PLOTTING SWR MODULATION 
#############################################################################################################

best_neurons = np.array(allneurons)[np.where(hd_info_neuron==1)[0]]

H0 = []
Hm = []
Hstd = []
bin_size 	= 5 # ms 
nb_bins 	= 200
confInt 	= 0.95
nb_iter 	= 1000
jitter  	= 150 # ms					
for n in best_neurons:
	a, b, c, d, e, f = 	xcrossCorr_fast(rip_tsd.as_units('ms').index.values, spikes[int(n.split("_")[1])].as_units('ms').index.values, bin_size, nb_bins, nb_iter, jitter, confInt)
	H0.append(a)
	Hm.append(b)
	Hstd.append(e)

H0 		= pd.DataFrame(index = f, data = np.array(H0).transpose(), columns = best_neurons)
Hm 		= pd.DataFrame(index = f, data = np.array(Hm).transpose(), columns = best_neurons)
Hstd 	= pd.DataFrame(data = np.array(Hstd), index = best_neurons)


figure(figsize = (10,100))

for i, n in enumerate(best_neurons):
	subplot(len(best_neurons),1,i+1)
	plot(H0[n])
	title(n)
	legend()

savefig("search_examples.pdf", dpi = 900, facecolor = 'white')
# os.system("evince search_examples.pdf &")



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
# TO SAVE
#############################################################################################################
path_snippet 	= "../../figures/figures_articles_v4/figure1/"
saveh5 			= pd.HDFStore(path_snippet+'snippet_'+session.split("/")[1]+'.h5')



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

# sys.exit()

saveh5.put('H0', H0)
saveh5.put('Hm', Hm)
saveh5.put('Hstd', Hstd)
# saveh5.put('lfp_filt_hpc_swr', lfp_filt_rip_band_hpc.loc[start_rip:end_rip].as_series())
# saveh5.put('lfp_hpc_swr', lfp_hpc.loc[start_rip:end_rip].as_series())
# for n in neurons:
# 	saveh5.put('spike_swr'+n, spikes[int(n.split("_")[1])].loc[start_rip:end_rip].as_series())
# index = [np.where(x == rip_tsd.index.values)[0][0] for x in rip_tsd.loc[start_rip:end_rip].index.values]
# saveh5.put('swr_ep', pd.DataFrame(rip_ep.loc[index]))

saveh5.close()



# figure()
# ax1 = subplot(211)
# plot(lfp_hpc.loc[start_rip:end_rip])
# plot(lfp_filt_rip_band_hpc.loc[start_rip:end_rip])
# index = [np.where(x == rip_tsd.index.values)[0][0] for x in rip_tsd.loc[start_rip:end_rip].index.values]
# for i in index:
# 	start3, end3 = rip_ep.loc[i]
# 	plot(lfp_hpc.loc[start3:end3])

# ax2 = subplot(212, sharex = ax1)
# count = 0
# for n in theta_mod_rem.index:	
# 	xt = spikes[int(n.split("_")[1])].loc[start_rip:end_rip].index.values
# 	if len(xt):			
# 		if n in neurons:			
# 			plot(xt, np.ones(len(xt))*count, '|', markersize = 5, markeredgewidth = 5)
# 		else:
# 			plot(xt, np.ones(len(xt))*count, '|', markersize = 2, markeredgewidth = 2)
# 		count+=1


