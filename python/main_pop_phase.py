#!/usr/bin/env python
'''
    File name: main_ripp_mod.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2

Sharp-waves ripples modulation 
Used to make figure 1

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

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {}

clients = ipyparallel.Client()	
dview = clients.direct_view()

zpre = pd.DataFrame()
zpos = pd.DataFrame()
hdn = {}

for session in datasets:	
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
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])	

	spikes_sws 		= {n:spikes[n].restrict(sws_ep) for n in spikes.keys() if len(spikes[n].restrict(sws_ep))}

	if len(sleep_ep) > 1:
		pre_ep 			= nts.IntervalSet(sleep_ep['start'][0], sleep_ep['end'][0])
		post_ep 		= nts.IntervalSet(sleep_ep['start'][1], sleep_ep['end'][1])
		if pre_ep.tot_length()/1000/1000/60 > 30.0 and post_ep.tot_length()/1000/1000/60 > 30.0:		
			pre_xmin_ep = pre_ep.copy()
			pre_xmin_ep['start'] = pre_xmin_ep['end'][0] - 60*60*1000*1000
			post_xmin_ep = post_ep.copy()
			post_xmin_ep['end'] = post_xmin_ep['start'][0] + 60*60*1000*1000

			def cross_correlation(tsd):
				spike_tsd, rip_tsd = tsd
				import numpy as np
				from functions import xcrossCorr_fast
				bin_size 	= 5 # ms 
				nb_bins 	= 200
				confInt 	= 0.95
				nb_iter 	= 1000
				jitter  	= 150 # ms					
				H0, Hm, HeI, HeS, Hstd, times = xcrossCorr_fast(rip_tsd, spike_tsd, bin_size, nb_bins, nb_iter, jitter, confInt)				
				return (H0 - Hm)/Hstd, times
			
			spikes_list = [spikes_sws[i].as_units('ms').index.values for i in spikes_sws.keys()]
			
			Hcorr_pre = dview.map_sync(cross_correlation, zip(spikes_list, [rip_tsd.restrict(pre_xmin_ep).as_units('ms').index.values for i in spikes_sws.keys()]))
			Hcorr_pos = dview.map_sync(cross_correlation, zip(spikes_list, [rip_tsd.restrict(post_xmin_ep).as_units('ms').index.values for i in spikes_sws.keys()]))
				
			for n,i in zip(spikes_sws.keys(), range(len(spikes_sws.keys()))):
				zpre = zpre.append(pd.DataFrame(data = np.atleast_2d(Hcorr_pre[i][0]), index = [session.split("/")[1]+"_"+str(n)], columns = Hcorr_pre[i][1]))
				zpos = zpos.append(pd.DataFrame(data = np.atleast_2d(Hcorr_pos[i][0]), index = [session.split("/")[1]+"_"+str(n)], columns = Hcorr_pos[i][1]))								
				hdn[session.split("/")[1]+"_"+str(n)] =  hd_info_neuron[int(n)]

			

import _pickle as cPickle
jpcadata  = cPickle.load(open('../figures/figures_articles/figure2/dict_fig2_article.pickle', 'rb'))
swr_modth = jpcadata['swr_modth']
phi = jpcadata['phi']
rX = jpcadata['rX']



hdn = pd.DataFrame.from_dict(hdn, 'index')
zpre = zpre[~zpre.isnull().any(1)]
zpos = zpos[~zpos.isnull().any(1)]
index = np.intersect1d(zpre.index.values, zpos.index.values)
zpre = zpre.loc[index]
zpos = zpos.loc[index]
index = np.intersect1d(swr_modth.index.values, zpos.index.values)
zpre = zpre.loc[index]
zpos = zpos.loc[index]
swr_modth = swr_modth.loc[index]
hdn = hdn.loc[index]
phi = phi.loc[index]
zpre = zpre[swr_modth.columns]
zpos = zpos[swr_modth.columns]
Rpre = pd.DataFrame(index = index, columns = index, data = np.corrcoef(zpre.values))
Rpos = pd.DataFrame(index = index, columns = index, data = np.corrcoef(zpos.values))
corr = pd.DataFrame(columns = ['pre','pos'])
corr['pre'] = Rpre.values[np.tril_indices(Rpre.shape[0], -1)]
corr['pos'] = Rpos.values[np.tril_indices(Rpos.shape[0], -1)]
scorepre = np.dot(zpre, rX)
scorepos = np.dot(zpos, rX)
phipre = pd.DataFrame(index = zpre.index, data = np.mod(np.arctan2(scorepre[:,1], scorepre[:,0]), 2*np.pi))
phipos = pd.DataFrame(index = zpos.index, data = np.mod(np.arctan2(scorepos[:,1], scorepos[:,0]), 2*np.pi))


from pylab import *
figure()
subplot(121)
plot(corr['pre'], corr['pos'])
subplot(122)
x = phi - phipre
y = phi - phipos
x[(x < -np.pi)] += 2*np.pi
y[(y < -np.pi)] += 2*np.pi
x[(x > np.pi)] -= 2*np.pi
y[(y > np.pi)] -= 2*np.pi
scatter(x[hdn == 0], y[hdn == 0])
scatter(x[hdn == 1], y[hdn == 1])
show()


toplot = pd.DataFrame(columns = ['phi', 'phipre', 'phipos'], data = np.hstack((phi.values, phipre.values, phipos.values)))
store = pd.HDFStore("../figures/figures_articles/figure3/pop_phase_shift.h5")
store.put('corr', corr)
store.put('phi', toplot)
store.close()