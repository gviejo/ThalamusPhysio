#!/usr/bin/env python
'''
    File name: main_ripp_mod.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2


'''
import sys
import numpy as np
import pandas as pd
import scipy.io
from functions import *
# from pylab import *
from multiprocessing import Pool
import os
import neuroseries as nts
from time import time
from pylab import *
from functions import quickBin
from sklearn.utils.random import sample_without_replacement
from numba import jit

# clients = ipyparallel.Client()	
# dview = clients.direct_view()
dview = Pool()

def sample_comb(dims, nsamp):
    idx = sample_without_replacement(np.prod(dims), nsamp)
    return np.vstack(np.unravel_index(idx, dims)).T

@jit(nopython=True)
def quickBin(spikelist, ts, bins, index):
	rates = np.zeros((len(ts), len(bins)-1, len(index)))
	for i, t in enumerate(ts):
		tbins = t + bins
		for j in range(len(spikelist)):					
			a, _ = np.histogram(spikelist[j], tbins)
			rates[i,:,j] = a
	return rates


data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

pearsonhd = {}
pearsonnohd = {}
zpearsonhd = {}
zpearsonnohd = {}

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
	rip_ep,rip_tsd 	= loadRipples(data_directory+session)
	rip_ep			= sws_ep.intersect(rip_ep)	
	rip_tsd 		= rip_tsd.restrict(sws_ep)	
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])

	####################################################################################################################
	# binning data
	####################################################################################################################	
	spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
	spikesnohd 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}
	hdneurons		= np.sort(list(spikeshd.keys()))
	nohdneurons		= np.sort(list(spikesnohd.keys()))	

	bin_size = 50
	n_ex = 1000
	normed = True
	

	####################################################################################################################
	# MEAN AND STD SWS
	####################################################################################################################
	# mean and standard deviation during SWS
	mean_sws = pd.DataFrame(index = np.sort(list(spikes.keys())), columns = ['mean', 'std'])
	for n in spikes.keys():
		r = []
		for e in sws_ep.index:
			bins = np.arange(sws_ep.loc[e,'start'], sws_ep.loc[e,'end'], bin_size*1e3)
			a, _ = np.histogram(spikes[n].restrict(sws_ep.loc[[e]]).index.values, bins)
			r.append(a)
		r = np.hstack(r)
		r = r / (bin_size*1e-3)
		mean_sws.loc[n,'mean']= r.mean()
		mean_sws.loc[n,'std']= r.std()
		
	bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2
	times = bins[0:-1] + np.diff(bins)/2

	ts = rip_tsd.as_units('ms').index.values

	####################################################################################################################
	# HD NEURONS
	####################################################################################################################	
	if len(spikeshd) >=5:
		rates = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)
		rates = rates/float(bin_size*1e-3)
		m = mean_sws.loc[hdneurons,'mean'].values.astype('float')
		s = mean_sws.loc[hdneurons,'std'].values.astype('float')
		pearson = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))
		for i, r in enumerate(rates):			
			tmp = np.diag(np.corrcoef(r), 1)
			pearson[i] = np.nan_to_num(tmp)

		zpearson = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))				
		zrates = (rates - m) / (s+1)
		for i, r in enumerate(zrates):			
			zpearson[i] = np.diag(np.corrcoef(r), 1)


		# random		
		rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, n_ex//len(sws_ep)) for i in sws_ep.index])))
		ts = rnd_tsd.as_units('ms').index.values
		rates2 = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)	
		
		rates2 = rates2/float(bin_size*1e-3)
		m = mean_sws.loc[hdneurons,'mean'].values.astype('float')
		s = mean_sws.loc[hdneurons,'std'].values.astype('float')

		shuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))	
		for i, r in enumerate(rates2):
			tmp = np.diag(np.corrcoef(r), 1)
			shuffled[i] = np.nan_to_num(tmp)

		zshuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))
		zrates2 = (rates2 - m) / (s+1)
		for i, r in enumerate(zrates2):
			zshuffled[i] = np.diag(np.corrcoef(r), 1)

		
		pearsonhd[session] = (pearson.mean(1) - shuffled.mean(1))/shuffled.mean(1)
		zpearsonhd[session] = (zpearson.mean(1) - zshuffled.mean(1))/zshuffled.mean(1)
		
		

	####################################################################################################################
	# NO HD NEURONS
	####################################################################################################################	
	if len(spikesnohd) >=5:
		rates = quickBin([spikesnohd[j].as_units('ms').index.values for j in nohdneurons], ts, bins, nohdneurons)	
		rates = rates/float(bin_size*1e-3)
		m = mean_sws.loc[nohdneurons,'mean'].values.astype('float')
		s = mean_sws.loc[nohdneurons,'std'].values.astype('float')
		pearson = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))		
		for i, r in enumerate(rates):			
			tmp = np.diag(np.corrcoef(r), 1)
			pearson[i] = np.nan_to_num(tmp)
				
		zpearson = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))		
		zrates = (rates - m) / (s+1)
		for i, r in enumerate(zrates):
			zpearson[i] = np.diag(np.corrcoef(r), 1)

		# random		
		rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, n_ex//len(sws_ep)) for i in sws_ep.index])))
		ts = rnd_tsd.as_units('ms').index.values
		rates2 = quickBin([spikesnohd[j].as_units('ms').index.values for j in nohdneurons], ts, bins, nohdneurons)			
		rates2 = rates2/float(bin_size*1e-3)
		m = mean_sws.loc[nohdneurons,'mean'].values.astype('float')
		s = mean_sws.loc[nohdneurons,'std'].values.astype('float')
		
		shuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))		
		for i, r in enumerate(rates2):
			tmp = np.diag(np.corrcoef(r), 1)
			shuffled[i] = np.nan_to_num(tmp)

		zshuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))		
		zrates2 = (rates2 - m) / (s+1)
		for i, r in enumerate(zrates2):
			zshuffled[i] = np.diag(np.corrcoef(r), 1)

		pearsonnohd[session] = (pearson.mean(1) - shuffled.mean(1))/shuffled.mean(1)
		zpearsonnohd[session] = (zpearson.mean(1) - zshuffled.mean(1))/zshuffled.mean(1)





pearsonhd = pd.DataFrame.from_dict(pearsonhd)
pearsonnohd = pd.DataFrame.from_dict(pearsonnohd)
zpearsonhd = pd.DataFrame.from_dict(zpearsonhd)
zpearsonnohd = pd.DataFrame.from_dict(zpearsonnohd)


subplot(211)
plot(pearsonhd.mean(1))
plot(pearsonnohd.mean(1))
title("Pearson correlation")
subplot(212)
plot(zpearsonhd.mean(1))
plot(zpearsonnohd.mean(1))
title("Pearson correlation + zscored")

sys.exit()

store = pd.HDFStore('../figures/figures_articles_v4/figure2/test.h5')
if normed:
	store.append('pearsonhd_normed', pearsonhd)
	store.append('pearsonnohd_normed', pearsonnohd)
else:
	store.append('pearsonhd', pearsonhd)
	store.append('pearsonnohd', pearsonnohd)

store.close()

