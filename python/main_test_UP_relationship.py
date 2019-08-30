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
import _pickle as cPickle



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

datasets = ['Mouse12/Mouse12-120806', 'Mouse12/Mouse12-120807', 'Mouse12/Mouse12-120808', 'Mouse12/Mouse12-120809', 'Mouse12/Mouse12-120810',
'Mouse17/Mouse17-130129', 'Mouse17/Mouse17-130130', 'Mouse32/Mouse32-140822']

allswrrad = []
allswrvel = []

# for session in ['Mouse32/Mouse32-140822']:
for session in datasets:
	print(session)
	generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
	shankStructure 	= loadShankStructure(generalinfo)
	if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
	else:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
	spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
	n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')	
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
	# lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])


	# if np.sum(hd_info_neuron)>2 and np.sum(hd_info_neuron==0)>2:			
	####################################################################################################################
	# binning data
	####################################################################################################################	
	spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
	spikesnohd 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}
	hdneurons		= np.sort(list(spikeshd.keys()))
	nohdneurons		= np.sort(list(spikesnohd.keys()))


	data = cPickle.load(open('../figures/figures_articles_v4/figure1/good_100ms_pickle/'+session.split("/")[1]+'.pickle', 'rb'))

	swrvel 			= []
	swrrad 			= []
	for n in data['swr'].keys():
		iwak		= data['swr'][n]['iwak']
		iswr		= data['swr'][n]['iswr']
		rtsd		= data['swr'][n]['rip_tsd']
		rip_spikes	= data['swr'][n]['rip_spikes']
		times 		= data['swr'][n]['times']
		wakangle	= data['swr'][n]['wakangle']
		neurons		= data['swr'][n]['neurons']
		tcurves		= data['swr'][n]['tcurves']

		normwak = np.sqrt(np.sum(np.power(iwak,2), 1))
		normswr = np.sqrt(np.sum(np.power(iswr, 2), -1))
		normswr = pd.DataFrame(index = times, columns = rtsd.index.values.astype('int'), data = normswr.T)
		swrrad.append(normswr)
		angswr = np.arctan2(iswr[:,:,1], iswr[:,:,0])
		angswr = (angswr + 2*np.pi)%(2*np.pi)		
		angvel = []
		for i in range(len(angswr)):
			a = np.unwrap(angswr[i])
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std=1.0)
			angvel.append(np.abs(np.diff(c.values))/0.1)
		
		angvel = pd.DataFrame(index = times[0:-1]+np.diff(times)/2, columns = rtsd.index.values.astype('int'), data = np.array(angvel).T)
		swrvel.append(angvel)

	swrvel = pd.concat(swrvel, 1)
	swrvel = swrvel.groupby(level=0,axis=1).first()
	swrrad = pd.concat(swrrad, 1)
	swrrad = swrrad.groupby(level=0,axis=1).first()

	# RANDOM
	rndvel 			= []
	rndrad			= []
	for n in data['rnd'].keys():
		irand 		= data['rnd'][n]['irand']
		iwak2 		= data['rnd'][n]['iwak2']
		normrnd = np.sqrt(np.sum(np.power(irand,2), -1))
		rndrad.append(normrnd)
		angrnd = np.arctan2(irand[:,:,1], irand[:,:,0])
		angrnd = (angrnd + 2*np.pi)%(2*np.pi)

		for i in range(len(angrnd)):
			a = np.unwrap(angrnd[i])
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std=1.0)
			rndvel.append(np.abs(np.diff(c.values))/0.1)

	rndvel = np.array(rndvel)
	rndrad = np.vstack(rndrad)

	basvel = pd.Series(index=times[0:-1]+np.diff(times)/2, data = rndvel.mean(0))
	basrad = pd.Series(index=times, data = rndrad.mean(0))


	swrrad = pd.DataFrame(index=swrrad.index, data = ((swrrad.values.T - basrad.values)/basrad.values).T,  columns = swrrad.columns)
	swrvel = pd.DataFrame(index=swrvel.index, data = ((swrvel.values.T - basvel.values)/basvel.values).T,  columns = swrvel.columns)

	

	# UP/DOWN
	down_ep, up_ep = loadUpDown(data_directory+session)
	rip_tsd = rip_tsd.restrict(up_ep)	
	up_tsd = nts.Ts(t = up_ep['start'].values)
	d = np.vstack(rip_tsd.index.values) - up_tsd.index.values
	d[d<0] = np.max(d)
	idx = np.argmin(d, 1)
	up_time = up_tsd.index.values[idx]

	interval = rip_tsd.index.values - up_time

	order = np.argsort(interval)

	a = swrrad[rip_tsd.index.values[order]]

	groups = []
	for idx in np.array_split(np.arange(a.shape[1]),3):
		groups.append(a.iloc[:,idx].mean(1))
	groups = pd.concat(groups, 1)

	a.columns = pd.Index(interval[order])

	b = swrvel[rip_tsd.index.values[order]]

	groups2 = []
	for idx in np.array_split(np.arange(b.shape[1]),3):
		groups2.append(b.iloc[:,idx].mean(1))
	groups2 = pd.concat(groups2, 1)

	b.columns = pd.Index(interval[order])

	allswrrad.append(a)
	allswrvel.append(b)


	# subplot(221)
	# imshow(a.T, aspect = 'auto', cmap = 'jet')
	# xlabel("Time from SWRs")
	# title("Radius")
	# ylabel("Time from Up")
	# subplot(222)
	# imshow(b.T, aspect = 'auto', cmap = 'jet')
	# xlabel("Time from SWRs")
	# title("Angular Velocity")
	# ylabel("Time from Up")	
	# subplot(223)
	# plot(groups)	
	# xlabel("Time from SWRs")
	# ylabel("Radius")
	# subplot(224)
	# plot(groups2)
	# xlabel("Time from SWRs")
	# ylabel("Angular velocity")
	# show()	


allswrrad = pd.concat(allswrrad, 1)
allswrvel = pd.concat(allswrvel, 1)

allswrrad = allswrrad.sort_index(1)
allswrvel = allswrvel.sort_index(1)

groups = []
bounds = []
for idx in np.array_split(np.arange(allswrrad.shape[1]),6):
	groups.append(allswrrad.iloc[:,idx].mean(1))
	bounds.append((str(int(allswrrad.columns[idx].min()/1000)), str(int(allswrrad.columns[idx].max()/1000))))
groups = pd.concat(groups, 1)
groups.columns = pd.Index(bounds)


groups2 = []
bounds = []
for idx in np.array_split(np.arange(allswrvel.shape[1]),6):
	groups2.append(allswrvel.iloc[:,idx].mean(1))
	bounds.append((str(int(allswrvel.columns[idx].min()/1000)), str(int(allswrvel.columns[idx].max()/1000))))
groups2 = pd.concat(groups2, 1)
groups2.columns = pd.Index(bounds)



figure()
subplot(121)
for c in groups.columns:
	plot(groups[c], label = c)
title("Radius")
legend()
subplot(122)
for c in groups2.columns:
	plot(groups2[c], label = c)
title("Angular velocity")
legend()
show()