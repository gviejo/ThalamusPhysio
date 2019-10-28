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


def compute_mua_timing_ratio(spk, evt_start, evt_end):
	tmp1 = np.vstack(spk) - evt_start
	tmp1 = tmp1.T	
	tmp1[tmp1<0] = tmp1.max()
	start_to_spk = np.nanmin(tmp1, 0)	

	tmp2 = np.vstack(evt_end) - spk
	tmp2[tmp2<0] = tmp2.max()
	spk_to_end = np.nanmin(tmp2, 0)

	d = start_to_spk/(start_to_spk+spk_to_end)
	return d


data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

datasets = ['Mouse12/Mouse12-120806', 'Mouse12/Mouse12-120807', 'Mouse12/Mouse12-120808', 'Mouse12/Mouse12-120809', 'Mouse12/Mouse12-120810',
'Mouse17/Mouse17-130130', 'Mouse32/Mouse32-140822']

allmua_hd = {}
allmua_nohd = {}
allswr_rates = {}

dview = Pool(8)

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
	
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])

	spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
	spikesnohd 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}
	hdneurons		= np.sort(list(spikeshd.keys()))
	nohdneurons		= np.sort(list(spikesnohd.keys()))

	# UP/DOWN
	down_ep, up_ep = loadUpDown(data_directory+session)

	bins = np.hstack((np.linspace(0,1,100)-1,np.linspace(0,1,100)[1:]))


	######################################################################################################
	# SWR RATES / UP DOWN
	######################################################################################################
	t1 = time.time()
	uprip = rip_tsd.restrict(up_ep).index.values
	tmp1 = np.vstack(uprip) - up_ep['start'].values
	tmp1 = tmp1.astype(np.float32).T
	tmp1[tmp1<0] = np.nan
	start_to_rip = np.nanmin(tmp1, 0)

	tmp2 = np.vstack(up_ep['end'].values) - uprip
	tmp2 = tmp2.astype(np.float32)
	tmp2[tmp2<0] = np.nan
	rip_to_end = np.nanmin(tmp2, 0)

	d = start_to_rip/(start_to_rip+rip_to_end)
	rip_up = d

	dwrip = rip_tsd.restrict(down_ep).index.values
	tmp1 = np.vstack(dwrip) - down_ep['start'].values
	tmp1 = tmp1.astype(np.float32).T
	tmp1[tmp1<0] = np.nan
	start_to_rip = np.nanmin(tmp1, 0)

	tmp2 = np.vstack(down_ep['end'].values) - dwrip
	tmp2 = tmp2.astype(np.float32)
	tmp2[tmp2<0] = np.nan
	rip_to_end = np.nanmin(tmp2, 0)

	d = start_to_rip/(start_to_rip+rip_to_end)
	rip_down = d

	p, _ = np.histogram(np.hstack((rip_down-1,rip_up)), bins)
	
	swr_rates = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = p)
	swr_rates = swr_rates/swr_rates.sum()

	print("Rip done", time.time()-t1)


	######################################################################################################
	# HD RATES / UP DOWN	
	######################################################################################################
	t2 = time.time()
	mua_hd = []

	for n in spikeshd.keys():
		spk = spikeshd[n].restrict(up_ep).index.values
		spk2 = np.array_split(spk, 10)
	
		# args = [[spk2[i], up_ep['start'].values, up_ep['end'].values] for i in range(len(spk2))]
	
		# result = dview.starmap_async(compute_mua_timing_ratio, args).get()

		# mua_up = np.hstack(result)
		
		# spk = spikeshd[n].restrict(down_ep).index.values
		# spk2 = np.array_split(spk, 8)
	
		# args = [[spk2[i], down_ep['start'].values, down_ep['end'].values] for i in range(len(spk2))]
	
		# result = dview.starmap_async(compute_mua_timing_ratio, args).get()

		# mua_down = np.hstack(result)

		start_to_spk = []
		for i in range(len(spk2)):
			tmp1 = np.vstack(spk2[i]) - up_ep['start'].values
			tmp1 = tmp1.astype(np.float32).T
			tmp1[tmp1<0] = np.nan
			start_to_spk.append(np.nanmin(tmp1, 0))
		start_to_spk = np.hstack(start_to_spk)

		spk_to_end = []
		for i in range(len(spk2)):
			tmp2 = np.vstack(up_ep['end'].values) - spk2[i]
			tmp2 = tmp2.astype(np.float32)
			tmp2[tmp2<0] = np.nan
			spk_to_end.append(np.nanmin(tmp2, 0))
		spk_to_end = np.hstack(spk_to_end)

		d = start_to_spk/(start_to_spk+spk_to_end)
		mua_up = d

		spk = spikeshd[n].restrict(down_ep).index.values
		tmp1 = np.vstack(spk) - down_ep['start'].values
		tmp1 = tmp1.astype(np.float32).T
		tmp1[tmp1<0] = np.nan
		start_to_spk = np.nanmin(tmp1, 0)

		tmp2 = np.vstack(down_ep['end'].values) - spk
		tmp2 = tmp2.astype(np.float32)
		tmp2[tmp2<0] = np.nan
		spk_to_end = np.nanmin(tmp2, 0)

		d = start_to_spk/(start_to_spk+spk_to_end)
		mua_down = d

		p, _ = np.histogram(np.hstack((mua_down-1,mua_up)), bins)

		mua_hd.append(p)

	mua_hd = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = np.array(mua_hd).sum(0))
	mua_hd = mua_hd/mua_hd.sum()


	print("HD done", time.time() - t2)

	######################################################################################################
	# non-HD RATES / UP DOWN
	######################################################################################################
	t3 = time.time()
	mua_nohd = []

	for n in spikesnohd.keys():
		# memory error here so dividing the spikes in 4		
		spk = spikesnohd[n].restrict(up_ep).index.values
		spk2 = np.array_split(spk, 10)

		start_to_spk = []
		for i in range(len(spk2)):
			tmp1 = np.vstack(spk2[i]) - up_ep['start'].values
			tmp1 = tmp1.astype(np.float32).T
			tmp1[tmp1<0] = np.nan
			start_to_spk.append(np.nanmin(tmp1, 0))
		start_to_spk = np.hstack(start_to_spk)

		spk_to_end = []
		for i in range(len(spk2)):
			tmp2 = np.vstack(up_ep['end'].values) - spk2[i]
			tmp2 = tmp2.astype(np.float32)
			tmp2[tmp2<0] = np.nan
			spk_to_end.append(np.nanmin(tmp2, 0))
		spk_to_end = np.hstack(spk_to_end)

		d = start_to_spk/(start_to_spk+spk_to_end)
		mua_up = d

		spk = spikesnohd[n].restrict(down_ep).index.values
		tmp1 = np.vstack(spk) - down_ep['start'].values
		tmp1 = tmp1.astype(np.float32).T
		tmp1[tmp1<0] = np.nan
		start_to_spk = np.nanmin(tmp1, 0)

		tmp2 = np.vstack(down_ep['end'].values) - spk
		tmp2 = tmp2.astype(np.float32)
		tmp2[tmp2<0] = np.nan
		spk_to_end = np.nanmin(tmp2, 0)

		d = start_to_spk/(start_to_spk+spk_to_end)
		mua_down = d

		p, _ = np.histogram(np.hstack((mua_down-1,mua_up)), bins)

		mua_nohd.append(p)

	mua_nohd = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = np.array(mua_nohd).sum(0))
	mua_nohd = mua_nohd/mua_nohd.sum()

	print("non hd done", time.time()-t3)

	# SAVING
	allmua_hd[session] = mua_hd
	allmua_nohd[session] = mua_nohd
	allswr_rates[session] = swr_rates


allmua_hd  	= pd.DataFrame.from_dict(allmua_hd)
allmua_nohd 	= pd.DataFrame.from_dict(allmua_nohd)
allswr_rates 	= pd.DataFrame.from_dict(allswr_rates)

figure()
plot(allmua_hd.mean(1), label = 'hd')
plot(allmua_nohd.mean(1), label = 'non-hd')
plot(allswr_rates.mean(1), label = 'SWRs rate')
legend()
show()

tosave = {'mua_hd':allmua_hd, 'mua_nohd':allmua_nohd, 'swr_rates':allswr_rates}

cPickle.dump(tosave, open('../figures/figures_articles_v4/figure1/UP_DOWN_RATES.pickle', 'wb'))