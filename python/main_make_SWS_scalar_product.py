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

# clients = ipyparallel.Client()	
# dview = clients.direct_view()
dview = Pool()

def sample_comb(dims, nsamp):
    idx = sample_without_replacement(np.prod(dims), nsamp)
    return np.vstack(np.unravel_index(idx, dims)).T



data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
angleall = {}
baselineall = {}

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

	####################################################################################################################
	# BIN SWR
	####################################################################################################################
	bin_size = 20
	bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2
	# dspike = {i:{j:spk[j].as_units('ms').index.values for j in spk} for i, spk in zip(['all', 'hd', 'nohd'], [spikes, spikeshd, spikesnohd])}
	ts = rip_tsd.as_units('ms').index.values
	
	order = []
	args = []
	if len(spikeshd) >=5:
		order.append('hd')
		args.append(({j:spikeshd[j].as_units('ms').index.values for j in spikeshd}, ts, bins))
	if len(spikesnohd) >=5:
		order.append('nohd')
		args.append(({j:spikesnohd[j].as_units('ms').index.values for j in spikesnohd}, ts, bins))

	if len(args):
		allrates = dview.starmap(quickBin, args)
		allrates = {n:allrates[i] for i, n in enumerate(order)}
		times = bins[0:-1] + np.diff(bins)/2
		angle = {n:pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd))) for n in order}

		for n in allrates.keys():
			for i, r in enumerate(allrates[n]):			
				tmp = np.sqrt(np.power(r+1, 2).sum(1))
				denom = tmp[0:-1] * tmp[1:]
				num = np.sum(r[0:-1]*r[1:], 1)
				angle[n][i] = num/denom

		angleall[session] = angle

	sys.exit()

	####################################################################################################################
	# BIN SWS
	####################################################################################################################
	if len(args):
		baseline = {}
		allbins = np.arange(sws_ep.as_units('ms').iloc[0,0], sws_ep.as_units('ms').iloc[-1,1], bin_size)
		for case, (dspike, _, _ ) in zip(order, args):
			rates = []
			for n in dspike:
				rates.append(np.histogram(dspike[n], allbins)[0])
			rates = nts.TsdFrame(t = allbins[0:-1]+bin_size/2, d = np.vstack(rates).T/(bin_size*1e-3), time_units = 'ms')
			rates = rates.restrict(sws_ep)
			r = rates.values
			tmp = np.sqrt(np.power(r+1, 2).sum(1))
			a, b = sample_comb((len(r),len(r)), 1000000).T
			denom = tmp[a] * tmp[b]
			num = np.sum(r[a,:] * r[b,:], 1)
			ba = num/denom
			baseline[case] = np.array([ba.mean(), ba.var(), ba.std(), scipy.stats.sem(ba)])
		baselineall[session] = baseline


datatosave = {'baseline':baselineall,
				'cosalpha':angleall}


import _pickle as cPickle
cPickle.dump(datatosave, open("/mnt/DataGuillaume/MergedData/SWR_SCALAR_PRODUCT.pickle", 'wb'))



hd = []

for s in angleall.keys():
	if 'hd' in list(angleall[s].keys()):
		tmp = angleall[s]['hd'] - baselineall[s]['hd'][0]
		tmp = tmp / baselineall[s]['hd'][2]
		hd.append(tmp)

hd = pd.concat(hd, 1)

nohd = []

for s in angleall.keys():
	if 'nohd' in list(angleall[s].keys()):
		tmp = angleall[s]['nohd'] - baselineall[s]['nohd'][0]
		tmp = tmp / baselineall[s]['nohd'][2]
		nohd.append(tmp)		

nohd = pd.concat(nohd, 1)


data = pd.DataFrame(index = hd.index.values, columns = pd.MultiIndex.from_product([['hd', 'nohd'], ['mean', 'sem']]))
data['hd', 'mean'] = hd.mean(1)
data['hd', 'sem'] = hd.sem(1)
data['nohd', 'mean'] = nohd.mean(1)
data['nohd', 'sem'] = nohd.sem(1)

data.to_hdf("../figures/figures_articles_v4/figure1/SWR_SCALAR_PRODUCT.h5", 'w')


subplot(211)
m = hd.mean(1)
v = hd.sem(1)
plot(hd.mean(1))
fill_between(hd.index.values, m+v, m-v, alpha = 0.5)
title("Only hd")
subplot(212)
title("No hd")
m = nohd.mean(1)
v = nohd.sem(1)
plot(nohd.mean(1))
fill_between(nohd.index.values, m+v, m-v, alpha = 0.5)

show()

