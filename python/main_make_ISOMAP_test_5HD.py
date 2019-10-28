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
# import ipyparallel
from multiprocessing import Pool
import os
import neuroseries as nts
from time import time
from pylab import *
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import _pickle as cPickle

####################################################################################################################
# FUNCTIONS
####################################################################################################################
@jit(nopython=True)
def histo(spk, obins):
	n = len(obins)
	count = np.zeros(n)
	for i in range(n):
		count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
	return count


data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

data = {}

for session in datasets:
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]	
	if np.sum(hd_info)>4:
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
		speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)
		hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])
		
		spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
		neurons 		= np.sort(list(spikeshd.keys()))

		print(session, len(neurons))

		# lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
		# tmp = [lfp_hpc.loc[t-1e6:t+1e6] for i, t in enumerate(rip_tsd.index.values)]
		# tmp = pd.concat(tmp, 0)
		# tmp = tmp[~tmp.index.duplicated(keep='first')]		
		# tmp.as_series().to_hdf(data_directory+session+'/'+session.split("/")[1]+'_EEG_SWR.h5', 'swr')
		
		# lfp_hpc 		= pd.read_hdf(data_directory+session+'/'+session.split("/")[1]+'_EEG_SWR.h5')

		####################################################################################################################
		# HEAD DIRECTION INFO
		####################################################################################################################
		spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
		neurons 		= np.sort(list(spikeshd.keys()))
		position 		= pd.read_csv(data_directory+session+"/"+session.split("/")[1] + ".csv", delimiter = ',', header = None, index_col = [0])
		angle 			= nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')
		tcurves 		= computeAngularTuningCurves(spikeshd, angle, wake_ep, nb_bins = 60, frequency = 1/0.0256)
		neurons 		= tcurves.idxmax().sort_values().index.values


		####################################################################################################################
		# binning data
		####################################################################################################################
		allrates		= {}
		bins_size = [300,10,50]

		####################################################################################################################
		# BIN WAKE
		####################################################################################################################
		# make sure that wake_ep < 30 min
		# if 'Mouse32' in session:
		wake_ep = wake_ep.intersect(nts.IntervalSet(start=wake_ep.loc[0,'start'], end = wake_ep.loc[0,'start']+30*60*1e6))
		# else:
		# 	wake_ep = wake_ep.intersect(nts.IntervalSet(start=wake_ep.loc[0,'start'], end = wake_ep.loc[0,'start']+20*60*1e6))
		bin_size = bins_size[0]
		bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
		spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
		for i in neurons:
			spks = spikeshd[i].as_units('ms').index.values
			spike_counts[i], _ = np.histogram(spks, bins)

		# allrates['wak'] = np.sqrt(spike_counts/(bins_size[0]*1e-3))
		rates_wak = np.sqrt(spike_counts/(bins_size[0]))

		wakangle = pd.Series(index = np.arange(len(bins)-1))
		idx = np.digitize(angle.restrict(wake_ep).as_units('ms').index.values, bins)-1
		tmp = angle.restrict(wake_ep).groupby(idx).mean()
		wakangle.loc[tmp.index] = tmp
		wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

		####################################################################################################################
		# BIN SWR
		####################################################################################################################
		n_ex = 5
		n_rip = len(rip_tsd)
		n_loop = n_rip//n_ex
		idx = np.random.randint(0, n_loop, n_rip)
		bin_size = bins_size[2]	
		left_bound = np.arange(-500-bin_size/2, 500 - bin_size/4,bin_size/4)
		obins = np.vstack((left_bound, left_bound+bin_size)).T
		times = obins[:,0]+(np.diff(obins)/2).flatten()
		

		datatosave = {'swr':{},'rnd':{}}

		# for i in range(n_loop):
		for i in range(1):
			good_ex = []

			tmp = rip_tsd.index.values[idx == i]
			subrip_tsd = pd.Series(index = np.hstack((good_ex, tmp)), data = np.nan)			
			rates_swr = []			
			tmp2 = subrip_tsd.index.values/1e3
			
			rip_spikes = {}			

			for j, t in enumerate(tmp2):				
				tbins = t + obins
				spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)
				rip_spikes[j] = {}
				for k in neurons:
					spks = spikeshd[k].as_units('ms').index.values
					spike_counts[k] = histo(spks, tbins)
					nspks = spks - t
					rip_spikes[j][k] = nspks[np.logical_and((spks-t)>=-500, (spks-t)<=500)]
				rates_swr.append(np.sqrt(spike_counts/(bins_size[-1])))

			
			####################################################################################################################
			# ISOMAP SWR SET
			####################################################################################################################
			# tmp1 = rates_wak.rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2).values
			tmp1 = rates_wak.rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2).values
			tmp3 = []
			for rates in rates_swr:
				# tmp3.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=2).values)
				tmp3.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=4).values)
			tmp3 = np.vstack(tmp3)

			n = len(tmp1)
			tmp = np.vstack((tmp1, tmp3))
			imap = Isomap(n_neighbors = 100, n_components = 2, n_jobs = -1).fit_transform(tmp)

			iwak = imap[0:n]
			iswr = imap[n:]
			iswr = iswr.reshape(len(subrip_tsd),len(times),2)

			datatosave['swr'][i] = {
					"iwak"		: iwak,
					"iswr"		: iswr,
					"rip_tsd"	: subrip_tsd,
					"rip_spikes": rip_spikes,
					"times" 	: times,
					"wakangle"	: wakangle,
					"neurons"	: neurons,
					"tcurves"	: tcurves,
					# "iwak2"		: iwak2,
					# "irand"		: irand
			}
			
		data[session] = datatosave['swr'][i]
		# ####################################################################################################################
		# # BIN RANDOM
		# ####################################################################################################################
		# n_sample = 10000
		# n_loop = n_sample//n_ex

		# for i in range(n_loop):
		# 	print("random", i)
		# 	tmp = []
		# 	rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[j,'start']+500000, sws_ep.loc[j,'end']+500000, np.maximum(1,n_ex//len(sws_ep))) for j in sws_ep.index])))
		# 	tmp3 = rnd_tsd.index.values/1000

		# 	for j, t in enumerate(tmp3):				
		# 		tbins = t + obins
		# 		spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)	
		# 		for k in neurons:
		# 			spks = spikeshd[k].as_units('ms').index.values	
		# 			spike_counts[k] = histo(spks, tbins)
		# 			nspks = spks - t
				
		# 		tmp.append(np.sqrt(spike_counts/(bins_size[-1])))
				
		# 	rates_rnd = tmp

		# 	####################################################################################################################
		# 	# ISOMAP RANDOM SET
		# 	####################################################################################################################
		# 	# tmp1 = rates_wak.rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2).values
		# 	tmp1 = rates_wak.rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=4).values
		# 	tmp2 = []
		# 	for rates in rates_rnd:
		# 		# tmp2.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=2).values)
		# 		tmp2.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=4).values)
		# 	tmp2 = np.vstack(tmp2)

		# 	tmp = np.vstack((tmp1, tmp2))

		# 	imap2 = Isomap(n_neighbors = 100, n_components = 2, n_jobs = -1).fit_transform(tmp)
		# 	n = len(tmp1)
		# 	iwak2 = imap2[0:n]
		# 	irand = imap2[n:]
		# 	irand = irand.reshape(len(rnd_tsd), len(times),2)

		# 	datatosave['rnd'][i] = {
		# 			"iwak2"		: iwak2,
		# 			"irand"		: irand
		# 	}

		# ####################################################################################################################
		# # SAVING
		# ####################################################################################################################

		# cPickle.dump(datatosave, open('../figures/figures_articles_v4/figure1/'+session.split("/")[1]+'.pickle', 'wb'))

datasets = ['Mouse12/Mouse12-120806', 'Mouse12/Mouse12-120807', 'Mouse12/Mouse12-120808', 'Mouse12/Mouse12-120809', 'Mouse12/Mouse12-120810',
'Mouse17/Mouse17-130130', 'Mouse32/Mouse32-140822']


figure()
for i, s in enumerate(list(data.keys())):
	subplot(3,5,i+1)
	cl = 'blue'
	if s in datasets:
		cl = 'red'
	scatter(data[s]['iwak'][:,0], data[s]['iwak'][:,1], s = 1, c = cl)
	title(s)
show()
