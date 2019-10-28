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



data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

datasets = ['Mouse12/Mouse12-120806', 'Mouse12/Mouse12-120807', 'Mouse12/Mouse12-120808', 'Mouse12/Mouse12-120809', 'Mouse12/Mouse12-120810',
'Mouse17/Mouse17-130130', 'Mouse32/Mouse32-140822']

allswrrad = []
allswrvel = []

bins = [10,196,374,672,2000]

groups_radius = {n:[] for n in range(1,len(bins))}
groups_velocity = {n:[] for n in range(1,len(bins))}

allradius = []
allvelocity = []

session_velocity = {}
session_radius = {}

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

	spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
	spikesnohd 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}
	hdneurons		= np.sort(list(spikeshd.keys()))
	nohdneurons		= np.sort(list(spikesnohd.keys()))

	
	# data = cPickle.load(open('../figures/figures_articles_v4/figure1/good_100ms_pickle/'+session.split("/")[1]+'.pickle', 'rb'))
	data = cPickle.load(open('../figures/figures_articles_v4/figure1/hd_isomap_30ms_mixed_swr_rnd_wake_good/'+session.split("/")[1]+'.pickle', 'rb'))

	###########################################################################
	# UP/DOWN
	###########################################################################
	down_ep, up_ep = loadUpDown(data_directory+session)
	rip_tsd = rip_tsd.restrict(up_ep)	
	up_tsd = nts.Ts(t = up_ep['start'].values)
	d = np.vstack(rip_tsd.index.values) - up_tsd.index.values
	d[d<0] = np.max(d)
	idx = np.argmin(d, 1)
	up_time = up_tsd.index.values[idx]
	interval = rip_tsd.index.values - up_time
	interval = pd.Series(index = rip_tsd.index, data = interval)
	
	sess_rad = []
	sess_vel = []

	###########################################################################
	# LOADING IMAP
	###########################################################################
	times 			= data['times']
	for n in data['imaps'].keys():		
		iswr		= data['imaps'][n]['swr']
		irnd 		= data['imaps'][n]['rnd']
		subrip_tsd 	= data['imaps'][n]['subrip_tsd']
		
		normswr = np.sqrt(np.sum(np.power(iswr, 2), -1))
		normrnd = np.sqrt(np.sum(np.power(irnd,2), -1))

		basrad = normrnd.mean(0)
		
		angswr = np.arctan2(iswr[:,:,1], iswr[:,:,0])
		angswr = (angswr + 2*np.pi)%(2*np.pi)
		angrnd = np.arctan2(irnd[:,:,1], irnd[:,:,0])
		angrnd = (angrnd + 2*np.pi)%(2*np.pi)

		tmp1 = []
		for i in range(len(angswr)):
			a = np.unwrap(angswr[i])
			b = pd.Series(index = times, data = a)
			c = b
			c = b.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=1)
			tmp1.append(np.abs(np.diff(c.values))/0.02)

		tmp2 = []
		for i in range(len(angrnd)):
			a = np.unwrap(angrnd[i])
			b = pd.Series(index = times, data = a)
			c = b
			c = b.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=1)
			tmp2.append(np.abs(np.diff(c.values))/0.02)
	
		tmp1 = np.array(tmp1)
		tmp2 = np.array(tmp2)
		
		basvel = tmp2.mean(0)

		index = np.digitize(interval.loc[subrip_tsd]/1000, bins)

		for i in range(1, len(bins)):
			swrrad = pd.Series(index = times, data = (normswr[index == i].mean(0) - basrad)/basrad)
			groups_radius[i].append(swrrad)
			swrvel = pd.Series(index = times[0:-1]+np.diff(times)/2, data = (tmp1[index == i].mean(0) - basvel)/basvel)			
			groups_velocity[i].append(swrvel)

			tmp = pd.DataFrame(index = times, columns = interval.loc[subrip_tsd].values, data = ((normswr-basrad)/basrad).T)
			allradius.append(tmp)

			tmp = pd.DataFrame(index = times[0:-1]+np.diff(times)/2, columns = interval.loc[subrip_tsd].values, data = ((tmp1-basvel)/basvel).T)
			allvelocity.append(tmp)

		sess_rad.append((normswr[(interval.loc[subrip_tsd] > 100000).values].mean(0)-basrad)/basrad)
		sess_vel.append((tmp1[(interval.loc[subrip_tsd] > 100000).values].mean(0)-basvel)/basvel)
	
	session_radius[session] = pd.Series(index = times, data = np.array(sess_rad).mean(0))
	session_velocity[session] = pd.Series(index = times[0:-1]+np.diff(times)/2, data = np.array(sess_vel).mean(0))


session_velocity = pd.DataFrame.from_dict(session_velocity)
session_radius = pd.DataFrame.from_dict(session_radius)


for n in range(1, len(bins)):
	groups_radius[n] = pd.concat(groups_radius[n], 1)
	groups_radius[n] = groups_radius[n].rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=1)
	groups_velocity[n] = pd.concat(groups_velocity[n], 1)
	groups_velocity[n] = groups_velocity[n].rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=1)


allradius = pd.concat(allradius, 1)
allvelocity = pd.concat(allvelocity, 1)

allradius = allradius.sort_index(1)
allvelocity = allvelocity.sort_index(1)


figure()
subplot(121)
for n in groups_radius.keys():
	plot(groups_radius[n].mean(1))
subplot(122)
for n in groups_velocity.keys():
	plot(groups_velocity[n].mean(1))
show()



tosave = {"radius":groups_radius,"velocity":groups_velocity, 'all':{'allradius':allradius,'allvelocity':allvelocity}}


cPickle.dump(tosave, open('../figures/figures_articles_v4/figure1/UP_ONSET_SWR.pickle', 'wb'))

