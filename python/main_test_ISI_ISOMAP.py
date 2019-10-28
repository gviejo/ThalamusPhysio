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
from scipy.signal import resample_poly

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

theta2 = pd.read_hdf("/mnt/DataGuillaume/MergedData/THETA_THAL_mod_2.h5")
 
for session in datasets:
	index = [n for n in theta2.index.values if session.split("/")[1] in n]
	ratio_theta = np.sum(theta2.loc[index,('wak', 'pval')]<0.001)/len(index)
	if ratio_theta > 0.75:
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
		# hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		# hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])
				
		theta_neuron = np.array(index)[(theta2.loc[index,('wak', 'pval')]<0.001).values]

		spikes 		= {k:spikes[int(k.split("_")[1])] for k in theta_neuron}
		neurons 		= np.sort(list(spikes.keys()))

		theta_ep 		= np.genfromtxt(data_directory+session+"/"+session.split("/")[1]+".wake.evt.theta")[:,0]
		theta_ep 		= theta_ep.reshape(len(theta_ep)//2,2)
		theta_ep 		= nts.IntervalSet(theta_ep[:,0], theta_ep[:,1], time_units = 'ms')

		print(session, len(neurons))

		# WAKE
		bin_size_wake = 2
		bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size_wake, bin_size_wake)		
		data = []
		for i in neurons:
			spks = spikes[i].restrict(wake_ep).as_units('ms').index.values
			idx = np.digitize(spks, bins)-1
			isi = np.zeros(len(bins)-1)
			for i in range(1, len(spks)):
				isi[idx[i-1]:idx[i]] = spks[i] - spks[i-1]
			isi = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = 1/(isi+1))
			
			# data.append(isi)
			


			new_isi = isi.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=10)

			# new_isi = scipy.signal.resample_poly(new_isi.values, 1, 10)
			# tmp = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size_wake, bin_size_wake*10)		
			# new_isi = pd.Series(index = tmp[0:-1] + np.diff(tmp)/2, data = new_isi)				
			data.append(new_isi)

		data = pd.concat(data,1)

		position = pd.read_csv(data_directory+session+"/"+session.split("/")[1] + ".csv", delimiter = ',', header = None, index_col = [0])
		angle = nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')		
		wakangle = pd.Series(index = np.arange(len(bins)-1))
		tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
		wakangle.loc[tmp.index] = tmp
		wakangle.index = data.index
		wakangle = wakangle.interpolate(method='nearest')

		data = nts.TsdFrame(t = data.index.values, d = data.values, time_units = 'ms')

		abins = np.linspace(0, 2*np.pi, 61)

		wakangle = wakangle.dropna()
		data = data.loc[wakangle.index]

		index = np.digitize(wakangle.values, abins)-1

		a = data.groupby(index).mean()


		data = data.restrict(theta_ep)



		imap = Isomap(n_neighbors = 100, n_components = 2).fit_transform(data.values[0:20000])

		scatter(imap[:,0], imap[:,1])
		show()