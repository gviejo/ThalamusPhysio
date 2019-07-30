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
from numba import jit
from spectrum import *


@jit(nopython=True)
def crossCorr(t1, t2, binsize, nbins):
	''' 
		Fast crossCorr 
	'''
	nt1 = len(t1)
	nt2 = len(t2)
	if np.floor(nbins/2)*2 == nbins:
		nbins = nbins+1

	m = -binsize*((nbins+1)/2)
	B = np.zeros(nbins)
	for j in range(nbins):
		B[j] = m+j*binsize

	w = ((nbins/2) * binsize)
	C = np.zeros(nbins)
	i2 = 1

	for i1 in range(nt1):
		lbound = t1[i1] - w
		while i2 < nt2 and t2[i2] < lbound:
			i2 = i2+1
		while i2 > 1 and t2[i2-1] > lbound:
			i2 = i2-1

		rbound = lbound
		l = i2
		for j in range(nbins):
			k = 0
			rbound = rbound+binsize
			while l < nt2 and t2[l] < rbound:
				l = l+1
				k = k+1

			C[j] += k

	# for j in range(nbins):
	# C[j] = C[j] / (nt1 * binsize)
	C = C/(nt1 * binsize/1000)

	return C

def compute_AutoCorrs(spks, ep, binsize = 5, nbins = 200):
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	autocorrs = pd.DataFrame(index = times, columns = np.arange(len(spks)))
	firing_rates = pd.Series(index = np.arange(len(spks)))

	for i in spks:
		spk_time = spks[i].restrict(ep).as_units('ms').index.values
		autocorrs[i] = crossCorr(spk_time, spk_time, binsize, nbins)
		firing_rates[i] = len(spk_time)/ep.tot_length('s')

	# autocorrs = autocorrs / firing_rates
	autocorrs.loc[0] = 0.0
	return autocorrs, firing_rates




start_init = time()
data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')


datatosave = {ep:[] for ep in ['wak', 'rem', 'sws']}

# clients = ipyparallel.Client()	
# dview = clients.direct_view()
# dview = Pool(8)

# firing_rate = pd.DataFrame(columns = ['wake', 'rem', 'sws'])


for session in datasets:	
		start = time()
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
		Hcorr_ep 		= {}
		for ep,k in zip([wake_ep, rem_ep, sws_ep], ['wak', 'rem', 'sws']):			
			AUT, FR = compute_AutoCorrs(spikes, ep, binsize = 0.5, nbins = 20000)
			AUT.columns = pd.Index([session.split("/")[1]+"_"+str(n) for n in spikes.keys()])
			datatosave[k].append(AUT)
			
		print(session, time() - start)
	

for e in datatosave.keys():
	datatosave[e] = pd.concat(datatosave[e], 1)


store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_FOR_FOURIER.h5", 'w')
store_autocorr.put('wak', datatosave['wak'])
store_autocorr.put('rem', datatosave['rem'])
store_autocorr.put('sws', datatosave['sws'])
store_autocorr.close()

from pychronux import *


store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_FOR_FOURIER.h5", 'r')
fte = {}
for e in ['wak', 'rem', 'sws']:	
	autocorr = store_autocorr[e]
	autocorr = autocorr.dropna(1, 'any')
	autocorr = autocorr[autocorr.columns[autocorr.sum(0) != 0]]
	ft = []
	for i, n in enumerate(autocorr.columns):
		S = mtspectrumc(autocorr.loc[:,n].values, 2000, [1, 1000], [10, 19])
		ft.append(S)	
	ft = pd.concat(ft, 1)
	ft.columns = autocorr.columns
	fte[e] = ft

store_fourier = pd.HDFStore("/mnt/DataGuillaume/MergedData/FOURIER_OF_AUTOCORR.h5", 'w')
store_fourier.put('wak', fte['wak'])
store_fourier.put('rem', fte['rem'])
store_fourier.put('sws', fte['sws'])
store_fourier.close()







