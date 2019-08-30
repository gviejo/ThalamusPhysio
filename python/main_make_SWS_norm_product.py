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
from numba import jit

@jit(nopython=True)
def scalarProduct(r):
	tmp = np.sqrt(np.power(r, 2).sum(1))
	denom = tmp[0:-1] * tmp[1:]
	num = np.sum(r[0:-1]*r[1:], 1)
	return num/(denom)


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

anglehd = {}
anglenohd = {}
zanglehd = {}
zanglenohd = {}


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
	
	bin_size = 20
	n_ex = 1000
	normed = True
	####################################################################################################################
	# MIN MAX SWS
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
		mean_sws.loc[n,'min']= r.min()
		mean_sws.loc[n,'max']= r.max()
		
	bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2
	times = bins[0:-1] + np.diff(bins)/2

	ts = rip_tsd.as_units('ms').index.values

	####################################################################################################################
	# HD NEURONS
	####################################################################################################################	
	if len(spikeshd) >=5:
		rates = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)
		rates = rates/float(bin_size*1e-3)

		angle = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))
		for i, r in enumerate(rates):
			tmp = scalarProduct(r)
			angle[i] = np.nan_to_num(tmp, 0)

		zangle = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))				
		a = mean_sws.loc[hdneurons,'min'].values.astype('float')
		b = mean_sws.loc[hdneurons,'max'].values.astype('float')		
		zrates = (rates - a) / (b-a)
		for i, r in enumerate(zrates):			
			zangle[i] = scalarProduct(r)


		# random		
		rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, n_ex//len(sws_ep)) for i in sws_ep.index])))
		ts = rnd_tsd.as_units('ms').index.values
		rates2 = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)	
		rates2 = rates2/float(bin_size*1e-3)


		shuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))	
		for i, r in enumerate(rates2):
			tmp = scalarProduct(r)
			shuffled[i] = np.nan_to_num(tmp, 0)

		zshuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))
		a = mean_sws.loc[hdneurons,'min'].values.astype('float')
		b = mean_sws.loc[hdneurons,'max'].values.astype('float')
		zrates2 = (rates2 - a) / (b-a)
		for i, r in enumerate(zrates2):
			zshuffled[i] = scalarProduct(r)			 

		
		anglehd[session] = (angle.mean(1) - shuffled.mean(1))/shuffled.mean(1)
		zanglehd[session] = (zangle.mean(1) - zshuffled.mean(1))/zshuffled.mean(1)
		
		
	####################################################################################################################
	# NO HD NEURONS
	####################################################################################################################	
	if len(spikesnohd) >=5:
		rates = quickBin([spikesnohd[j].as_units('ms').index.values for j in nohdneurons], ts, bins, nohdneurons)	
		rates = rates/float(bin_size*1e-3)

		angle = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))		
		for i, r in enumerate(rates):			
			tmp = scalarProduct(r)
			angle[i] = np.nan_to_num(tmp, 0)
				
		zangle = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))		
		a = mean_sws.loc[nohdneurons,'min'].values.astype('float')
		b = mean_sws.loc[nohdneurons,'max'].values.astype('float')		
		zrates = (rates - a) / (b-a)
		for i, r in enumerate(zrates):
			zangle[i] = scalarProduct(r)

		# random		
		rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, n_ex//len(sws_ep)) for i in sws_ep.index])))
		ts = rnd_tsd.as_units('ms').index.values
		rates2 = quickBin([spikesnohd[j].as_units('ms').index.values for j in nohdneurons], ts, bins, nohdneurons)			
		rates2 = rates2/float(bin_size*1e-3)		
		
		
		shuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))		
		for i, r in enumerate(rates2):
			tmp = scalarProduct(r)
			shuffled[i] = np.nan_to_num(tmp, 0)

		zshuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))		
		a = mean_sws.loc[nohdneurons,'min'].values.astype('float')
		b = mean_sws.loc[nohdneurons,'max'].values.astype('float')
		zrates2 = (rates2 - a) / (b-a)
		for i, r in enumerate(zrates2):
			zshuffled[i] = scalarProduct(r)

		anglenohd[session] = (angle.mean(1) - shuffled.mean(1))/shuffled.mean(1)
		zanglenohd[session] = (zangle.mean(1) - zshuffled.mean(1))/zshuffled.mean(1)



anglehd = pd.DataFrame.from_dict(anglehd)
anglenohd = pd.DataFrame.from_dict(anglenohd)

zanglehd = pd.DataFrame.from_dict(zanglehd)
zanglenohd = pd.DataFrame.from_dict(zanglenohd)

subplot(211)
plot(anglehd.mean(1))
plot(anglenohd.mean(1))
title("Scalar product")
subplot(212)
plot(zanglehd.mean(1))
plot(zanglenohd.mean(1))
title("Scalar product + zscored")

sys.exit()


store = pd.HDFStore('../figures/figures_articles_v4/figure2/test.h5')
if normed:
	store.append('anglehd_normed', anglehd)
	store.append('anglenohd_normed', anglenohd)
else:
	store.append('anglehd', anglehd)
	store.append('anglenohd', anglenohd)
store.close()


figure()
store = pd.HDFStore('../figures/figures_articles_v4/figure2/test.h5')
subplot(2,2,1)
plot(store['anglehd'].mean(1), label = 'HD')
plot(store['anglenohd'].mean(1), label = 'non-HD')
legend()
title("Scalar Product")

subplot(2,2,2)
plot(store['pearsonhd'].mean(1), label = 'HD')
plot(store['pearsonnohd'].mean(1), label = 'non-HD')
legend()
title("Pearson Correlation")

subplot(2,2,3)
plot(store['anglehd_normed'].mean(1), label = 'HD')
plot(store['anglenohd_normed'].mean(1), label = 'non-HD')
legend()
title("Scalar Product normalized")

subplot(2,2,4)
plot(store['pearsonhd_normed'].mean(1), label = 'HD')
plot(store['pearsonnohd_normed'].mean(1), label = 'non-HD')
legend()
title("Pearson Correlation normalized")

show()

sys.exit()


anglehd = pd.DataFrame.from_dict(anglehd)
anglenohd = pd.DataFrame.from_dict(anglenohd)


plot(anglehd.mean(1), label = 'hd')
plot(anglenohd.mean(1), label = 'nohd')
legend()
show()


sys.exit()

datatosave = cPickle.load(open("/mnt/DataGuillaume/MergedData/SWR_SCALAR_PRODUCT.pickle", 'rb'))

angleall = datatosave['cosalpha']
baselineall = datatosave['baseline']


hd = pd.DataFrame()

for s in angleall.keys():
	if 'hd' in list(angleall[s].keys()):
		tmp1 = angleall[s]['hd'].rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=0.5)
		tmp2 = baselineall[s]['hd'].rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=0.5)
		tmp = (tmp1.mean(1) - tmp2.mean(1))/tmp2.mean(1)
		hd[s.split("/")[1]] = tmp		


nohd = pd.DataFrame()

for s in angleall.keys():
	if 'nohd' in list(angleall[s].keys()):
		tmp1 = angleall[s]['nohd'].rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
		tmp2 = baselineall[s]['nohd'].rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
		tmp = (tmp1.mean(1) - tmp2.mean(1))/tmp2.mean(1)
		nohd[s.split("/")[1]] = tmp		




data = pd.DataFrame(index = hd.index.values, columns = pd.MultiIndex.from_product([['hd', 'nohd'], ['mean', 'sem']]))
data['hd', 'mean'] = hd.mean(1)
data['hd', 'sem'] = hd.sem(1)
data['nohd', 'mean'] = nohd.mean(1)
data['nohd', 'sem'] = nohd.sem(1)

data.to_hdf("../figures/figures_articles_v4/figure2/SWR_SCALAR_PRODUCT.h5", 'w')


subplot(111)
m = hd.mean(1)
v = hd.sem(1)
plot(hd.mean(1), label = 'hd')
fill_between(hd.index.values, m+v, m-v, alpha = 0.5)
# title("Only hd")
# subplot(212)
# title("No hd")
m = nohd.mean(1)
v = nohd.sem(1)
plot(nohd.mean(1), label = 'nohd')
fill_between(nohd.index.values, m+v, m-v, alpha = 0.5)
legend()

figure()
subplot(121)
plot(hd, color = 'grey')
plot(hd.mean(1), color = 'red')
title("HD")
subplot(122)
plot(nohd, color = 'grey')
plot(nohd.mean(1), color = 'black')
title("No HD")

show()

