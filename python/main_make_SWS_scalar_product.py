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
import _pickle as cPickle

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
	
	bin_size = 40
	n_ex = 2000

	rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, np.maximum(1,n_ex//len(sws_ep))) for i in sws_ep.index])))

	####################################################################################################################
	# MEAN AND STD SWS
	####################################################################################################################
	# # mean and standard deviation during SWS
	# mean_sws = pd.DataFrame(index = np.sort(list(spikes.keys())), columns = ['min', 'max'])
	# for n in spikes.keys():
	# 	r = []
	# 	for e in sws_ep.index:
	# 		bins = np.arange(sws_ep.loc[e,'start'], sws_ep.loc[e,'end'], bin_size*1e3)
	# 		a, _ = np.histogram(spikes[n].restrict(sws_ep.loc[[e]]).index.values, bins)
	# 		r.append(a)
	# 	r = np.hstack(r)
	# 	r = r / (bin_size*1e-3)
	# 	mean_sws.loc[n,'min']= r.min()
	# 	mean_sws.loc[n,'max']= r.max()
		
	bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2
	times = bins[0:-1] + np.diff(bins)/2
	
	####################################################################################################################
	# HD NEURONS
	####################################################################################################################	
	if len(spikeshd) >=5:
		ts = rip_tsd.as_units('ms').index.values
		rates = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)
		# # rates = rates /float(bin_size*1e-3)
		# angle = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))
		# for i, r in enumerate(rates):
		# 	tmp = scalarProduct(r)
		# 	angle[i] = tmp

		# random				
		ts = rnd_tsd.as_units('ms').index.values
		rates2 = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)	
		# # rates2 = rates2/float(bin_size*1e-3)
		# shuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))	
		# for i, r in enumerate(rates2):
		# 	tmp = scalarProduct(r)
		# 	shuffled[i] = tmp

		# anglehd[session] = (angle.mean(1) - shuffled.mean(1))/shuffled.mean(1)

		# normalized
		zangle = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))
		min_ = rates.min(0).min(0)
		max_ = rates.max(0).max(0)
		zrates = (rates - min_) / (max_ - min_)
		for i, r in enumerate(zrates):			
			tmp = scalarProduct(r)
			zangle[i] = tmp

		# random
		zshuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))
		min_ = rates2.min(0).min(0)
		max_ = rates2.max(0).max(0)
		zrates2 = (rates2 - min_) / (max_ - min_)
		for i, r in enumerate(zrates2):
			tmp = scalarProduct(r)
			zshuffled[i] = tmp
		
		
		zanglehd[session] = (zangle.mean(1) - zshuffled.mean(1))/zshuffled.mean(1)
		anglehd[session] = zangle #.fillna(0)
		
	####################################################################################################################
	# NO HD NEURONS
	####################################################################################################################	
	if len(spikesnohd) >=5:
		ts = rip_tsd.as_units('ms').index.values
		rates = quickBin([spikesnohd[j].as_units('ms').index.values for j in nohdneurons], ts, bins, nohdneurons)	
		# # rates = rates/float(bin_size*1e-3)
		# angle = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))		
		# for i, r in enumerate(rates):			
		# 	angle[i] = scalarProduct(r)

		# random		
		ts = rnd_tsd.as_units('ms').index.values
		rates2 = quickBin([spikesnohd[j].as_units('ms').index.values for j in nohdneurons], ts, bins, nohdneurons)			
		# # rates2 = rates2/float(bin_size*1e-3)
		
		# shuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))		
		# for i, r in enumerate(rates2):
		# 	shuffled[i] = scalarProduct(r)

		# anglenohd[session] = (angle.mean(1) - shuffled.mean(1))/shuffled.mean(1)

		# normalized				
		zangle = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))		
		min_ = rates.min(0).min(0)
		max_ = rates.max(0).max(0)
		zrates = (rates - min_) / (max_ - min_)
		for i, r in enumerate(zrates):
			zangle[i] = scalarProduct(r)

		# random
		zshuffled = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rnd_tsd)))		
		# zrates2 = (rates2 - m) / (s+1)
		min_ = rates2.min(0).min(0)
		max_ = rates2.max(0).max(0)
		zrates2 = (rates2 - min_) / (max_ - min_)
		for i, r in enumerate(zrates2):
			zshuffled[i] = scalarProduct(r)
		
		zanglenohd[session] = (zangle.mean(1) - zshuffled.mean(1))/zshuffled.mean(1)
		anglenohd[session] = zangle #.fillna(0)


# anglehd = pd.DataFrame.from_dict(anglehd)
# anglenohd = pd.DataFrame.from_dict(anglenohd)

# anglehd = anglehd.rolling(window=10,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
# anglenohd = anglenohd.rolling(window=10,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)

zanglehd = pd.DataFrame.from_dict(zanglehd)
zanglenohd = pd.DataFrame.from_dict(zanglenohd)

zanglehd = zanglehd.rolling(window=10,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
zanglenohd = zanglenohd.rolling(window=10,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)


# subplot(211)
# plot(anglehd.mean(1), label = 'hd')
# plot(anglenohd.mean(1), label = 'no hd')
# legend()
# title("Scalar product")
# subplot(212)
figure()
plot(zanglehd.mean(1))
plot(zanglenohd.mean(1))
legend()
title("Scalar product + norm [0 1]")

# comparing with isomap radius
path = '../figures/figures_articles_v4/figure1/'
files = [f for f in os.listdir(path) if '.pickle' in f and 'Mouse' in f]
files.remove("Mouse17-130129.pickle")
radius = []
velocity = []
stability = []
order = []

for f in files:
	data = cPickle.load(open(path+f, 'rb'))
	swrvel 			= []
	swrrad 			= []
	for n in data['swr'].keys():
		iswr		= data['swr'][n]['iswr']
		rip_tsd		= data['swr'][n]['rip_tsd']
		times 		= data['swr'][n]['times']	

		normswr = np.sqrt(np.sum(np.power(iswr, 2), -1))
		normswr = pd.DataFrame(index = times, columns = rip_tsd.index.values.astype('int'), data = normswr.T)
		swrrad.append(normswr)

		angswr = np.arctan2(iswr[:,:,1], iswr[:,:,0])
		angswr = (angswr + 2*np.pi)%(2*np.pi)
		tmp = []
		for i in range(len(angswr)):
			a = np.unwrap(angswr[i])
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std=1.0)
			tmp.append(np.abs(np.diff(c.values))/0.1)		
		tmp = pd.DataFrame(index = times[0:-1] + np.diff(times)/2, columns = rip_tsd.index.values.astype('int'), data = np.array(tmp).T)
		swrvel.append(tmp)

	swrvel = pd.concat(swrvel, 1)
	swrrad = pd.concat(swrrad, 1)
	swrvel = swrvel.sort_index(1)
	swrrad = swrrad.sort_index(1)

	s = f.split('-')[0]+'/'+ f.split('.')[0]
	stab = anglehd[s]
	# cutting between -500 to 500
	stab = stab.loc[-500:500]
	
	# aligning swrrad.index to stab.index
	newswrrad = []
	for i in swrrad.columns:
		y = swrrad[i].values
		if len(y.shape) ==2 : 
			print("Bug in ", f)
			y = y[:,0]
		fi = scipy.interpolate.interp1d(swrrad.index.values, y)
		newswrrad.append(fi(stab.index.values))
	newswrrad = pd.DataFrame(index = stab.index.values, columns = swrrad.columns, data = np.array(newswrrad).T)

	newswrvel = []
	for i in swrvel.columns:
		y = swrvel[i].values
		if len(y.shape) ==2 :
			y = y[:,0]
		fi = scipy.interpolate.interp1d(swrvel.index.values, y)
		newswrvel.append(fi(stab.index.values))
	newswrvel = pd.DataFrame(index = stab.index.values, columns = swrvel.columns, data = np.array(newswrvel).T)

	radius.append(newswrrad.mean(1))
	stability.append(stab.mean(1))
	velocity.append(newswrvel.mean(1))
	order.append(f)

radius = pd.concat(radius, 1)
stability = pd.concat(stability, 1)
velocity = pd.concat(velocity, 1)
velocity = 

stability = stability.apply(scipy.stats.zscore)
radius = radius.apply(scipy.stats.zscore)
velocity = velocity.apply(scipy.stats.zscore)


figure()
subplot(231)
for i in radius.columns:
	plot(radius[i])
title("Radius")
subplot(232)
for i in velocity.columns:
	plot(velocity[i])
title("Ang velocity")
subplot(233)
for i in stability.columns:
	plot(stability[i])
title("Stability")
subplot(234)
for i in radius.columns:
	scatter(radius[i], stability[i])
	xlabel("Radius")
	ylabel("Stability")
subplot(235)
for i in radius.columns:
	scatter(velocity[i], stability[i])
	xlabel("velocity")
	ylabel("Stability")



tosave = {'velocity':velocity,
			'radius':radius}




show()

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

