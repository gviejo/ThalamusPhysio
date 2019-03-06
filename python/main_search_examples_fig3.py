import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
import sys
sys.path.append("../")
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import matplotlib.cm as cm
import os
import matplotlib.gridspec as gridspec


###############################################################################################################
# TO LOAD
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

# WHICH NEURONS
space = pd.read_hdf("../figures/figures_articles_v2/figure1/space.hdf5")
burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
burst = burst.loc[space.index]

hd_index = space.index.values[space['hd'] == 1]

neurontoplot = [np.intersect1d(hd_index, space.index.values[space['cluster'] == 1])[0],
				burst.loc[space.index.values[space['cluster'] == 0]].sort_values('sws').index[3],
				burst.sort_values('sws').index.values[-20]]

firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
fr_index = firing_rate.index.values[((firing_rate >= 1.0).sum(1) == 3).values]

# SWR MODULATION
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
swr = swr.loc[-500:500]

# AUTOCORR FAST
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")
autocorr_wak = store_autocorr['wake'].loc[0.5:]
autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
autocorr_sws = 	store_autocorr['sws'].loc[0.5:]
autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_wak = autocorr_wak[2:20]
autocorr_rem = autocorr_rem[2:20]
autocorr_sws = autocorr_sws[2:20]


neurons = np.intersect1d(swr.dropna(1).columns.values, autocorr_sws.dropna(1).columns.values)
neurons = np.intersect1d(neurons, fr_index)

X = np.copy(swr[neurons].values.T)
Y = np.copy(np.vstack((autocorr_wak[neurons].values,autocorr_rem[neurons].values, autocorr_sws[neurons].values))).T
Y = Y - Y.mean(1)[:,np.newaxis]
Y = Y / Y.std(1)[:,np.newaxis]	
pca_swr = PCA(n_components=10).fit(X)
pca_aut = PCA(n_components=10).fit(Y)
pc_swr = pca_swr.transform(X)
pc_aut = pca_aut.transform(Y)


m = 'Mouse17'
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
mappings = mappings.loc[neurons]
mappings = mappings[mappings.index.str.contains(m)]

neurons_ad = mappings.index[np.where(mappings['nucleus'] == 'AD')]
neurons_am = mappings.index[np.where(mappings['nucleus'] == 'AVd')]
groups_ad = mappings.loc[neurons_ad].groupby(by=['session','shank']).groups
groups_am = mappings.loc[neurons_am].groupby(by=['session','shank']).groups

pc_aut = pd.DataFrame(index = neurons, data = pc_aut)
pc_swr = pd.DataFrame(index = neurons, data = pc_swr)

info_ad = pd.DataFrame(index = list(groups_ad.keys()), columns=pd.MultiIndex.from_product((('aut', 'swr'), ('mean', 'var'))))
for k in groups_ad.keys():
	if len(groups_ad[k]) > 3:
		corr_aut = np.zeros((len(groups_ad[k]),len(groups_ad[k])))
		corr_swr = np.zeros((len(groups_ad[k]),len(groups_ad[k])))
		for i,n in enumerate(groups_ad[k]):
			for j,m in enumerate(groups_ad[k]):
				corr_aut[i,j] = scipy.stats.pearsonr(pc_aut.loc[n].values, pc_aut.loc[m].values)[0]
				corr_swr[i,j] = scipy.stats.pearsonr(pc_swr.loc[n].values, pc_swr.loc[m].values)[0]
		distance = np.vstack((corr_aut[np.triu_indices_from(corr_aut, 1)], corr_swr[np.triu_indices_from(corr_swr, 1)])).T
		
		info_ad.loc[k,('aut','mean')] = np.mean(distance[:,0])
		info_ad.loc[k,('swr','mean')] = np.mean(distance[:,1])
		info_ad.loc[k,('aut','var')] = np.var(distance[:,0])
		info_ad.loc[k,('swr','var')] = np.var(distance[:,1])



info_am = pd.DataFrame(index = list(groups_am.keys()), columns=pd.MultiIndex.from_product((('aut', 'swr'), ('mean', 'var'))))
for k in groups_am.keys():
	if len(groups_am[k]) > 3:
		corr_aut = np.zeros((len(groups_am[k]),len(groups_am[k])))
		corr_swr = np.zeros((len(groups_am[k]),len(groups_am[k])))
		for i,n in enumerate(groups_am[k]):
			for j,m in enumerate(groups_am[k]):
				corr_aut[i,j] = scipy.stats.pearsonr(pc_aut.loc[n].values, pc_aut.loc[m].values)[0]
				corr_swr[i,j] = scipy.stats.pearsonr(pc_swr.loc[n].values, pc_swr.loc[m].values)[0]
		distance = np.vstack((corr_aut[np.triu_indices_from(corr_aut, 1)], corr_swr[np.triu_indices_from(corr_swr, 1)])).T
		
		info_am.loc[k,('aut','mean')] = np.mean(distance[:,0])
		info_am.loc[k,('swr','mean')] = np.mean(distance[:,1])
		info_am.loc[k,('aut','var')] = np.var(distance[:,0])
		info_am.loc[k,('swr','var')] = np.var(distance[:,1])


info_ad = info_ad.dropna()
info_am = info_am.dropna()
info_ad = info_ad.sort_values(('swr', 'var'))
info_am = info_am.sort_values(('swr', 'var'))

neurons_ad = groups_ad[info_ad.index[0]]


# neurons_am = groups_am[(17,2)]
# neurons_am = groups_am[list(groups_am.keys())[8]] # good one
neurons_am = groups_am[list(groups_am.keys())[8]]
# neurons_am = groups_am[(11,3)]

# Sorting by channel position
# neurons_am = mappings.loc[neurons_am, 'channel'].sort_values().index.values
corr_aut_am = pd.DataFrame(index = neurons_am, columns = np.arange(len(neurons_am)-1))
corr_swr_am = pd.DataFrame(index = neurons_am, columns = np.arange(len(neurons_am)-1))

for n in neurons_am:
	neurons_am2 = list(np.copy(neurons_am))
	neurons_am2.remove(n)
	for i, m in enumerate(neurons_am2):
		corr_aut_am.loc[n,i] = scipy.stats.pearsonr(pc_aut.loc[m].values, pc_aut.loc[n].values)[0]
		corr_swr_am.loc[n,i] = scipy.stats.pearsonr(pc_swr.loc[m].values, pc_swr.loc[n].values)[0]

corr_swr_aut = pd.Series(index = neurons_am)
for n in neurons_am:
	corr_swr_aut[n] = scipy.stats.pearsonr(corr_aut_am.loc[n].values, corr_swr_am.loc[n].values)[0]

neuron_seed = corr_swr_aut.index[6]
neurons_am2 = list(np.copy(neurons_am))
neurons_am2.remove(neuron_seed)

neurons_am2 = np.array(neurons_am2)[corr_aut_am.loc[neuron_seed].sort_values().index.values]

# neuron_seed = neurons_am[0]
# neurons_am2 = neurons_am[1:]

corr_am2 = pd.Series(index = neurons_am2)
for n in neurons_am2:
	corr_am2[n] = scipy.stats.pearsonr(pc_aut.loc[n].values, pc_aut.loc[neuron_seed].values)[0]
corr_am2 = corr_am2.sort_values()[::-1]

figure()
gs = gridspec.GridSpec(2,len(corr_am2))
for i,n in enumerate(corr_am2.index):
	pair = [neuron_seed, n]
	subplot(gs[0,i])
	tmp = pd.concat([autocorr_wak[pair], autocorr_rem[pair], autocorr_sws[pair]])
	plot(tmp.values)
	title(i)
	subplot(gs[1,i])
	plot(swr[pair])
show()



neurons_to_plot = [corr_am2.index[i] for i in [0,4,8]]

figure()
gs = gridspec.GridSpec(2, 3)
for i, n in enumerate(neurons_to_plot):
	pair = [neuron_seed, n]
	subplot(gs[0,i])
	tmp = pd.concat([autocorr_wak[pair], autocorr_rem[pair], autocorr_sws[pair]])
	plot(tmp.values)
	title(i)
	subplot(gs[1,i])
	plot(swr[pair])
show()


