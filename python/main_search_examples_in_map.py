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
from itertools import combinations


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

firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
fr_index = firing_rate.index.values[((firing_rate >= 2.0).sum(1) == 3).values]

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

m = 'Mouse17'
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
mappings = mappings.loc[neurons]
mappings = mappings[mappings.index.str.contains(m)]

# only non-hd neurons
mappings = mappings[mappings['hd'] == 0]

neurons = mappings.index.values


X = np.copy(swr[neurons].values.T)
Y = np.copy(np.vstack((autocorr_wak[neurons].values,autocorr_rem[neurons].values, autocorr_sws[neurons].values))).T
Y = Y - Y.mean(1)[:,np.newaxis]
Y = Y / Y.std(1)[:,np.newaxis]	
pca_swr = PCA(n_components=10).fit(X)
pca_aut = PCA(n_components=10).fit(Y)
pc_swr = pca_swr.transform(X)
pc_aut = pca_aut.transform(Y)
pc_aut = pd.DataFrame(index = neurons, data = pc_aut)
pc_swr = pd.DataFrame(index = neurons, data = pc_swr)



from itertools import product
corrpair = pd.DataFrame(index=list(combinations(neurons,r=2)),columns = ['swr', 'auto', 'distance'])

for i, j in corrpair.index.values:
	print(i,j)
	corrpair.loc[(i,j),'swr'] =  float(scipy.stats.pearsonr(pc_swr.loc[i].values, pc_swr.loc[j].values)[0])
	corrpair.loc[(i,j),'auto'] = float(scipy.stats.pearsonr(pc_aut.loc[i].values, pc_aut.loc[j].values)[0])
	x = space.loc[[i,j], 'shank'].values
	y = space.loc[[i,j], 'session'].values
	corrpair.loc[(i,j),'distance'] = float(np.sqrt(np.power(np.diff(x), 2) + np.power(np.diff(y), 2))[0])


a = (corrpair[['swr','auto']]>0.85).prod(1)
idx = a[a==1].index.values

dist = corrpair.loc[idx,'distance'].astype('float')

# totest = corrpair.loc[idx, 'distance'].sort_values()[::-1]
# totest = corrpair.loc[idx,'auto'].sort_values().index.values
totest = (dist[dist>1]).index.values

position = pd.DataFrame(index = totest, columns = [0,1])
for i,j in totest:
	position.loc[(i,j),0] = space.loc[i,'session']
	position.loc[(i,j),1] = space.loc[j,'session']

t = -3

for j, ep in enumerate(['wake', 'rem', 'sws']):	
	subplot(2,3,j+1)
	for n in totest[t]:
		tmp = store_autocorr[ep][n].loc[-50:50]
		tmp.loc[0] = 0.0
		plot(tmp)
subplot(2,1,2)
for n in totest[t]:
	plot(swr[n])
show()


n = 'Mouse17-130211_20'

pairs = [(n,i) for i in neurons]

cp = corrpair.loc[pairs].dropna()
a = (cp[['swr','auto']]>0.8).prod(1)
idx = a[a==1].index.values

