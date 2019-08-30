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
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
hd_index = mappings.index[np.where(mappings['hd'] == 1)[0]]
hd_index = hd_index[np.where((firing_rate.loc[hd_index]>1.0).all(axis=1))[0]]


# SWR MODULATION
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (1,)).transpose())
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
neurons = np.intersect1d(neurons, hd_index)


# looking for exemple
from itertools import product
hdmouse17 = [n for n in neurons if 'Mouse17' in n]
sess = np.unique([n.split("_")[0] for n in hdmouse17])
groups = {s:[n for n in hdmouse17 if s in n] for s in sess}
corrs = {}
for s in sess:
	if len(groups[s])>2:
		corr = pd.DataFrame(index = groups[s], columns = groups[s])
		for i, j in product(groups[s], groups[s]):
			corr.loc[i,j] = float(scipy.stats.pearsonr(swr[i].values, swr[j].values)[0])
		corr.iloc[range(len(groups[s])),range(len(groups[s]))] = 0
		corrs[s] = corr

sys.exit()



X = np.copy(swr[neurons].values.T)
Y = np.copy(np.vstack((autocorr_wak[neurons].values,autocorr_rem[neurons].values, autocorr_sws[neurons].values))).T
Y = Y - Y.mean(1)[:,np.newaxis]
Y = Y / Y.std(1)[:,np.newaxis]	
pca_swr = PCA(n_components=10).fit(X)
pca_aut = PCA(n_components=10).fit(Y)
pc_swr = pca_swr.transform(X)
pc_aut = pca_aut.transform(Y)
All = np.hstack((pc_swr, pc_aut))
corr = np.corrcoef(All.T)
d_hd = np.linalg.det(corr)

n_shuffling = 1000
shuffling = []

for j in range(n_shuffling):
	print(j)
	X = np.copy(swr[neurons].values.T)
	Y = np.copy(np.vstack((autocorr_wak[neurons].values,autocorr_rem[neurons].values, autocorr_sws[neurons].values))).T
	Y = Y - Y.mean(1)[:,np.newaxis]
	Y = Y / Y.std(1)[:,np.newaxis]	
	np.random.shuffle(X)
	np.random.shuffle(Y)
	pc_swr = PCA(n_components=10).fit_transform(X)
	pc_aut = PCA(n_components=10).fit_transform(Y)
	All = np.hstack((pc_swr, pc_aut))
	corr = np.corrcoef(All.T)
	d = np.linalg.det(corr)
	shuffling.append(d)


hist(1-np.array(shuffling),100)
axvline(1-d_hd)

show()


figure()
gs = gridspec.GridSpec(2,3, wspace = 0.3, hspace = 0.4)


