import ternary
import numpy as np
import pandas as pd
from functions import *
import sys
from functools import reduce
from sklearn.manifold import *
from sklearn.cluster import *
from pylab import *
import _pickle as cPickle
from skimage.filters import gaussian

############################################################################################################
# LOADING DATA
############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
burstiness 				= pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
lambdaa  = pd.read_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_AUTOCORR.h5")[('rem', 'b')]
lambdaa = lambdaa[np.logical_and(lambdaa>0.0,lambdaa<30.0)]



theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta 					= pd.DataFrame(	index = theta_ses['rem'], 
										columns = pd.MultiIndex.from_product([['wak', 'rem'], ['phase', 'pvalue', 'kappa']]),
										data = np.hstack((theta_mod['wake'], theta_mod['rem'])))
theta 					= theta.dropna()
rippower 				= pd.read_hdf("../figures/figures_articles/figure2/power_ripples_2.h5")
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
swr_phase = pd.read_hdf("/mnt/DataGuillaume/MergedData/SWR_PHASE.h5")

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
store_autocorr.close()

space = pd.read_hdf("../figures/figures_articles_v2/figure1/space.hdf5")

############################################################################################################
# WHICH NEURONS
############################################################################################################
firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
fr_index = firing_rate.index.values[((firing_rate >= 1.0).sum(1) == 3).values]

neurons = np.intersect1d(swr.dropna(1).columns.values, autocorr_sws.dropna(1).columns.values)
neurons = np.intersect1d(neurons, fr_index)

from sklearn.decomposition import PCA



data = pd.DataFrame(columns = ['sws', 'rem', 'wak', 'all', 'distance'])

for i, auto, ep in zip(range(3), [autocorr_sws, autocorr_rem, autocorr_wak], ['sws', 'rem', 'wak']):
	X = np.copy(swr[neurons].values.T)
	Y = np.copy(auto[neurons].values.T)
	pc_swr = PCA(n_components=10).fit_transform(X)
	pc_aut = PCA(n_components=10).fit_transform(Y)	
	All = np.hstack((pc_swr, pc_aut))
	corr = np.corrcoef(All)
	data[ep] = corr[np.triu_indices_from(corr,1)]


X = np.copy(swr[neurons].values.T)
Y = np.copy(np.vstack((autocorr_wak[neurons].values,autocorr_rem[neurons].values, autocorr_sws[neurons].values))).T
pca_swr = PCA(n_components=10).fit(X)
pca_aut = PCA(n_components=10).fit(Y)
pc_swr = pca_swr.transform(X)
pc_aut = pca_aut.transform(Y)
All = np.hstack((pc_swr, pc_aut))
corr = np.corrcoef(All)
data['all'] = corr[np.triu_indices_from(corr,1)]


# sys.exit()

from itertools import combinations

groups = {m:[n for n in neurons if m in n] for m in ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']}

pairs = {m:list(combinations(groups[m],r=2)) for m in groups.keys()}

dist = pd.DataFrame(index=neurons,columns=neurons, data = 0.0)

for m in groups.keys():
	x = space.loc[groups[m], 'x'].values
	y = space.loc[groups[m], 'y'].values
	tmp = np.sqrt(np.power(np.atleast_2d(x).T - x, 2) + np.power(np.atleast_2d(y).T - y, 2))
	distm = pd.DataFrame(index=groups[m],columns=groups[m], data=tmp)
	for i,j in pairs[m]:
		dist.loc[i,j] = distm.loc[i,j]

tmp = dist.values
tmp = tmp[np.triu_indices_from(tmp,1)]
data['distance'] = tmp

bins = np.arange(data['distance'].min(), data['distance'].max()+0.01, 0.1)
idx = np.digitize(data['distance'].values, bins)
mean = data[['wak', 'rem', 'sws', 'all']].groupby(idx).mean()
std = data[['wak', 'rem', 'sws', 'all']].groupby(idx).sem()
plot(mean['all'])
x = mean['all'].index.values
y = mean['all'].values
s = std['all'].values
fill_between(x, y-s, y+s, alpha = 0.5)
show()
	