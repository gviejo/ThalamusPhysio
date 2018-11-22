import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import neuroseries as nts
import sys
import scipy.ndimage.filters as filters
from sklearn.mixture import GaussianMixture
from sklearn.cluster import *
from functools import reduce
from multiprocessing import Pool
import h5py as hd
from scipy.stats import zscore
from sklearn.manifold import TSNE, SpectralEmbedding
from skimage import filters
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV

# store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG_SMALLBINS.h5")

firing_rate = pd.HDFStore("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")['firing_rate']
fr_index = firing_rate.index.values[((firing_rate > 1.0).sum(1) == 3).values]

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")




#########################################################################################
# COMPARING LENGTH OF AUTOCORR FOR KMEANS
#########################################################################################
# cuttime = np.arange(10,5000,10)
cuttime = np.geomspace(10, 5000, num = 40, dtype = np.int)
n_repeat = 100
score = pd.DataFrame(index = cuttime, columns = ['score'])
ct = 0

for c in cuttime:
	print(ct)
	ct+=1
	autocorr_wak = store_autocorr['wak']
	autocorr_rem = store_autocorr['rem']
	autocorr_sws = store_autocorr['sws']
	autocorr_wak = store_autocorr['wak'].loc[0.5:]
	autocorr_rem = store_autocorr['rem'].loc[0.5:]
	autocorr_sws = store_autocorr['sws'].loc[0.5:]
	autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
	autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
	autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
	neurons = np.intersect1d(np.intersect1d(autocorr_wak.columns, autocorr_rem.columns), autocorr_sws.columns)
	neurons = np.intersect1d(neurons, fr_index)
	autocorr = pd.concat([autocorr_sws[2:c][neurons],autocorr_rem[2:c][neurons],autocorr_wak[2:c][neurons]], ignore_index = False)
	if autocorr.isnull().any().any(): autocorr = autocorr.dropna(axis = 1, how = 'any')	
	neurons = autocorr.columns
	hd = mappings.loc[neurons, 'hd'].values.astype('int')
	data = autocorr.values.T	
	clf = LogisticRegressionCV(cv = 8, random_state = 0).fit(data, hd)
	# test
	# idx = np.hstack((np.where(hd)[0],np.random.choice(np.where(~hd)[0], np.sum(hd), replace=False)))
	idx = np.where(hd)[0]
	score.loc[c] = clf.score(data[idx], hd[idx])
	# score.loc[c] = clf.predict_proba(data[idx])[:,1].mean()
	

score.to_hdf("../figures/figures_articles/figure1/score_logreg.h5", 'count')

sys.exit()



figure()
for i, c in enumerate(np.arange(10, 250, 10)):
	autocorr_wak = store_autocorr['wak']
	autocorr_rem = store_autocorr['rem']
	autocorr_sws = store_autocorr['sws']
	autocorr_wak = store_autocorr['wak'].loc[0.5:]
	autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
	autocorr_sws = 	store_autocorr['sws'].loc[0.5:]
	autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
	autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
	autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
	neurons = np.intersect1d(np.intersect1d(autocorr_wak.columns, autocorr_rem.columns), autocorr_sws.columns)
	neurons = np.intersect1d(neurons, fr_index)
	autocorr = pd.concat([autocorr_sws[2:c][neurons],autocorr_rem[2:c][neurons],autocorr_wak[2:c][neurons]], ignore_index = False)
	if autocorr.isnull().any().any(): autocorr = autocorr.dropna(axis = 1, how = 'any')	
	neurons = autocorr.columns
	hd_index = mappings.index.values[np.where(mappings['hd'])]
	hd_index = np.intersect1d(hd_index, neurons)
	data = autocorr.values.T
	TSNE, divergence = makeAllTSNE(data, 1)
	tsne = pd.DataFrame(index = neurons, data = TSNE[0].T)
	# km 	= KMeans(n_clusters=2).fit(data)
	km = AgglomerativeClustering(n_clusters=2).fit(data)
	subplot(4,6,i+1)
	scatter(tsne[0], tsne[1], s = 10, c = km.labels_)
	scatter(tsne.loc[hd_index,0], tsne.loc[hd_index,1], s = 3)

show()
