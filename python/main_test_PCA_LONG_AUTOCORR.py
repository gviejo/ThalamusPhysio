	#!/usr/bin/env python

'''
	File name: main_make_mucleus_TSNE.py
	Author: Guillaume Viejo
	Date created: 28/09/2017    
	Python Version: 3.5.2


'''

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
from sklearn.cluster import KMeans
from functools import reduce
from multiprocessing import Pool
import h5py as hd
from scipy.stats import zscore
from sklearn.manifold import TSNE, SpectralEmbedding
from skimage import filters

###############################################################################################################
# LOADING DATA
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

# store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG.h5")
# autocorr_rem = store_autocorr['rem']
# autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
# autocorr_rem = autocorr_rem.drop(autocorr_rem.columns[autocorr_rem.apply(lambda col: col.max() > 100.0)], axis = 1)
# autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 4.0)
# autocorr_rem = autocorr_rem[0:3000]


store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")
autocorr_sws = store_autocorr['sws']
autocorr_sws = 	store_autocorr['sws'].loc[0.5:]
autocorr_sws = autocorr_sws.drop(autocorr_sws.columns[autocorr_sws.apply(lambda col: col.max() > 100.0)], axis = 1)
autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
autocorr_sws = autocorr_sws[0:50]

# sys.exit()
lambdaa  = pd.read_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_AUTOCORR.h5")

l = lambdaa[('rem', 'b')]

l = l[np.logical_and(l>0, l<=5)]

neurons = np.intersect1d(l.index.values, autocorr_sws.columns.values)

# pc_long_rem = PCA(n_components=30).fit_transform(autocorr_rem[neurons].values.T)
pc_short_sws = PCA(n_components=30).fit_transform(autocorr_sws[neurons].values.T)

# var_long_rem = PCA(n_components=30).fit(autocorr_rem[neurons].values.T)
# var_short_sws =PCA(n_components=30).fit(autocorr_sws[neurons].values.T)

print(scipy.stats.pearsonr(pc_short_sws[:,0], l.loc[neurons].values))

scatter(pc_short_sws[:,0], l.loc[neurons].values)
xlabel("non-REM pc 1")
ylabel("REM decay time (s)")
show()

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']


print(scipy.stats.pearsonr(pc_short_sws[:,0], burst.loc[neurons,'sws'].values))

sys.exit()

tosave = {'session':{'hd':hd_corr_sessions,'nohd':nohd_corr_sessions},
			'shank':{'hd':hd_corr_shanks,'nohd':nohd_corr_shanks}
			}

cPickle.dump(tosave, open('../figures/figures_articles_v4/figure5/AUTOCORR_CORRELATION_SHANKS.pickle', 'wb'))


