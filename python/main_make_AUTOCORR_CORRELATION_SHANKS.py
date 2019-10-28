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

store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")

autocorr_wak = store_autocorr['wake']
autocorr_rem = store_autocorr['rem']
autocorr_sws = store_autocorr['sws']

# 1. starting at 2
autocorr_wak = store_autocorr['wake'].loc[0.5:]
autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
autocorr_sws = 	store_autocorr['sws'].loc[0.5:]

# 3. lower than 200 
autocorr_wak = autocorr_wak.drop(autocorr_wak.columns[autocorr_wak.apply(lambda col: col.max() > 200.0)], axis = 1)
autocorr_rem = autocorr_rem.drop(autocorr_rem.columns[autocorr_rem.apply(lambda col: col.max() > 200.0)], axis = 1)
autocorr_sws = autocorr_sws.drop(autocorr_sws.columns[autocorr_sws.apply(lambda col: col.max() > 200.0)], axis = 1)
# # 4. gauss filt
# autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
# autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
# autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)

autocorr_wak = autocorr_wak[0:20]
autocorr_rem = autocorr_rem[0:20]
autocorr_sws = autocorr_sws[0:20]

# 6 combining all 
neurons = np.intersect1d(np.intersect1d(autocorr_wak.columns, autocorr_rem.columns), autocorr_sws.columns)

# 7 doing PCA
pc_short_rem = PCA(n_components=30).fit_transform(autocorr_rem[neurons].values.T)
pc_short_wak = PCA(n_components=30).fit_transform(autocorr_wak[neurons].values.T)
pc_short_sws = PCA(n_components=30).fit_transform(autocorr_sws[neurons].values.T)
# pc_short_rem = np.log((pc_short_rem - pc_short_rem.min(axis = 0))+1)
# pc_short_wak = np.log((pc_short_wak - pc_short_wak.min(axis = 0))+1)
# pc_short_sws = np.log((pc_short_sws - pc_short_sws.min(axis = 0))+1)

# data = np.hstack([autocorr_sws[neurons].values.T,autocorr_rem[neurons].values.T,autocorr_wak[neurons].values.T])
# data = autocorr_sws[neurons].values.T
data = np.hstack((pc_short_sws, pc_short_wak, pc_short_rem))


data = pd.DataFrame(columns = neurons, data = data.T)

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")

mappings = mappings.loc[neurons]

sessions = np.array([n.split("_")[0] for n in neurons])

mouse = np.array([n.split("-")[0] for n in neurons])

mappings['mouse'] = mouse

mouse_index = mappings.groupby('mouse').groups

##############################################################################################################
# PER SESSIONS
##############################################################################################################

corr_sessions = {0:[], 1:[]}

for m in np.sort(list(mouse_index.keys())):
	submap = mappings.loc[mouse_index[m]]
	session_index = submap.groupby('session').groups
	for s in session_index.keys():
		session_hd_index = submap.loc[session_index[s]].groupby('hd').groups		
		for n in session_hd_index.keys():
			if len(session_hd_index[n]) > 1:
				autoc = data[session_hd_index[n]].values				
				C = np.corrcoef(autoc.T)
				corr_sessions[n].append(C[np.triu_indices_from(C,1)])

hd_corr_sessions = np.hstack(corr_sessions[1])
nohd_corr_sessions = np.hstack(corr_sessions[0])

##############################################################################################################
# PER SHANKS
##############################################################################################################

corr_shanks = {0:[], 1:[]}

var_sessions_hd = {}
var_sessions_nonhd = {}
count_hd = {}
count_nohd = {}

for m in np.sort(list(mouse_index.keys())):
	submap = mappings.loc[mouse_index[m]]
	session_index = submap.groupby('session').groups
	for s in session_index.keys(): # sessions
		session_shank_index = submap.loc[session_index[s]].groupby('shank').groups
		for k in session_shank_index.keys(): # shank			
			session_shank_hd_index = submap.loc[session_shank_index[k]].groupby('hd').groups		
			for n in session_shank_hd_index.keys():	# hd/nonhd
				if len(session_shank_hd_index[n]) > 1:
					autoc = data[session_shank_hd_index[n]].values					
					C = np.corrcoef(autoc.T)
					corr_shanks[n].append(C[np.triu_indices_from(C,1)])


hd_corr_shanks = np.hstack(corr_shanks[1])
nohd_corr_shanks = np.hstack(corr_shanks[0])

bins = np.linspace(-1,1,20)

tosave = {'session':{'hd':hd_corr_sessions,'nohd':nohd_corr_sessions},
			'shank':{'hd':hd_corr_shanks,'nohd':nohd_corr_shanks}
			}

cPickle.dump(tosave, open('../figures/figures_articles_v4/figure5/AUTOCORR_CORRELATION_SHANKS.pickle', 'wb'))

figure()
subplot(121)
hist(nohd_corr_sessions, 20, weights=np.ones_like(nohd_corr_sessions)/float(len(nohd_corr_sessions)), alpha = 0.5)
hist(hd_corr_sessions, 20, weights=np.ones_like(hd_corr_sessions)/float(len(hd_corr_sessions)), alpha = 0.5)
subplot(122)
hist(nohd_corr_shanks, 20, weights=np.ones_like(nohd_corr_shanks)/float(len(nohd_corr_shanks)), alpha = 0.5)
hist(hd_corr_shanks, 20, weights=np.ones_like(hd_corr_shanks)/float(len(hd_corr_shanks)), alpha = 0.5)

show()

