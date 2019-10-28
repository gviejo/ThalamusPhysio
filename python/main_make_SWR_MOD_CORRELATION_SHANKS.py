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

swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
swr = swr.drop(swr.columns[swr.isnull().any()].values, axis = 1)
swr = swr.loc[-500:500]

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")


neurons = np.intersect1d(swr.columns.values, mappings.index.values)

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
				swrmod = swr[session_hd_index[n]].values
				C = np.corrcoef(swrmod.T)
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
		if m == 'Mouse17':
			var_sessions_hd[s] = np.zeros(8)*np.nan
			var_sessions_nonhd[s] = np.zeros(8)*np.nan
			count_hd[s] = np.ones(8)*np.nan
			count_nohd[s] = np.ones(8)*np.nan
		for k in session_shank_index.keys(): # shank			
			session_shank_hd_index = submap.loc[session_shank_index[k]].groupby('hd').groups		
			for n in session_shank_hd_index.keys():	# hd/nonhd
				if len(session_shank_hd_index[n]) > 1:
					swrmod = swr[session_shank_hd_index[n]].values
					C = np.corrcoef(swrmod.T)
					corr_shanks[n].append(C[np.triu_indices_from(C,1)])
					if m == 'Mouse17':
						if n == 1:
							var_sessions_hd[s][k] = np.var(C[np.triu_indices_from(C,1)])
							count_hd[s][k] = len(session_shank_hd_index[n])
						if n == 0:
							var_sessions_nonhd[s][k] = np.var(C[np.triu_indices_from(C,1)])
							count_nohd[s][k] = len(session_shank_hd_index[n])


hd_corr_shanks = np.hstack(corr_shanks[1])
nohd_corr_shanks = np.hstack(corr_shanks[0])

bins = np.linspace(-1,1,20)

tosave = {'session':{'hd':hd_corr_sessions,'nohd':nohd_corr_sessions},
			'shank':{'hd':hd_corr_shanks,'nohd':nohd_corr_shanks}
			}

cPickle.dump(tosave, open('../figures/figures_articles_v4/figure2/SWR_MOD_CORRELATION_SHANKS.pickle', 'wb'))

figure()
subplot(121)
hist(nohd_corr_sessions, bins=bins, weights=np.ones_like(nohd_corr_sessions)/float(len(nohd_corr_sessions)), alpha = 0.5)
hist(hd_corr_sessions, bins=bins, weights=np.ones_like(hd_corr_sessions)/float(len(hd_corr_sessions)), alpha = 0.5)
subplot(122)
hist(nohd_corr_shanks, bins=bins, weights=np.ones_like(nohd_corr_shanks)/float(len(nohd_corr_shanks)), alpha = 0.5)
hist(hd_corr_shanks, bins=bins, weights=np.ones_like(hd_corr_shanks)/float(len(hd_corr_shanks)), alpha = 0.5)

show()


# to pick exemple for figure 2
var_sessions_hd = pd.DataFrame.from_dict(var_sessions_hd).T
var_sessions_nonhd = pd.DataFrame.from_dict(var_sessions_nonhd).T
count_hd = pd.DataFrame.from_dict(count_hd).T
count_nohd = pd.DataFrame.from_dict(count_nohd).T

hd = pd.concat([var_sessions_hd.mean(1),count_hd.sum(1)],1)
nohd = pd.concat([var_sessions_nonhd.mean(1),count_nohd.sum(1)],1)

hd_ex = 'Mouse17/Mouse17-130130'
nohd_ex = 'Mouse17/Mouse17-130212'