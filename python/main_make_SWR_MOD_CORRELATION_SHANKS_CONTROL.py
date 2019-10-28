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
# PER SHANKS
##############################################################################################################

corr_shanks = {0:[], 1:[]}


for m in np.sort(list(mouse_index.keys())):
	submap = mappings.loc[mouse_index[m]]
	session_index = submap.groupby('session').groups
	for s in session_index.keys(): # sessions
		session_shank_index = submap.loc[session_index[s]].groupby('shank').groups
		for k in session_shank_index.keys(): # shank			
			session_shank_hd_index = submap.loc[session_shank_index[k]].groupby('hd').groups		
			for n in session_shank_hd_index.keys():	# hd/nonhd
				if len(session_shank_hd_index[n]) > 1:
					if n == 0:
						if 1 not in session_shank_hd_index.keys():
							swrmod = swr[session_shank_hd_index[n]].values
							C = np.corrcoef(swrmod.T)
							corr_shanks[n].append(C[np.triu_indices_from(C,1)])							
					else: 
						swrmod = swr[session_shank_hd_index[n]].values
						C = np.corrcoef(swrmod.T)
						corr_shanks[n].append(C[np.triu_indices_from(C,1)])



hd_corr_shanks = np.hstack(corr_shanks[1])
nohd_corr_shanks = np.hstack(corr_shanks[0])

bins = np.linspace(-1,1,20)

tosave = {'hd':hd_corr_shanks,'nohd':nohd_corr_shanks}

# cPickle.dump(tosave, open('../figures/figures_articles_v4/figure2/SWR_MOD_CORRELATION_SHANKS.pickle', 'wb'))

figure()
hist(nohd_corr_shanks, bins=bins, weights=np.ones_like(nohd_corr_shanks)/float(len(nohd_corr_shanks)), alpha = 0.5)
hist(hd_corr_shanks, bins=bins, weights=np.ones_like(hd_corr_shanks)/float(len(hd_corr_shanks)), alpha = 0.5)

show()

