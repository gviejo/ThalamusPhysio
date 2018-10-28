

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import time
import os, sys
import ipyparallel
import matplotlib.cm as cm
from scipy.optimize import curve_fit

def func(x, a, b, c):
	return a*np.exp(-b*x) + c


hd_sessions 	= np.sort(os.listdir("/mnt/DataGuillaume/corr_pop_hd/"))
nohd_sessions 	= np.sort(os.listdir("/mnt/DataGuillaume/corr_pop_no_hd/"))

corr = {c:{e:pd.Series() for e in ['wak', 'rem', 'rip']} for c in ['hd', 'nohd']}
for c, sessions, path in zip(['hd', 'nohd'], [hd_sessions, nohd_sessions], ['corr_pop_hd', 'corr_pop_no_hd']):
	for s in sessions:
		store 			= pd.HDFStore("/mnt/DataGuillaume/"+path+"/"+s)
		for e in ['allwak_corr', 'allrem_corr', 'allrip_corr']:
			tmp = store[e]
			tmp = tmp[np.abs(tmp.index) < 3.0]
			corr[c][e[3:6]] = corr[c][e[3:6]].append(tmp)
		store.close()

dt = 0.1
bins = np.arange(0, 3.1, dt)
times = bins[0:-1] + dt/2
columns = pd.MultiIndex.from_product([['wak', 'rem', 'rip'], ['hd', 'nohd'], ['mean', 'sem']], names = ['episode', 'label', 'mean-sem'])
meancorr = pd.DataFrame(index = np.concatenate([-times[::-1], times]), columns = columns)

for e in ['wak', 'rem', 'rip']:
	for l in ['hd', 'nohd']:
		idx = np.digitize(np.abs(corr[l][e].index.values), bins)-1
		meancorr.loc[0:,(e,l,'mean')] = np.array([corr[l][e][idx == i].mean(skipna=True) for i in range(len(bins)-1)]).flatten()
		meancorr.loc[:0,(e,l,'mean')] = meancorr.loc[0:, (e,l,'mean')].values[::-1]
		meancorr.loc[0:,(e,l,'sem')] = np.array([corr[l][e][idx == i].sem(skipna=True) for i in range(len(bins)-1)]).flatten()
		meancorr.loc[:0,(e,l,'sem')] = meancorr.loc[0:, (e,l,'sem')].values[::-1]		 

meancorr.to_hdf("../figures/figures_articles/figure4/meancorr_hd_nohd.h5", 'meancorr')
