

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
import _pickle as cPickle
import time
import os, sys
import ipyparallel
import neuroseries as nts
import scipy.stats
from pylab import *
from multiprocessing import Pool

from scipy.optimize import curve_fit

def func(x, a, b, c):
	return a*np.exp(-b*x) + c


nucleus = os.listdir('/mnt/DataGuillaume/corr_pop_nucleus/')
nucleus.remove('U')
nucleus.remove('sm')

sessions = {n:os.listdir('/mnt/DataGuillaume/corr_pop_nucleus/'+n+'/') for n in nucleus}
sessions = {n:sessions[n] for n in sessions.keys() if len(sessions[n])}

dt = 0.1
bins = np.arange(0, 3.1, dt)
times = bins[0:-1] + dt/2

df = pd.DataFrame(index = np.concatenate([-times[::-1], times]), columns = pd.MultiIndex.from_product([sessions.keys(), ['wak', 'rem', 'sws'], ['mean', 'sem']]))

for n in sessions.keys():
	corr = {e:pd.Series() for e in ['wak', 'rem', 'rip']}
	print(n)
	for s in sessions[n]:		
		store 			= pd.HDFStore("/mnt/DataGuillaume/corr_pop_nucleus/"+n+"/"+s)
		for e in ['allwak_corr', 'allrem_corr', 'allrip_corr']:
			tmp = store[e]
			tmp = tmp[np.abs(tmp.index) < 3.0]
			corr[e[3:6]] = corr[e[3:6]].append(tmp)
		store.close()
	for e in ['wak', 'rem', 'rip']:
		idx = np.digitize(np.abs(corr[e].index.values), bins)-1
		df.loc[0:,(n,e,'mean')] = np.array([corr[e][idx == i].mean(skipna=True) for i in range(len(bins)-1)]).flatten()
		df.loc[:0,(n,e,'mean')] = df.loc[0:, (n,e,'mean')].values[::-1]
		df.loc[0:,(n,e,'sem')] = np.array([corr[e][idx == i].sem(skipna=True) for i in range(len(bins)-1)]).flatten()
		df.loc[:0,(n,e,'sem')] = df.loc[0:, (n,e,'sem')].values[::-1]		 


df.to_hdf("../figures/figures_articles/figure4/meancorrpop_nucleus.h5", 'meancorr_nucleus')



lambdaa = pd.DataFrame(	index = np.unique(df.columns.get_level_values(0)), 
						columns = pd.MultiIndex.from_product([['wak', 'rem', 'rip'], ['a', 'b', 'c']]))

for ep in ['wak', 'rem', 'rip']:
	for n in lambdaa.index.values:
		tmp = df[n][ep]['mean']
		tmp = tmp.loc[0:]
		tmp = tmp.loc[tmp.argmax():]
		try:
			popt, pcov = curve_fit(func, tmp.index.values.astype('float32'), tmp.values.astype('float32'))
			lambdaa.loc[n, ep] = popt
		except:
			print(ep, n)
			pass


lambdaa.to_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_POPCORR_NUCLEUS.h5", 'lambdaanucleus')




figure()
for i, e in enumerate(['wak', 'rem', 'rip']):
	subplot(1,3,i+1)
	tmp = df.xs([e, 'mean'], 1, [1,2])
	for n in tmp.columns:
		plot(tmp[n], label = n)
	legend()