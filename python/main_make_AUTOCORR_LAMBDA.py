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

from scipy.optimize import curve_fit

def func(x, a, b, c):
	return a*np.exp(-(1./b)*x) + c


store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG.h5")
autocorr_wak = store_autocorr['wak']
autocorr_rem = store_autocorr['rem']
autocorr_sws = store_autocorr['sws']
data = {'wak':autocorr_wak,'rem':autocorr_rem,'sws':autocorr_sws}
firing_rate = pd.HDFStore("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")['firing_rate']
fr_index = firing_rate.index.values[((firing_rate > 1.0).sum(1) == 3).values]
neurons = np.intersect1d(np.intersect1d(autocorr_wak.columns, autocorr_rem.columns), autocorr_sws.columns)
neurons = np.intersect1d(neurons, fr_index)




tmp1 = []
tmp2 = []
for ep in ['wak', 'rem', 'sws']:
	for i in ['a', 'b', 'c']:
		tmp1.append(ep)
		tmp2.append(i)

columns = pd.MultiIndex.from_arrays([tmp1, tmp2], names = ['episode', 'expfit'])
lambdaa_autocorr = pd.DataFrame(index = neurons, columns = columns, data = np.nan)


to_drop = []

for ep in ['wak', 'rem', 'sws']:	
	for n in neurons:
		tmp = data[ep][n].copy()		
		tmp.loc[0] = 0.0
		tmp = tmp.loc[tmp.loc[0.1:25.0].argmax():]
		tmp2 = tmp.rolling(window = 100, win_type='gaussian', center=True, min_periods=1).mean(std=5.0)
		tmp3 = tmp2 - tmp2.min()
		tmp3 = tmp3 / tmp3.max()
		try:
			popt, pcov = curve_fit(func, tmp3.index.values*1e-3, tmp3.values)		
			lambdaa_autocorr.loc[n, ep] = popt
		except:
			to_drop.append(n)
			pass

lambdaa_autocorr = lambdaa_autocorr.drop(to_drop)




toplot = np.array_split(neurons, 66)

ep = 'wak'

for i in range(len(toplot)):
	figure()
	for j,n in enumerate(toplot[i]):
		tmp = data[ep][n].copy()		
		tmp.loc[0] = 0.0
		tmp = tmp.loc[tmp.loc[0.1:25.0].argmax():]
		tmp2 = tmp.rolling(window = 100, win_type='gaussian', center=True, min_periods=1).mean(std=5.0)
		tmp3 = tmp2 - tmp2.min()
		tmp3 = tmp3 / tmp3.max()
		popt = lambdaa_autocorr.loc[n,ep].values
		subplot(4,5,j+1)
		plot(tmp3, label = None)
		plot(tmp3.index.values, func(tmp3.index.values*1e-3, popt[0], popt[1], popt[2]), '--', label = str(np.round(popt[1],2)))
		legend()
	show()



# bins = np.arange(0, 5040, 20)
# idx = np.digitize(tmp.index.values, bins)-1
# tmp2 = pd.Series(index = bins[0:-1], data = [np.mean(tmp.values[idx == i]) for i in np.arange(len(bins)-1)])
# tmp3 = tmp2.rolling(window = 100, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 10.0)
# try:
# 	popt, pcov = curve_fit(func, tmp3.index.values*1e-3, tmp3.values)		
# 	lambdaa_autocorr.loc[n, ep] = popt
# except:
# 	to_drop.append(n)
# 	pass

# # plot(tmp)
# plot(tmp3.index.values, func(tmp3.index.values*1e-3, popt[0], popt[1], popt[2]))




lambdaa_autocorr.to_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_AUTOCORR.h5", key = 'lambdaa', mode = 'w')