import numpy as np
import pandas as pd
import sys, os
from functions import *
from pylab import *
from sklearn.manifold import TSNE

store_fourier = pd.HDFStore("/mnt/DataGuillaume/MergedData/FOURIER_OF_AUTOCORR.h5", 'r')

fte = {}
for e in ['wak', 'rem', 'sws']:
	fte[e] = store_fourier[e]
store_fourier.close()

firing_rate = pd.HDFStore("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")['firing_rate']
fr_index = firing_rate.index.values[((firing_rate >= 1.0).sum(1) == 3).values]


neurons = np.intersect1d(np.intersect1d(fte['wak'].columns, fte['rem'].columns), fte['sws'].columns)
neurons = np.intersect1d(neurons, fr_index)

# for e in fte.keys():
# 	fte[e] 	= fte[e].rolling(window=100, win_type='gaussian', center=True, min_periods =1).mean(std=10.0)
# 	fte[e] = fte[e].groupby(np.digitize(fte[e].index.values, np.arange(0, 1000, 2))).mean()


data = np.hstack([fte['wak'].loc[0:150,neurons].values.T, fte['rem'].loc[0:150,neurons].values.T, fte['sws'].loc[0:150,neurons].values.T])


mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
hd_index = mappings.index[mappings['hd'] == 1]

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']



tsne = TSNE(n_components=2, perplexity = 30).fit_transform(data)
tsne = pd.DataFrame(index = neurons, data = tsne)
scatter(tsne[0], tsne[1], s = 40, c = burst.loc[neurons, 'sws'].values)
scatter(tsne.loc[hd_index,0], tsne.loc[hd_index,1], s = 3, color = 'red')

show()


