import ternary
import numpy as np
import pandas as pd
from functions import *
import sys
from functools import reduce
from sklearn.manifold import *
from sklearn.cluster import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from pylab import *
import _pickle as cPickle
from skimage.filters import gaussian
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

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
									columns = ['phase', 'pvalue', 'kappa'],
									data = theta_mod['rem'])
# rippower 				= pd.read_hdf("../figures/figures_articles/figure2/power_ripples_2.h5")
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
swr = swr.loc[-200:200]

# AUTOCORR FAST
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")
autocorr_wak = store_autocorr['wake'].loc[0.5:]
autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
autocorr_sws = 	store_autocorr['sws'].loc[0.5:]
autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_wak = autocorr_wak[2:150]
autocorr_rem = autocorr_rem[2:150]
autocorr_sws = autocorr_sws[2:150]

# HISTOGRAM THETA
theta_hist = pd.read_hdf("/mnt/DataGuillaume/MergedData/THETA_THAL_HISTOGRAM_2.h5")
theta_hist = theta_hist.rolling(window = 5, win_type='gaussian', center = True, min_periods=1).mean(std=1.0)
theta_wak = theta_hist.xs(('wak'), 1, 1)
theta_rem = theta_hist.xs(('rem'), 1, 1)

# AUTOCORR LONG
store_autocorr2 = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG.h5")
autocorr2_wak = store_autocorr2['wak'].loc[0.5:]
autocorr2_rem = store_autocorr2['rem'].loc[0.5:]
autocorr2_sws = store_autocorr2['sws'].loc[0.5:]
autocorr2_wak = autocorr2_wak.rolling(window = 100, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 10.0)
autocorr2_rem = autocorr2_rem.rolling(window = 100, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 10.0)
autocorr2_sws = autocorr2_sws.rolling(window = 100, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 10.0)
autocorr2_wak = autocorr2_wak[2:2000]
autocorr2_rem = autocorr2_rem[2:2000]
autocorr2_sws = autocorr2_sws[2:2000]

############################################################################################################
# WHICH NEURONS
############################################################################################################
firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
fr_index = firing_rate.index.values[((firing_rate >= 1.0).sum(1) == 3).values]
# neurons = reduce(np.intersect1d, (burstiness.index.values, theta.index.values, rippower.index.values, fr_index))
neurons = reduce(np.intersect1d, (fr_index, autocorr_sws.columns, autocorr2_rem.columns, theta_rem.columns, swr.columns, lambdaa.index.values))

# neurons = np.array([n for n in neurons if 'Mouse17' in n])

# nucleus = ['AD', 'AM', 'AVd', 'AVv', 'VA', 'LDvl', 'CM']
# neurons = np.intersect1d(neurons, mappings.index[mappings['nucleus'].isin(nucleus)])
count_nucl = pd.DataFrame(columns = ['12', '17','20', '32'])

for m in ['12', '17','20', '32']:
	subspace = pd.read_hdf("../figures/figures_articles/figure1/subspace_Mouse"+m+".hdf5")	
	nucleus = np.unique(subspace['nucleus'])
	total = [np.sum(subspace['nucleus'] == n) for n in nucleus]
	count_nucl[m] = pd.Series(index = nucleus, data = total)
nucleus = list(count_nucl.dropna().index.values)
allnucleus = list(np.unique(mappings.loc[neurons,'nucleus']))
tokeep = np.array([n for n in neurons if mappings.loc[n,'nucleus'] in nucleus])
############################################################################################################
# STACKING DIMENSIONS
############################################################################################################


# pc_short_rem = PCA(n_components=10).fit_transform(autocorr_rem[neurons].values.T)
# pc_short_wak = PCA(n_components=10).fit_transform(autocorr_wak[neurons].values.T)
# pc_short_sws = PCA(n_components=10).fit_transform(autocorr_sws[neurons].values.T)
# pc_short_rem = np.log((pc_short_rem - pc_short_rem.min(axis = 0))+1)
# pc_short_wak = np.log((pc_short_wak - pc_short_wak.min(axis = 0))+1)
# pc_short_sws = np.log((pc_short_sws - pc_short_sws.min(axis = 0))+1)
# pc_long = PCA(n_components=1).fit_transform(autocorr2_rem[neurons].values.T)
# pc_long = np.log((pc_long - pc_long.min(axis=0))+1) 
# # pc_long = np.log(lambdaa.loc[neurons].values[:,np.newaxis])
# # pc_theta = np.hstack([np.cos(theta.loc[neurons,'phase']).values[:,np.newaxis],np.sin(theta.loc[neurons,'phase']).values[:,np.newaxis],np.log(theta.loc[neurons,'kappa'].values[:,np.newaxis])])
# pc_theta = np.hstack([np.log(theta.loc[neurons,'kappa'].values[:,np.newaxis])])
# pc_swr   = np.hstack([np.log(rippower.loc[neurons].values[:,np.newaxis])])
# pc_theta = PCA(n_components=3).fit_transform(theta_rem[neurons].values.T)
# pc_theta = np.log((pc_theta - pc_theta.min(axis = 0))+1)
# pc_swr 	 = PCA(n_components=3).fit_transform(swr[neurons].values.T)
# pc_swr 	 = np.log((pc_swr - pc_swr.min(axis = 0))+1)
# pc_theta -= pc_theta.min(axis = 0)
# pc_swr 	 -= pc_swr.min(axis = 0)
# pc_theta = np.log(pc_theta+1)
# pc_swr 	 = np.log(pc_swr+1)
# data = []
# for tmp in [autocorr_sws[neurons].values.T,autocorr2_rem[neurons].values.T,theta_rem[neurons].values.T,swr[neurons].values.T]:
# 	tmp = tmp - tmp.min()
# 	tmp = tmp / tmp.max()
# 	data.append(tmp)
# data = np.hstack([pc_short_rem, pc_short_sws, pc_long, pc_short_wak, pc_long, pc_theta, pc_swr])
# data = np.hstack([pc_short_rem, pc_short_sws, pc_short_wak])
# data = np.hstack([pc_theta, pc_swr])


data = np.vstack([	autocorr_wak[tokeep].values,autocorr_rem[tokeep].values,autocorr_sws[tokeep].values,
					autocorr2_wak[tokeep].values,autocorr2_rem[tokeep].values,autocorr2_sws[tokeep].values,
					theta_hist.xs(('wak'),1,1)[tokeep].values,theta_hist.xs(('rem'),1,1)[tokeep].values,
					swr[tokeep].values]).T

# data = np.vstack([	autocorr_wak[neurons].values,autocorr_rem[neurons].values,autocorr_sws[neurons].values]).T

##########################################################################################################
# CLASSIFIER
##########################################################################################################
labels = np.array([nucleus.index(n) for n in mappings.loc[tokeep,'nucleus']])



clf = LogisticRegressionCV(cv = 5, random_state = 0, multi_class='multinomial', verbose = 3, n_jobs = 8, max_iter = 100).fit(data, labels)






mean_score = pd.DataFrame(index = nucleus,columns=['score'])
for i,n in enumerate(nucleus):
	mean_score.loc[n] = clf.score(data[labels == i], labels[labels==i])

mean_score = mean_score.loc[nucleus].sort_values('score', ascending=False)


mean_score_random = pd.DataFrame(index = nucleus,columns = np.arange(10))
for i in range(mean_score_random.shape[1]):
	tmp = labels.copy()
	np.random.shuffle(tmp)
	clf_random = LogisticRegressionCV(cv = 5, random_state = np.random.randint(100), multi_class='multinomial', verbose = 3, n_jobs = 8, max_iter = 100).fit(data, tmp)
	for j,n in enumerate(nucleus):
		mean_score_random.loc[n,i] = clf_random.score(data[labels==j], labels[labels==j])


p = clf.predict_proba(data)
mean_p = pd.DataFrame(index = allnucleus, columns = allnucleus)
for i, n in enumerate(allnucleus):
	mean_p.loc[n] = p[labels==i].mean(0)

figure()
subplot(121)
imshow(mean_p.values.astype('float'))
xticks(np.arange(len(allnucleus)),mean_p.index.values)
yticks(np.arange(len(allnucleus)),mean_p.index.values)
subplot(122)
bar(np.arange(len(nucleus)), mean_score.values.flatten())
xticks(np.arange(len(nucleus)), mean_score.index.values)
show()

