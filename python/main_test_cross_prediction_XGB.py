import ternary
import numpy as np
import pandas as pd
from functions import *
import sys
from functools import reduce
from sklearn.manifold import *
from sklearn.cluster import *
from pylab import *
import _pickle as cPickle
from skimage.filters import gaussian

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
										columns = pd.MultiIndex.from_product([['wak', 'rem'], ['phase', 'pvalue', 'kappa']]),
										data = np.hstack((theta_mod['wake'], theta_mod['rem'])))
theta 					= theta.dropna()
rippower 				= pd.read_hdf("../figures/figures_articles/figure2/power_ripples_2.h5")
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
swr = swr.loc[-500:500]

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
# neurons = reduce(np.intersect1d, (burstiness.index.values, theta.index.values, rippower.index.values))
# neurons = reduce(np.intersect1d, (autocorr_sws.columns, autocorr2_rem.columns, theta_rem.columns, swr.columns, lambdaa.index.values))
neurons = reduce(np.intersect1d, (autocorr_rem.columns, autocorr_wak.columns, autocorr_sws.columns, autocorr2_wak.columns, autocorr2_rem.columns, theta.index.values, rippower.index.values))
# neurons = np.array([n for n in neurons if 'Mouse17' in n])

count_nucl = pd.DataFrame(columns = ['12', '17','20', '32'])

for m in ['12', '17','20', '32']:
	subspace = pd.read_hdf("/mnt/DataGuillaume/MergedData/subspace_Mouse"+m+".hdf5")
	nucleus = np.unique(subspace['nucleus'])
	total = [np.sum(subspace['nucleus'] == n) for n in nucleus]
	count_nucl[m] = pd.Series(index = nucleus, data = total)
nucleus = list(count_nucl.dropna().index.values)
allnucleus = list(np.unique(mappings.loc[neurons,'nucleus']))
tokeep = np.array([n for n in neurons if mappings.loc[n,'nucleus'] in nucleus])


############################################################################################################
# 
############################################################################################################
import xgboost as xgb
from sklearn.model_selection import KFold

def xgb_prediction(Xr, Yr, Xt):          
	dtrain = xgb.DMatrix(Xr, label=Yr)
	dtest = xgb.DMatrix(Xt)

	params = {'objective': "reg:linear",
	'eval_metric': "rmse", #loglikelihood loss
	'seed': np.random.randint(1, 10000), #for reproducibility
	'silent': 1,
	'learning_rate': 0.05,
	'min_child_weight': 2, 
	'n_estimators': 100,
	# 'subsample': 0.5,
	'max_depth': 5, 
	'gamma': 0.5}

	num_round = 1000
	bst = xgb.train(params, dtrain, num_round)
	ymat = bst.predict(dtest)	
	return ymat

def fit_cv(X, Y, n_cv=10, verbose=1, shuffle = False):
	if np.ndim(X)==1:
		X = np.transpose(np.atleast_2d(X))
	cv_kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
	skf  = cv_kf.split(X)    
	Y_hat=np.zeros(len(Y))*np.nan	

	for idx_r, idx_t in skf:        
		Xr = np.copy(X[idx_r, :])
		Yr = np.copy(Y[idx_r])
		Xt = np.copy(X[idx_t, :])
		Yt = np.copy(Y[idx_t])
		if shuffle: np.random.shuffle(Yr)
		Yt_hat = xgb_prediction(Xr, Yr, Xt)
		Y_hat[idx_t] = Yt_hat
		
	return Y_hat

meanmse = pd.DataFrame(index = nucleus, columns = pd.MultiIndex.from_product([['swr', 'auto'], ['mean', 'sem']]))

# SWR -> Burstiness
X = swr[tokeep].values.T
Y = burstiness.loc[tokeep,'sws'].values
Yp = fit_cv(X, Y)
mse = pd.DataFrame(index = tokeep, data = np.sqrt(np.power(Y - Yp,2)))

for n in nucleus:
	meanmse.loc[n,('swr','mean')] = mse.loc[mappings.groupby('nucleus').groups[n]].mean()[0]
	meanmse.loc[n,('swr','sem')] = mse.loc[mappings.groupby('nucleus').groups[n]].sem()[0]

# AUTOCORR -> KAPPA
X = autocorr_sws[tokeep].values.T
Y = theta.loc[tokeep,('rem', 'kappa')].values
Yp = fit_cv(X, Y)
mse = pd.DataFrame(index = tokeep, data = np.sqrt(np.power(Y - Yp,2)))

for n in nucleus:
	meanmse.loc[n,('auto','mean')] = mse.loc[mappings.groupby('nucleus').groups[n]].mean()[0]
	meanmse.loc[n,('auto','sem')] = mse.loc[mappings.groupby('nucleus').groups[n]].sem()[0]


meanmse = meanmse.sort_values(('swr', 'mean'))

# bar(np.arange(len(nucleus)),meanmse.sort_values('mean')['mean'])
figure()
subplot(211)
bar(np.arange(len(nucleus)), meanmse[('swr', 'mean')])
xticks(np.arange(len(nucleus)), nucleus)
title("swr -> burstiness")
subplot(212)
bar(np.arange(len(nucleus)), meanmse[('auto', 'mean')])
xticks(np.arange(len(nucleus)), nucleus)
title("autocorr -> kappa")

show()
