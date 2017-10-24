#!/usr/bin/env python

'''
    File name: main_ripp_mod.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2

Sharp-waves ripples modulation 
Used to make figure 1,2

'''

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle

###############################################################################################################
# LOADING DATA
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
spind_mod, spind_ses 	= loadSpindMod('/mnt/DataGuillaume/MergedData/SPINDLE_mod.pickle', datasets, return_index=True)

nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 					= pd.DataFrame(	index = swr_ses, 
										columns = times,
										data = swr_mod)

phase 					= pd.DataFrame(index = theta_ses['wake'], columns = ['theta_wake', 'theta_rem', 'spindle_hpc', 'spindle_thl'])
phase.loc[theta_ses['wake'],'theta_wake'] = theta_mod['wake'][:,0]
phase.loc[theta_ses['rem'], 'theta_rem'] = theta_mod['rem'][:,0]
phase.loc[spind_ses['hpc'], 'spindle_hpc'] = spind_mod['hpc'][:,0]
phase.loc[spind_ses['thl'], 'spindle_thl'] = spind_mod['thl'][:,0]

pvalue 					= pd.DataFrame(index = theta_ses['wake'], columns = ['theta_wake', 'theta_rem', 'spindle_hpc', 'spindle_thl'])
pvalue.loc[theta_ses['wake'], 'theta_wake'] = theta_mod['wake'][:,1]
pvalue.loc[theta_ses['rem'], 'theta_rem'] = theta_mod['rem'][:,1]
pvalue.loc[spind_ses['hpc'], 'spindle_hpc'] = spind_mod['hpc'][:,1]
pvalue.loc[spind_ses['thl'], 'spindle_thl'] = spind_mod['thl'][:,1]

kappa 					= pd.DataFrame(index = theta_ses['wake'], columns = ['theta_wake', 'theta_rem', 'spindle_hpc', 'spindle_thl'])
kappa.loc[theta_ses['wake'], 'theta_wake'] = theta_mod['wake'][:,2]
kappa.loc[theta_ses['rem'], 'theta_rem'] = theta_mod['rem'][:,2]
kappa.loc[spind_ses['hpc'], 'spindle_hpc'] = spind_mod['hpc'][:,2]
kappa.loc[spind_ses['thl'], 'spindle_thl'] = spind_mod['thl'][:,2]

# filtering swr_mod
swr 				= pd.DataFrame(	index = swr.index, 
									columns = swr.columns,
									data = gaussFilt(swr.values, (10,)))

# Cut swr_mod from -500 to 500
nbins 				= 200
binsize				= 5
times 				= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 				= swr.loc[:,times]

# CHECK FOR NAN
tmp1 			= swr.index[np.unique(np.where(np.isnan(swr))[0])]
tmp2 			= phase.index[phase.isnull().any(1)]
# CHECK P-VALUE 
tmp3	 		= pvalue.index[(pvalue['theta_rem'] > 0.1).values]
tmp 			= np.unique(np.concatenate([tmp1,tmp2,tmp3]))
# copy and delete 
if len(tmp):
	swr_modth 	= swr.drop(tmp)
	theta_modth = np.zeros((len(swr_modth),3))	
	theta_modth[:,0] = phase.loc[swr_modth.index,	'theta_rem']
	theta_modth[:,1] = pvalue.loc[swr_modth.index,	'theta_rem']
	theta_modth[:,2] = kappa.loc[swr_modth.index,	'theta_rem']


neuron_index = swr_modth.index
swr_modth = swr_modth.values

sessions = []
for n in neuron_index:
	sessions.append(n.split("_")[0])
sessions = np.array(sessions)


best_kappa_theta_neuron_per_session = {}
min_value = 0
max_value = 0
best_s = sessions[0]
for s in np.unique(sessions):
	index = np.where(s == sessions)[0]
	neurons_in_session = list(neuron_index[index])	
	best_kappa_theta_neuron_per_session[s] = kappa.loc[neurons_in_session,'theta_rem'].sort_values()
	y = best_kappa_theta_neuron_per_session[s].values	
	if y[0] > min_value and y[-1] > max_value and y[-1] < 1.0:
		best_s = s
		min_value = y[0]
		max_value = y[-1]	
	

