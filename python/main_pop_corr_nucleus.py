

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

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
nucleus = np.unique(mappings['nucleus'])
sessions = np.unique([n.split("_")[0] for n in mappings.index])

# determining number of neurons per nucleus et per sessions
count = pd.DataFrame(index=sessions, columns = nucleus,data=0)
for s in count.index:
	for n in nucleus:	
		count.loc[s,n] = (mappings[mappings.index.str.contains(s)]['nucleus'] == n).sum()

nucleus_session = {n:count.index.values[count[n]>5] for n in nucleus}

# sys.exit()

# make directory for each nucleus
for n in nucleus:
	try:
		os.mkdir("/mnt/DataGuillaume/corr_pop_nucleus/"+n)
	except:
		pass


def compute_population_correlation(nuc, session):
	start_time = time.clock()
	print(session)

	store 			= pd.HDFStore("/mnt/DataGuillaume/population_activity/"+session+".h5")
	rip_pop 		= store['rip']
	rem_pop 		= store['rem']
	wak_pop 		= store['wake']	
	store.close()

	# WHICH columns to keep
	mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
	tmp = mappings[mappings.index.str.contains(session)]['nucleus'] == nuc
	neurons = tmp.index.values[np.where(tmp)[0]]
	idx = np.array([int(n.split("_")[1]) for n in neurons])
	rip_pop = rip_pop[idx]
	rem_pop = rem_pop[idx]
	wak_pop = wak_pop[idx]


	###############################################################################################################
	# POPULATION CORRELATION FOR EACH RIPPLES
	###############################################################################################################
	#matrix of distance between ripples	in second	
	interval_mat = np.vstack(nts.TsdFrame(rip_pop).as_units('s').index.values) - nts.TsdFrame(rip_pop).as_units('s').index.values
	rip_corr = np.ones(interval_mat.shape)*np.nan
	# doing the upper part of the diagonal
	# rip_corr = np.eye(interval_mat.shape[0])
	# bad
	tmp = np.zeros_like(rip_corr)
	tmp[np.triu_indices(interval_mat.shape[0], 1)] += 1
	tmp[np.tril_indices(interval_mat.shape[0], 300)] += 1
	index = np.where(tmp == 2)
	
	for i, j in zip(index[0], index[1]):
		rip_corr[i,j] = scipy.stats.pearsonr(rip_pop.iloc[i].values, rip_pop.iloc[j].values)[0]
		rip_corr[j,i] = rip_corr[i,j]
		# print(rip_corr[i,j])

	allrip_corr = pd.DataFrame(index = interval_mat[index], data = rip_corr[index])
	rip_corr = pd.DataFrame(index = rip_pop.index.values, data = rip_corr, columns = rip_pop.index.values)		
	
	np.fill_diagonal(rip_corr.values, 1.0)
	rip_corr = rip_corr.fillna(0)

	###############################################################################################################
	# POPULATION CORRELATION FOR EACH THETA CYCLE OF REM
	###############################################################################################################
	# compute all time interval for each ep of theta
	interval_mat = np.vstack(nts.TsdFrame(rem_pop).as_units('s').index.values) - nts.TsdFrame(rem_pop).as_units('s').index.values
	rem_corr = np.ones(interval_mat.shape)*np.nan
	# index = np.where(np.logical_and(interval_mat < 3.0, interval_mat >= 0.0))
	# rem_corr = np.eye(interval_mat.shape[0])
	# bad
	tmp = np.zeros_like(rem_corr)
	tmp[np.triu_indices(interval_mat.shape[0], 1)] += 1
	tmp[np.tril_indices(interval_mat.shape[0], 300)] += 1
	index = np.where(tmp == 2)

	for i, j in zip(index[0], index[1]):
		rem_corr[i,j] = scipy.stats.pearsonr(rem_pop.iloc[i].values, rem_pop.iloc[j].values)[0]
		rem_corr[j,i] = rem_corr[i,j]

	allrem_corr = pd.DataFrame(index = interval_mat[index], data = rem_corr[index])
	rem_corr = pd.DataFrame(index = rem_pop.index.values, data = rem_corr, columns = rem_pop.index.values)		
	np.fill_diagonal(rem_corr.values, 1.0)
	rem_corr = rem_corr.fillna(0)

	###############################################################################################################
	# POPULATION CORRELATION FOR EACH THETA CYCLE OF WAKE
	###############################################################################################################
	# compute all time interval for each ep of theta
	interval_mat = np.vstack(nts.TsdFrame(wak_pop).as_units('s').index.values) - nts.TsdFrame(wak_pop).as_units('s').index.values
	wak_corr = np.ones(interval_mat.shape)*np.nan
	# index = np.where(np.logical_and(interval_mat < 3.0, interval_mat >= 0.0))
	# wak_corr = np.eye(interval_mat.shape[0])
	# bad
	tmp = np.zeros_like(wak_corr)
	tmp[np.triu_indices(interval_mat.shape[0], 1)] += 1
	tmp[np.tril_indices(interval_mat.shape[0], 300)] += 1
	index = np.where(tmp == 2)
		
	for i, j in zip(index[0], index[1]):
		wak_corr[i,j] = scipy.stats.pearsonr(wak_pop.iloc[i].values, wak_pop.iloc[j].values)[0]
		wak_corr[j,i] = wak_corr[i,j]

	allwak_corr = pd.DataFrame(index = interval_mat[index], data = wak_corr[index])
	wak_corr = pd.DataFrame(index = wak_pop.index.values, data = wak_corr, columns = wak_pop.index.values)		
	np.fill_diagonal(wak_corr.values, 1.0)
	wak_corr = wak_corr.fillna(0)
	
	###############################################################################################################
	# STORING
	###############################################################################################################
	store 			= pd.HDFStore("/mnt/DataGuillaume/corr_pop_nucleus/"+nuc+"/"+session+".h5")
	store.put('rip_corr', rip_corr)
	store.put('allrip_corr', allrip_corr)
	store.put('wak_corr', wak_corr)
	store.put('allwak_corr', allwak_corr)
	store.put('rem_corr', rem_corr)
	store.put('allrem_corr', allrem_corr)	
	store.close()
	print(time.clock() - start_time, "seconds")
	return time.clock() - start_time


dview = Pool(8)

for n in nucleus:
	print(n)
	a = dview.starmap_async(compute_population_correlation, zip([n]*len(nucleus_session[n]),nucleus_session[n])).get()
# a = compute_population_correlation('AD', nucleus_session['AD'][0])


# ###############################################################################################################
# # PLOT
# ###############################################################################################################
# last = np.max([np.max(allrip_corr[:,0]),np.max(alltheta_corr[:,0])])
# bins = np.arange(0.0, last, 0.2)
# # average rip corr
# index_rip = np.digitize(allrip_corr[:,0], bins)
# mean_ripcorr = np.array([np.mean(allrip_corr[index_rip == i,1]) for i in np.unique(index_rip)[0:30]])
# # average theta corr
# index_theta = np.digitize(alltheta_corr[:,0], bins)
# mean_thetacorr = np.array([np.mean(alltheta_corr[index_theta == i,1]) for i in np.unique(index_theta)[0:30]])


# xt = list(bins[0:30][::-1]*-1.0)+list(bins[0:30])
# ytheta = list(mean_thetacorr[0:30][::-1])+list(mean_thetacorr[0:30])
# yrip = list(mean_ripcorr[0:30][::-1])+list(mean_ripcorr[0:30])
# plot(xt, ytheta, 'o-', label = 'theta')
# plot(xt, yrip, 'o-', label = 'ripple')
# legend()
# xlabel('s')
# ylabel('r')
# show()



