

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


data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
sampling_freq = 1250 #Hz

clients = ipyparallel.Client()
print(clients.ids)
dview = clients.direct_view()

# session = datasets[0]

def compute_population_correlation(session):
	data_directory = '/mnt/DataGuillaume/MergedData/'
	import numpy as np	
	import scipy.io			
	import _pickle as cPickle
	import time
	import os, sys	

	sampling_freq = 1250 #Hz
	###############################################################################################################
	# GENERAL INFO
	###############################################################################################################
	generalinfo = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/GeneralInfo.mat')

	###############################################################################################################
	# SHANK INFO
	###############################################################################################################
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]
		else :
			shankStructure[k[0]] = []

	###############################################################################################################
	# SPIKE
	###############################################################################################################
	spikedata = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/SpikeData.mat')
	shank = spikedata['shank']
	shankIndex = np.where(shank == shankStructure['thalamus'])[0]

	nb_channels = len(spikedata['S'][0][0][0])
	spikes = []
	# for i in range(nb_channels):
	for i in shankIndex:	
		spikes.append(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2])

	###############################################################################################################
	# BEHAVIORAL EPOCHS
	###############################################################################################################
	behepochs = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/BehavEpochs.mat')
	sleep_pre_ep = behepochs['sleepPreEp'][0][0]
	sleep_pre_ep = np.hstack([sleep_pre_ep[1],sleep_pre_ep[2]])
	sleep_pre_ep_index = behepochs['sleepPreEpIx'][0]
	sleep_post_ep = behepochs['sleepPostEp'][0][0]
	sleep_post_ep = np.hstack([sleep_post_ep[1],sleep_post_ep[2]])
	sleep_post_ep_index = behepochs['sleepPostEpIx'][0]
	sleep_ep = np.vstack((sleep_pre_ep, sleep_post_ep))
	# merge sleep ep
	corresp = sleep_ep[1:,0].astype('int') == sleep_ep[0:-1,1].astype('int')
	start = sleep_ep[0,0]
	tmp = []
	for i,j in zip(corresp,range(len(corresp))):
		if not i:
			stop = sleep_ep[j,1]
			tmp.append([start, stop])
			start = sleep_ep[j+1,0]
	tmp.append([start, sleep_ep[-1,1]])
	sleep_ep = np.array(tmp)
	# wake ep
	wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
	# restrict linear speed tsd to wake ep
	linear_speed_tsd = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/linspeed.mat')['speed']
	tmp = []
	for e in wake_ep:
		start, stop = e
		for t in linear_speed_tsd:
			if t[0] > start and t[0] < stop:
				tmp.append(t)
	linear_speed_tsd = np.array(tmp)
	index = (linear_speed_tsd[:,1] > 1.0)*1.0
	start = np.where((index[1:]-index[0:-1] == 1))[0]+1
	stop = np.where((index[1:]-index[0:-1] == -1))[0]
	if len(start) == len(stop):
		linear_speed_ep = np.vstack([linear_speed_tsd[start,0],linear_speed_tsd[stop,0]]).transpose()
	else:
		m = np.min([len(start), len(stop)])
		linear_speed_ep = np.vstack([linear_speed_tsd[start[0:m],0],linear_speed_tsd[stop[0:m],0]]).transpose()
	# restrict wake ep to speed > 2cm / s
	wake_ep = linear_speed_ep

	# load SWS EP 
	if session.split("/")[1]+'.sts.SWS' in os.listdir(data_directory+'/'+session+'/'):
		sws = np.genfromtxt(data_directory+'/'+session+'/'+session.split("/")[1]+'.sts.SWS')		
		sws_ep = sws/float(sampling_freq)
	elif session.split("/")[1]+'-states.mat' in os.listdir(data_directory+'/'+session+'/'):
		sws = scipy.io.loadmat(data_directory+'/'+session+'/'+session.split("/")[1]+'-states.mat')['states'][0]
		index = np.logical_or(sws == 2, sws == 3)*1.0
		index = index[1:] - index[0:-1]
		start = np.where(index == 1)[0]+1
		stop = np.where(index == -1)[0]
		if len(start) == len(stop):
			sws_ep = np.hstack((np.vstack(start),
							np.vstack(stop))).astype('float')
		else :
			m = np.min([len(start), len(stop)])
			sws_ep = np.hstack((np.vstack(start[0:m]),
							np.vstack(stop[0:m]))).astype('float')

	# load REM EP
	if session.split("/")[1]+'.sts.REM' in os.listdir(data_directory+'/'+session+'/'):
		rem = np.genfromtxt(data_directory+'/'+session+'/'+session.split("/")[1]+'.sts.REM')
		rem_ep = rem/float(sampling_freq)
	elif session.split("/")[1]+'-states.mat' in os.listdir(data_directory+'/'+session+'/'):
		rem = scipy.io.loadmat(data_directory+'/'+session+'/'+session.split("/")[1]+'-states.mat')['states'][0]
		index = (rem == 5)*1.0
		index = index[1:] - index[0:-1]
		rem_ep = np.hstack((np.vstack(np.where(index == 1)[0]),
							np.vstack(np.where(index == -1)[0]))).astype('float')

	# restrict sws_ep and rem_ep by sleep_ep
	tmp1 = []
	tmp2 = []
	for e in sleep_ep:
		start, stop = e
		for s in sws_ep:
			substart, substop = s
			if substart > start and substop < stop:
				tmp1.append(s)
		for s in rem_ep:
			substart, substop = s
			if substart > start and substop < stop:
				tmp2.append(s)		
	sws_ep = np.array(tmp1)
	rem_ep = np.array(tmp2)


	###############################################################################################################
	# RIPPLES 
	###############################################################################################################
	ripples = np.genfromtxt(data_directory+'/'+session+'/'+session.split("/")[1]+'.sts.RIPPLES')
	# create interval set from ripples
	# 0 : debut
	# 1 : milieu
	# 2 : fin
	# 3 : amplitude nombre de sd au dessus de bruit
	# 4 : frequence instantané
	ripples_ep = ripples[:,(0,2)]
	# restrict ripples_ep to sws_ep
	tmp = []
	for e in sws_ep:
		start, stop = e
		for s in ripples_ep:
			substart, substop = s
			if substart >= start and substop <= stop:
				tmp.append(s)
	ripples_ep = np.array(tmp)
	# create time stamp from ripples
	ripples_tsd = ripples[:,(1,3,4)]
	# restrict rip_tsd to sws_ep
	tmp = []
	for e in sws_ep:
		start, stop = e
		for s in ripples_tsd:
			middle = s[0]
			if middle > start and middle < stop:
				tmp.append(s)
	ripples_tsd = np.array(tmp)
	print("ripples ", len(ripples_ep))

	###############################################################################################################
	# THETA REM
	###############################################################################################################
	thetaInfo = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/ThetaInfo.mat')
	creux_tsd = thetaInfo['thetaTrghs'][0][0][2].flatten()
	# TODO CHECK FOR SAMPLING FREQUENCY WITH ADRIEN

	# restrict creux tsd to rem_ep
	tmp = []
	for e in rem_ep:
		start, stop = e
		tmp.append(creux_tsd[np.logical_and(creux_tsd >= start, creux_tsd <= stop)])	
	creux_rem_tsd = np.array(tmp)
	theta_rem_ep = []
	# restrict theta_rem_ep to more than 20 cycles
	for i in range(len(creux_rem_tsd)):
		tmp = np.vstack((creux_rem_tsd[i][0:-1],creux_rem_tsd[i][1:])).transpose()
		if len(tmp) > 20:
			theta_rem_ep.append(tmp)

	theta_rem_ep = np.array(theta_rem_ep)
	print("rem ", len(theta_rem_ep))

	###############################################################################################################
	# THETA WAKE
	###############################################################################################################
	# restrict creux tsd to wake_ep
	tmp = []
	for e in wake_ep:
		start, stop = e
		tmp.append(creux_tsd[np.logical_and(creux_tsd >= start, creux_tsd <= stop)])	
	creux_wake_tsd = np.array(tmp)
	theta_wake_ep = []
	# restrict theta_wake_ep to more than 50 cycles
	for i in range(len(creux_wake_tsd)):
		tmp = np.vstack((creux_wake_tsd[i][0:-1],creux_wake_tsd[i][1:])).transpose()
		if len(tmp) > 60:
			theta_wake_ep.append(tmp)

	theta_wake_ep = np.array(theta_wake_ep)
	print("wake ", len(theta_wake_ep))

	if len(theta_wake_ep) and len(theta_rem_ep):
		start_time = time.clock()
		###############################################################################################################
		# POPULATION CORRELATION FOR EACH RIPPLES
		###############################################################################################################
		n_ripples = len(ripples_ep)
		n_neurons = len(spikes)
		# population activity vector
		pop = np.zeros( (n_ripples, n_neurons) )

		for i in range(n_ripples):
			start = ripples_tsd[i,0] - 0.05
			stop = ripples_tsd[i,0] + 0.05
			for j in range(n_neurons):
				index = np.logical_and(spikes[j] >= start, spikes[j] <= stop)
				pop[i,j] = float(index.sum())
		# normalize by each ripple length
		pop = pop/np.vstack(ripples_ep[:,1]-ripples_ep[:,0])
		# divide by mean firing rate
		pop = pop / (pop.mean(0)+1e-12)

		rip_corr = np.zeros(n_ripples-1)
		#matrix of distance between ripples
		interval_mat = np.vstack(ripples_tsd[:,0]) - ripples_tsd[:,0]
		interval_index = np.logical_and(interval_mat < 5.0, interval_mat > 0.0)
		corr = np.ones(interval_mat.shape)
		index = np.tril_indices(n_ripples,-1)
		for i, j in zip(index[0], index[1]):	
			corr[i,j] = scipy.stats.pearsonr(pop[i], pop[j])[0]
			corr[j,i] = corr[i,j]

		rip_corr = (interval_mat, corr)
		allrip_corr = np.vstack((rip_corr[0][index], rip_corr[1][index])).transpose()

		###############################################################################################################
		# POPULATION CORRELATION FOR EACH THETA CYCLE OF REM
		###############################################################################################################
		n_theta = [len(i) for i in theta_rem_ep]
		n_neurons = len(spikes)
		# population activity vector
		pop = []
		for i in range(len(theta_rem_ep)):
			subpop = np.zeros( (n_theta[i], n_neurons) )
			for j in range(n_theta[i]):
				start = theta_rem_ep[i][j][0]
				stop  = theta_rem_ep[i][j][1]
				for k in range(n_neurons):
					index = np.logical_and(spikes[k] >= start, spikes[k] <= stop)
					subpop[j,k] = float(index.sum())
			# normalized by each theta length
			subpop = subpop/np.vstack(theta_rem_ep[i][:,1] - theta_rem_ep[i][:,0])
			subpop = subpop/(subpop.mean(0)+1e-12)
			pop.append(subpop)
				

		# correlation
		# compute all time interval for each ep of theta
		theta_rem_corr = []
		alltheta_rem_corr = []
		for i in range(len(theta_rem_ep)):
			ep = theta_rem_ep[i]
			# matrix of distance between creux
			interval_mat = np.vstack(ep[:,0]) - ep[:,0]
			interval_index = np.logical_and(interval_mat < 5.0, interval_mat > 0.0)
			corr = np.zeros(interval_mat.shape)
			subpop = pop[i]
			index = np.tril_indices(n_theta[i],-1)
			for j, k in zip(index[0], index[1]):	
				corr[j,k] = scipy.stats.pearsonr(subpop[j], subpop[k])[0]
				corr[k,j] = corr[j,k]	

			# theta_corr.append(np.vstack((interval_mat[interval_index],corr[interval_index])).transpose())
			theta_rem_corr.append((interval_mat,corr))
			alltheta_rem_corr.append(np.vstack((theta_rem_corr[i][0][index], theta_rem_corr[i][1][index])).transpose())

		alltheta_rem_corr = np.vstack(alltheta_rem_corr)

		###############################################################################################################
		# POPULATION CORRELATION FOR EACH THETA CYCLE OF WAKE
		###############################################################################################################
		n_theta = [len(i) for i in theta_wake_ep]
		n_neurons = len(spikes)
		# population activity vector
		pop = []
		for i in range(len(theta_wake_ep)):
			subpop = np.zeros( (n_theta[i], n_neurons) )
			for j in range(n_theta[i]):
				start = theta_wake_ep[i][j][0]
				stop  = theta_wake_ep[i][j][1]
				for k in range(n_neurons):
					index = np.logical_and(spikes[k] >= start, spikes[k] <= stop)
					subpop[j,k] = float(index.sum())
			# normalized by each theta length
			subpop = subpop/np.vstack(theta_wake_ep[i][:,1] - theta_wake_ep[i][:,0])
			subpop = subpop/(subpop.mean(0)+1e-12)
			pop.append(subpop)
				

		# correlation
		# compute all time interval for each ep of theta
		theta_wake_corr = []
		alltheta_wake_corr = []
		for i in range(len(theta_wake_ep)):
			ep = theta_wake_ep[i]
			# matrix of distance between creux
			interval_mat = np.vstack(ep[:,0]) - ep[:,0]
			interval_index = np.logical_and(interval_mat < 5.0, interval_mat > 0.0)
			corr = np.zeros(interval_mat.shape)
			subpop = pop[i]
			index = np.tril_indices(n_theta[i],-1)
			for j, k in zip(index[0], index[1]):	
				corr[j,k] = scipy.stats.pearsonr(subpop[j], subpop[k])[0]
				corr[k,j] = corr[j,k]	

			# theta_corr.append(np.vstack((interval_mat[interval_index],corr[interval_index])).transpose())
			theta_wake_corr.append((interval_mat,corr))
			alltheta_wake_corr.append(np.vstack((theta_wake_corr[i][0][index], theta_wake_corr[i][1][index])).transpose())

		alltheta_wake_corr = np.vstack(alltheta_wake_corr)

		print(time.clock() - start_time, "seconds")
		print('\n')

		tosave =	 {	'rip_corr':rip_corr,
						'theta_wake_corr':theta_wake_corr,
						'theta_rem_corr':theta_rem_corr
				}
		cPickle.dump(tosave, open('../data/corr_pop/'+session.split("/")[1]+".pickle", 'wb'))

		return session

a = dview.map_sync(compute_population_correlation, datasets)
	


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



