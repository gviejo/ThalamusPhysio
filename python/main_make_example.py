import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
# from pylab import *
import ipyparallel
import os,sys

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {}

session = 'Mouse12/Mouse12-120810'


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

# refine
shankIndex = np.sort(np.array([np.where(shank == 5)[0][7],
				np.where(shank == 5)[0][3],
				# np.where(shank == 5)[0][8],
				np.where(shank == 5)[0][11],
				# np.where(shank == 5)[0][14],
				# np.where(shank == 6)[0][0],
				np.where(shank == 6)[0][1]]))


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
index = (linear_speed_tsd[:,1] > 2.0)*1.0
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

###############################################################################################################
# JITTERED CROSS-CORRELATION FOR EACH NEURON
###############################################################################################################

rip_tsd = ripples_tsd[:,0]

for i in range(len(spikes)):
	spikes[i] = spikes[i].flatten()
# restrict spikes to sws_ep
tmp = []
for i in range(len(spikes)):
	tmp.append([])
	for e in sws_ep:
		start, stop = e
		for s in spikes[i]:
			if s > start and s < stop:
				tmp[-1].append(s)

for i in range(len(tmp)):
	tmp[i] = np.array(tmp[i])
spikes_sws = np.array(tmp)

Hcorr = []
Hjiter = []

for i in range(len(spikes_sws)):
	spike_tsd   = spikes_sws[i]
	bin_size 	= 0.005 # ms 	
	nb_bins 	= 200
	confInt 	= 0.95
	nb_iter 	= 1000
	jitter  	= 150 # ms			

	
	C = crossCorr(rip_tsd, spike_tsd, bin_size, nb_bins)

	jitter_count = np.zeros((nb_iter,nb_bins+1))
	for j in range(nb_iter):		
		print(i,j)
		spike_tsd_jittered = spike_tsd+np.random.uniform(-jitter,+jitter,len(spike_tsd))
		jitter_count[j,:] = crossCorr(rip_tsd, spike_tsd_jittered, bin_size, nb_bins)
	J = jitter_count.mean(0)		

	Hcorr.append(C)
	Hjiter.append(J)

		
Hcorr = np.array(Hcorr)
Hjiter = np.array(Hjiter)

Z = (Hcorr - Hjiter) / np.vstack(np.std(Hcorr, 1))

###############################################################################################################
# THETA MODULATION
###############################################################################################################
thetaInfo = scipy.io.loadmat(data_directory+session+'/Analysis/ThetaInfo.mat')
thMod = thetaInfo['thetaModWake'][shankIndex,:] # all neurons vs 

thetaphase = []
for i in shankIndex:
	spike_phase = np.hstack([thetaInfo['thetaPh'][i][0][0][0][2],thetaInfo['thetaPh'][i][0][0][0][3]])
	spike_phase[:,1] += np.pi
	# spike_phase[:,1] = np.mod(spike_phase[:,1], 2*np.pi)
	# restrict to rem
	tmp = []
	for e in wake_ep:
		start, stop = e
		for t in spike_phase:
			if t[0] > start and t[0] < stop:
				tmp.append(t)
	spike_phase = np.array(tmp)

	thetaphase.append(spike_phase)

nbins = 30
bins = np.linspace(0, 2*np.pi+0.0001, nbins)

spikecount = []

for i in range(len(thetaphase)):
	index = np.digitize(thetaphase[i][:,1], bins)
	tmp = [np.sum(index == j) for j in np.arange(1, nbins)]
	tmp = np.array(tmp)
	# normalize
	tmp = tmp - np.min(tmp)
	tmp = tmp / np.max(tmp)
	spikecount.append(tmp)


spikecount = np.array(spikecount)

# mod theta
a = np.argsort(thMod[:,2])


# TO SAVE
tosave = {
'Hcorr'		:	Hcorr	,
'Hjitt'		:	Hjiter	,
'Z'			:	Z		,
'theta'		:	spikecount,
'thmod'		: 	thMod[:,0]
}

import _pickle as cPickle
cPickle.dump(tosave, open('../data/to_plot_examples.pickle', 'wb'))