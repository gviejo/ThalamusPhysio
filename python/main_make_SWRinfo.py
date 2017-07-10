import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

session = datasets[0]

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
for i in range(nb_channels):
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

sws = scipy.io.loadmat(data_directory+'/'+session+'/'+session.split("/")[1]+'-states.mat')['states'][0]
# need to create an interval set from sws.
# sws = {2,3}
# rem = 5
# is = 4
# wake = 1
index = np.logical_or(sws == 2, sws == 3)*1.0
index = index[1:] - index[0:-1]
sws_ep = np.hstack((np.vstack(np.where(index == 1)[0]),
					np.vstack(np.where(index == -1)[0])))

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

# restrict sleep_ep by sws_ep
tmp = []
for e in sleep_ep:
	start, stop = e
	for s in sws_ep:
		substart, substop = s
		if substart > start and substop < stop:
			tmp.append(s)
sws_ep = np.array(tmp)

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
