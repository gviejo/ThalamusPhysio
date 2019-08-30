#!/usr/bin/env python
'''
    File name: main_ripp_mod.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2


'''
import sys
import numpy as np
import pandas as pd
import scipy.io
from functions import *
# from pylab import *
from multiprocessing import Pool
import os
import neuroseries as nts
from time import time
from pylab import *
from functions import quickBin
from sklearn.utils.random import sample_without_replacement
from numba import jit

# clients = ipyparallel.Client()	
# dview = clients.direct_view()
dview = Pool()

def sample_comb(dims, nsamp):
    idx = sample_without_replacement(np.prod(dims), nsamp)
    return np.vstack(np.unravel_index(idx, dims)).T

@jit(nopython=True)
def quickBin(spikelist, ts, bins, index):
	rates = np.zeros((len(ts), len(bins)-1, len(index)))
	for i, t in enumerate(ts):
		tbins = t + bins
		for j in range(len(spikelist)):					
			a, _ = np.histogram(spikelist[j], tbins)
			rates[i,:,j] = a
	return rates

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
data = {}

datasets = [s for s in datasets if 'Mouse17' in s]
# sys.exit()

# for session in datasets[9:]:
# for session in ['Mouse17/Mouse17-130205']:
for session in ['Mouse32/Mouse32-140822']:
# for session in ['Mouse17/Mouse17-130216']:
		print(session)
		generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
		shankStructure 	= loadShankStructure(generalinfo)
		if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
			hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
		else:
			hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
		spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
		n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')	
		wake_ep 		= loadEpoch(data_directory+session, 'wake')
		sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
		sws_ep 			= loadEpoch(data_directory+session, 'sws')
		rem_ep 			= loadEpoch(data_directory+session, 'rem')
		sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
		sws_ep 			= sleep_ep.intersect(sws_ep)	
		rem_ep 			= sleep_ep.intersect(rem_ep)
		rip_ep,rip_tsd 	= loadRipples(data_directory+session)
		rip_ep			= sws_ep.intersect(rip_ep)	
		rip_tsd 		= rip_tsd.restrict(sws_ep)	
		lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
		hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])

		down_ep, up_ep = loadUpDown(data_directory+session)

		rip_tsd = rip_tsd.restrict(up_ep)

		# if np.sum(hd_info_neuron)>2 and np.sum(hd_info_neuron==0)>2:			
		####################################################################################################################
		# binning data
		####################################################################################################################	
		spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
		spikesnohd 		= {k:spikes[k] for k in np.where(hd_info_neuron==0)[0] if k not in []}
		hdneurons		= np.sort(list(spikeshd.keys()))
		nohdneurons		= np.sort(list(spikesnohd.keys()))

		####################################################################################################################
		# MEAN AND STD SWS
		####################################################################################################################
		bin_size = 10
		# mean and standard deviation during SWS
		mean_sws = pd.DataFrame(index = np.sort(list(spikes.keys())), columns = ['mean', 'std'])
		for n in spikes.keys():
			r = []
			for e in sws_ep.index:
				bins = np.arange(sws_ep.loc[e,'start'], sws_ep.loc[e,'end'], bin_size*1e3)
				a, _ = np.histogram(spikes[n].restrict(sws_ep.loc[[e]]).index.values, bins)
				r.append(a)
			r = np.hstack(r)
			r = r / float(bin_size)
			mean_sws.loc[n,'mean']= r.mean()
			mean_sws.loc[n,'std']= r.std()


		####################################################################################################################
		# BIN SWR
		####################################################################################################################		
		bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2
		# bins = np.arange(0, 200+2*bin_size, bin_size) - 100
			
		ts = rip_tsd.as_units('ms').index.values
		
		rates = quickBin([spikeshd[j].as_units('ms').index.values for j in hdneurons], ts, bins, hdneurons)

		rates = rates/float(bin_size)
		m = mean_sws.loc[hdneurons,'mean'].values.astype('float')
		s = mean_sws.loc[hdneurons,'std'].values.astype('float')
		rates2 = (rates - m) / (s+1)

		times = bins[0:-1] + np.diff(bins)/2
			
		# no z-scoring
		angle1 = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))

		for i, r in enumerate(rates):
			tmp = np.sqrt(np.power(r, 2).sum(1))
			denom = tmp[0:-1] * tmp[1:]
			num = np.sum(r[0:-1]*r[1:], 1)
			angle1[i] = num/(denom+1)		

		# zscoring
		angle2 = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))

		for i, r in enumerate(rates2):
			tmp = np.sqrt(np.power(r, 2).sum(1))
			denom = tmp[0:-1] * tmp[1:]
			num = np.sum(r[0:-1]*r[1:], 1)
			angle2[i] = num/(denom+1)


		figure()
		subplot(211)
		plot(angle1.mean(1), label = session)
		legend()
		title("Pas de zscore")
		subplot(212)
		plot(angle2.mean(1), label = session)
		title("Zscored")
		legend()
		show()

		sys.exit()


		# UP/DOWN

		up_tsd = nts.Ts(t = up_ep['start'].values)

		d = np.vstack(rip_tsd.index.values) - up_tsd.index.values
		d[d<0] = np.max(d)
		idx = np.argmin(d, 1)
		up_time = up_tsd.index.values[idx]

		interval = rip_tsd.index.values - up_time

		a = angle2.iloc[:,np.argsort(interval)]

		groups = []
		for idx in np.array_split(np.arange(a.shape[1]),3):
			groups.append(a[idx].mean(1))
		groups = pd.concat(groups, 1)



		subplot(211)
		imshow(a.T, aspect = 'auto', cmap = 'jet')
		xlabel("Time from SWRs")
		title("scalar product")
		ylabel("Time from Up")
		subplot(212)
		plot(groups)
		legend()
		xlabel("Time from SWRs")




		# pearson without z-scoring
		pearson1 = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)), data = 0)

		for i, r in enumerate(rates[0:1000]):		
			for j in range(len(r)-1):
				c = scipy.stats.pearsonr(r[j],r[j+1])[0]
				if not np.isnan(c):
					pearson1.iloc[j,i] = c



		# pearson with z-scoring
		pearson2 = pd.DataFrame(index = times[0:-1], columns = np.arange(len(rip_tsd)))

		tmp = []
		for i, r in enumerate(rates2):
			tmp.append(np.corrcoef(r))
			pearson2[i] = np.diag(np.corrcoef(r), 1)





		sys.exit()
		
		# choosing a million random points during sws
		n_ex = 10000
		rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, n_ex//len(sws_ep)) for i in sws_ep.index])))
		ts = rnd_tsd.as_units('ms').index.values		
		args2 = []
		for case, (dspike, _, _ , index) in zip(order, args):
			args2.append((dspike, ts, bins, index))

		shuffle = dview.starmap(quickBin, args2)

		r = allrates['nohd']
		shuffle = shuffle[0]
		a = r[:,1,:]
		b = shuffle[:,1,:]
		d = np.vstack((a, b))
		from sklearn.manifold import MDS
		X = MDS(n_components = 2, metric = True).fit_transform(d)
		c = X[0:len(a),:]
		d = X[len(a):,:]
		scatter(c[:,0], c[:,0])
		scatter(c[:,1], c[:,1])
		show()


		sys.exit()

		idx = noangle.loc[-50].sort_values().index.values[::-1]

		# for i in range(20):
		for i in range(0,10):
			print(i)
			t = rip_tsd.index.values[idx[i]]
			ex_ep = nts.IntervalSet(start = t - 5e5, end = t + 5e5)

			figure()
			ax=subplot(311)
			plot(lfp_hpc.restrict(ex_ep))			
			axvline(t)

			subplot(312, sharex=ax)
			ct = 0
			# for n in spikeshd:
			# 	plot(spikes[n].restrict(ex_ep).fillna(ct), '|', color = 'red')
			# 	ct+=1
			for n in spikesnohd:
				plot(spikes[n].restrict(ex_ep).fillna(ct), '|', color = 'black')
				ct+=1
			ylim(-1,ct+1)

			subplot(313,sharex=ax)
			# tmp = hdangle[idx[i]].loc[-500:500]			
			# plot(tmp.index.values*1000+t, tmp.values, '-', color = 'red', linewidth = 0.5)
			tmp = noangle[idx[i]].loc[-500:500]
			plot(tmp.index.values*1000+t, tmp.values, '-', color = 'black', linewidth = 0.5)

			show()



sys.exit()

# SAVING GOOD EXEMPLE
i = 9
t = rip_tsd.index.values[idx[i]]
ex_ep = nts.IntervalSet(start = t - 5e5, end = t + 5e5)
data = { 
	'ex_tsd':pd.Series(index=[t],data=np.nan),
	'ex_ep':ex_ep,
	'lfp':lfp_hpc.restrict(ex_ep).as_series(),
	# 'spikeshd':{n:spikeshd[n].restrict(ex_ep) for n in spikeshd},
	'spikesnohd':{n:spikesnohd[n].restrict(ex_ep) for n in spikesnohd},
	# 'hdangle':angle['hd'][idx[i]],
	'noangle':angle['nohd'][idx[i]],
	'session':session
}

import _pickle as cPickle
cPickle.dump(data, open('../figures/figures_articles_v4/figure2/Ex2.pickle', 'wb'))