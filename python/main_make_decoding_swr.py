
from pylab import *
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
from numba import jit

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

@jit(nopython=True)
def histo(spk, obins):
	n = len(obins)
	count = np.zeros(n)
	for i in range(n):
		count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
	return count

def decodeHD(tuning_curves, spikes, ep, bin_size = 200, px = None):
	"""
		See : Zhang, 1998, Interpreting Neuronal Population Activity by Reconstruction: Unified Framework With Application to Hippocampal Place Cells
		tuning_curves: pd.DataFrame with angular position as index and columns as neuron
		spikes : dictionnary of spike times
		ep : nts.IntervalSet, the epochs for decoding
		bin_size : in ms (default:200ms)
		px : Occupancy. If None, px is uniform
	"""		
	if len(ep) == 1:
		bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
	else:
		# ep2 = nts.IntervalSet(ep.copy().as_units('ms'))
		# ep2 = ep2.drop_short_intervals(bin_size*2)
		# bins = []
		# for i in ep2.index:
		# 	bins.append(np.arange())
		# bins = np.arange(ep2.start.iloc[0], ep.end.iloc[-1], bin_size)
		print("TODO")
		sys.exit()


	order = tuning_curves.columns.values
	# TODO CHECK MATCH
	
	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = order)
	for k in spike_counts:
		spks = spikes[k].restrict(ep).as_units('ms').index.values
		spike_counts[k], _ = np.histogram(spks, bins)

	tcurves_array = tuning_curves.values
	spike_counts_array = spike_counts.values
	proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))

	part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
	if px is not None:
		part2 = px
	else:
		part2 = np.ones(tuning_curves.shape[0])
	#part2 = np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
	
	for i in range(len(proba_angle)):
		part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
		p = part1 * part2 * part3
		proba_angle[i] = p/p.sum() # Normalization process here

	proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)	
	proba_angle = proba_angle.astype('float')		
	decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')
	return decoded, proba_angle

datatosave = {}
count = {}

decoded_angle = {}

for session in datasets:
# for session in ['Mouse32/Mouse32-140822']:
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	if np.sum(hd_info)>5:		
		############################################################################################################
		# LOADING DATA
		############################################################################################################				
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
		speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)
		hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])
		
		spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0]}
		neurons 		= np.sort(list(spikeshd.keys()))

		print(session, len(neurons))

		bin_size = 30
		nb_bins_tcurves = 61
		std_spike_count = 3
		std_angle = 3
		nan_empty = False

		n_rnd = 1

		# left_bound = np.arange(-1000-bin_size/2, 1000 - bin_size/4,bin_size/4) # 75% overlap
		left_bound = np.arange(-500 - bin_size/3/2, 500 - bin_size/2, bin_size/2) # 50% overlap
		# left_bound = np.arange(-1000-bin_size+3*bin_size/4, 1000 - 3*bin_size/4,3*bin_size/4) # 25% overlap
		obins = np.vstack((left_bound, left_bound+bin_size)).T
		times = obins[:,0]+(np.diff(obins)/2).flatten()
		
		# cutting times between -500 to 500
		# times = times[np.logical_and(times>=-500, times<=500)]

		
		n_rip = len(rip_tsd)		
		

		####################################################################################################################
		# TUNING CURVES
		####################################################################################################################
		position 		= pd.read_csv(data_directory+session+"/"+session.split("/")[1] + ".csv", delimiter = ',', header = None, index_col = [0])
		angle 			= nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')
		tcurves 		= computeAngularTuningCurves(spikeshd, angle, wake_ep, nb_bins = nb_bins_tcurves, frequency = 1/0.0256)

		neurons 		= tcurves.idxmax().sort_values().index.values

		####################################################################################################################
		# SWR
		####################################################################################################################
		# BINNING
		good_ex = (np.array([4644.8144,4924.4720,5244.9392,7222.9480,7780.2968,11110.1888,11292.3240,11874.5688])*1e6).astype('int')
		# tmp = good_ex/1000
		tmp = rip_tsd.index.values/1000
		# tmp = tmp[0:500]
		rates_swr = []
		n_swr = len(tmp)
		for j, t in enumerate(tmp):
			print('swr', j/len(tmp))
			tbins = t + obins
			spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)
			for k in neurons:
				spks = spikes[k].as_units('ms').index.values
				spike_counts[k] = histo(spks, tbins)
				
			spike_counts = spike_counts.rolling(window = 20, win_type='gaussian', center = True, min_periods=1).mean(std=std_spike_count)

			rates_swr.append(spike_counts)

		####################################################################################################################
		# RANDOM
		####################################################################################################################			
		# BINNING
		
		rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[j,'start']+500000, sws_ep.loc[j,'end']+500000, np.maximum(1,n_rnd//len(sws_ep))) for j in sws_ep.index])))		
		rnd_tsd = rnd_tsd
		rates_rnd = []
		tmp3 = rnd_tsd.index.values/1000
		tmp3 = tmp3
		n_rnd = len(tmp3)
		for j, t in enumerate(tmp3):				
			print('rnd', j/len(tmp3))
			tbins = t + obins
			spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)	
			for k in neurons:
				spks = spikes[k].as_units('ms').index.values
				spike_counts[k] = histo(spks, tbins)

			spike_counts = spike_counts.rolling(window = 20, win_type='gaussian', center = True, min_periods=1).mean(std=std_spike_count)
			
			rates_rnd.append(spike_counts)



		rates_swr = pd.concat(rates_swr)
		rates_rnd = pd.concat(rates_rnd)

		####################################################################################################################
		# DECODING
		####################################################################################################################
		tcurves_array = tcurves[neurons].values

		#SWR
		spike_counts_array = rates_swr.values
		proba_angle = np.zeros((spike_counts_array.shape[0], tcurves_array.shape[0]))
		part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
		part2 = np.histogram(angle, np.linspace(0, 2*np.pi, len(tcurves)+1), weights = np.ones_like(angle)/float(len(angle)))[0]
		for j in range(len(proba_angle)):
			part3 = np.prod(tcurves_array**spike_counts_array[j], 1)
			p = part1 * part2 * part3
			proba_angle[j] = p/p.sum() # Normalization process here

		proba_angle_swr = proba_angle.copy()
		proba_angle_swr = proba_angle_swr.reshape(len(rip_tsd), len(times), len(tcurves))

		angswr = [tcurves.index.values[proba_angle_swr[i].argmax(1)] for i in range(len(proba_angle_swr))]
		angswr = pd.DataFrame(index = times, data = np.array(angswr).T)

		decoded_angle[session] = angswr

		nanidx = (rates_swr.sum(1) == 0).values
		nanidx = nanidx.reshape(angswr.shape)
		swrvel = []
		for i in angswr.columns:
			a = np.unwrap(angswr[i].values)
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=std_angle)
			if nan_empty:
				c[nanidx[:,i]] = np.nan
			swrvel.append(np.abs(np.diff(c.values))/bin_size)
		swrvel = np.array(swrvel).T
		# swrvel[np.where(nanidx[0:-1])] = np.nan

		swrvel = pd.DataFrame(index = times[0:-1]+np.diff(times)/2, data = swrvel)
		
		#RND
		spike_counts_array = rates_rnd.values
		proba_angle = np.zeros((spike_counts_array.shape[0], tcurves_array.shape[0]))
		part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
		part2 = np.histogram(angle, np.linspace(0, 2*np.pi, len(tcurves)+1), weights = np.ones_like(angle)/float(len(angle)))[0]
		for j in range(len(proba_angle)):
			part3 = np.prod(tcurves_array**spike_counts_array[j], 1)
			p = part1 * part2 * part3
			proba_angle[j] = p/p.sum() # Normalization process here

		proba_angle_rnd = proba_angle.copy()
		proba_angle_rnd = proba_angle_rnd.reshape(n_rnd, len(times), len(tcurves))

		angrnd = [tcurves.index.values[proba_angle_rnd[i].argmax(1)] for i in range(n_rnd)]
		angrnd = pd.DataFrame(index = times, data = np.array(angrnd).T)
		nanidx = (rates_rnd.sum(1) == 0).values
		nanidx = nanidx.reshape(angrnd.shape)
		rndvel = []
		for i in angrnd.columns:
			a = np.unwrap(angrnd[i].values)
			b = pd.Series(index = times, data = a)
			c = b.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=std_angle)
			if nan_empty:
				c[nanidx[:,i]] = np.nan
			rndvel.append(np.abs(np.diff(c.values))/bin_size)
		rndvel = np.array(rndvel).T
		# rndvel[np.where(nanidx[0:-1])] = np.nan

		rndvel = pd.DataFrame(index = times[0:-1]+np.diff(times)/2, data = rndvel)


		total = (swrvel.mean(1)-rndvel.mean(1))/rndvel.mean(1)

		datatosave[session] = total
		count[session] = np.sum(hd_info)

		# figure()
		# subplot(211)
		# plot(swrvel.mean(1).loc[-500:500])
		# plot(rndvel.mean(1).loc[-500:500])
		# subplot(212)
		# plot(total.loc[-500:500])
		# show()

		# rates_swr = rates_swr.values.reshape(len(rip_tsd), len(times), len(neurons))
		

		# figure()
		# for i, t in enumerate(good_ex[0:3]/1000):
		# 	idx = np.where(int(t*1000) == rip_tsd.index.values)[0][0]
		# 	# spikes
		# 	subplot(5,3,i+1)
		# 	for j,n in enumerate(neurons):
		# 		plot(spikes[n].as_units('ms').loc[t-500:t+500].fillna(j), '|', color = 'black')
		# 	xlim(t-500,t+500)
		# 	ylim(0, len(neurons))
		# 	# spikes count
		# 	subplot(5,3,i+1+3)
		# 	tmp = pd.DataFrame(index = times, data = rates_swr[idx])
		# 	imshow(tmp.loc[-500:500].T, origin = 'lower', aspect = 'auto')

		# 	# matrix decoding
		# 	subplot(5,3,i+1+6)
		# 	tmp = pd.DataFrame(index = times, data = proba_angle_swr[idx])
		# 	imshow(tmp.loc[-500:500].T, extent = (times[0], times[-1], 0, 2*np.pi), origin = 'lower', aspect = 'auto')
		# 	plot(angswr.iloc[:,idx])			

		# 	# angle
		# 	subplot(5,3,i+1+9)
		# 	plot(angswr.iloc[:,idx])			
		# 	ylim(0, 2*np.pi)

		# 	# velocity
		# 	subplot(5,3,i+1+12)
		# 	plot(swrvel.iloc[:,idx])

		# show()
		# subplot(311)
		# imshow(rates_swr.T, aspect = 'auto')
		# subplot(312)
		# plot(angswr)
		# subplot(313)
		# plot(swrvel)
		# xlim(times[0], times[-1])
		# show()

		# sys.exit()

datasets = ['Mouse12/Mouse12-120806', 'Mouse12/Mouse12-120807', 'Mouse12/Mouse12-120808', 'Mouse12/Mouse12-120809', 'Mouse12/Mouse12-120810',
'Mouse17/Mouse17-130130', 'Mouse32/Mouse32-140822']


pref_angle = {}
bins = np.linspace(0, 2*np.pi, 13)
figure()
for i,s in enumerate(datasets):
	a = scipy.stats.circmean(decoded_angle[s].loc[-50:50], 0, 2*np.pi, 0)
	b, c = np.histogram(a, bins)
	pref_angle[s] = pd.Series(index = c[0:-1]+np.diff(c)/2, data = b)
	subplot(3,5,i+1)
	hist(a, bins)
	title(s)

pref_angle = pd.DataFrame.from_dict(pref_angle)

show()

sys.exit()

datatosave = pd.DataFrame.from_dict(datatosave)
count = pd.Series(count)

alldata = {'swrvel':datatosave,'count':count}

# cPickle.dump(alldata, open("../figures/figures_articles_v4/figure1/decodage_bayesian.pickle", 'wb'))


figure()
subplot(131)
plot(datatosave[count[count>10].index], linewidth = 1, color = 'grey')
plot(datatosave[count[count>10].index].mean(1), linewidth = 3)
ylim(-0.3, 0.3)
title(">10 neurons")
subplot(132)
plot(datatosave[count[count<10].index], linewidth = 1, color = 'grey')
plot(datatosave[count[count<10].index].mean(1), linewidth = 3)
ylim(-0.3, 0.3)
title("<10 neurons")
subplot(133)
plot(datatosave, linewidth = 1, color = 'grey')
plot(datatosave.mean(1), linewidth = 3)
ylim(-0.3, 0.3)
title("all neurons")
show()





