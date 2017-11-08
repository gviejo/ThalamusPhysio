

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

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

bins2 = np.arange(-1012.5,1025,25)*1000
tsmax = {i:pd.DataFrame(columns = ['pre', 'pos']) for i in range(2)}

# clients = ipyparallel.Client()
# print(clients.ids)
# dview = clients.direct_view()

# def compute_pop_pca_shank(session):
# 	data_directory = '/mnt/DataGuillaume/MergedData/'
# 	import numpy as np	
# 	import scipy.io	
# 	import scipy.stats		
# 	import _pickle as cPickle
# 	import time
# 	import os, sys	
# 	import neuroseries as nts
# 	from functions import loadShankStructure, loadSpikeData, loadEpoch, loadSpeed, loadXML, loadRipples, loadLFP, downsample, getPeaksandTroughs, butter_bandpass_filter, gaussFilt
# 	import pandas as pd	

# 	bins2 = np.arange(-1012.5,1025,25)*1000	
# 	tsmax = {i:pd.DataFrame(columns = ['pre', 'pos']) for i in range(2)}

for session in datasets:
# for session in ['Mouse20/Mouse20-130603']:

	start_time = time.clock()
	print(session)
	generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
	shankStructure 	= loadShankStructure(generalinfo)	
	if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
	else:
		hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
	spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
	wake_ep 		= loadEpoch(data_directory+session, 'wake')
	sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
	sws_ep 			= loadEpoch(data_directory+session, 'sws')
	rem_ep 			= loadEpoch(data_directory+session, 'rem')
	sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
	sws_ep 			= sleep_ep.intersect(sws_ep)	
	rem_ep 			= sleep_ep.intersect(rem_ep)
	speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)	
	speed_ep 		= nts.IntervalSet(speed[speed>2.5].index.values[0:-1], speed[speed>2.5].index.values[1:]).drop_long_intervals(26000).merge_close_intervals(50000)
	wake_ep 		= wake_ep.intersect(speed_ep).drop_short_intervals(3000000)	
	n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')	
	rip_ep,rip_tsd 	= loadRipples(data_directory+session)
	# rip_ep			= sws_ep.intersect(rip_ep)	
	# rip_tsd 		= rip_tsd.restrict(rip_ep)
	hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
	hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])		
	shankIndex 		= np.array([shank[i] for i in spikes.keys()]).flatten()

	if len(sleep_ep) > 1:
		store 			= pd.HDFStore("../data/population_activity_100ms/"+session.split("/")[1]+".h5")	
		all_pop 		= store['allwake']
		store.close()

		store 			= pd.HDFStore("../data/population_activity_25ms/"+session.split("/")[1]+".h5")	
		pre_pop 		= store['presleep']
		pos_pop 		= store['postsleep']
		store.close()		



		def compute_eigen(popwak):
			popwak = popwak - popwak.mean(0)
			popwak = popwak / (popwak.std(0)+1e-8)
			from sklearn.decomposition import PCA	
			pca = PCA(n_components = popwak.shape[1])
			xy = pca.fit_transform(popwak.values)
			index = pca.explained_variance_ > (1 + np.sqrt(1/(popwak.shape[0]/popwak.shape[1])))**2.0
			eigen = pca.components_[index]			
			return eigen

		def compute_score(ep_pop, eigen):
			ep_pop = ep_pop - ep_pop.mean(0)
			ep_pop = ep_pop / (ep_pop.std(0)+1e-8)			
			a = ep_pop.values
			score = np.zeros(len(ep_pop))
			for i in range(len(eigen)):
				score += np.dot(a, eigen[i])**2.0 - np.dot(a**2.0, eigen[i]**2.0)
			score = nts.Tsd(t = ep_pop.index.values, d = score)
			return score

		def compute_rip_score(tsd, score, bins):								
			times = np.floor(((bins[0:-1] + (bins[1] - bins[0])/2)/1000)).astype('int')			
			rip_score = pd.DataFrame(index = times, columns = [])
			for r,i in zip(tsd.index.values,range(len(tsd))):
				xbins = (bins + r).astype('int')
				y = score.groupby(pd.cut(score.index.values, bins=xbins, labels = times)).mean()
				if ~y.isnull().any():
					rip_score[r] = y

			return rip_score
	
		pre_ep 			= nts.IntervalSet(sleep_ep['start'][0], sleep_ep['end'][0])
		post_ep 		= nts.IntervalSet(sleep_ep['start'][1], sleep_ep['end'][1])
		# pre_sws_ep 		= sws_ep.intersect(pre_ep)
		# pos_sws_ep 		= sws_ep.intersect(post_ep)
		# pre_sws_ep 		= pre_sws_ep.intersect(nts.IntervalSet(pre_sws_ep['end'].iloc[-1] - 60*60*1000*1000, pre_sws_ep['end'].iloc[-1]))
		# pos_sws_ep 		= pos_sws_ep.intersect(nts.IntervalSet(pos_sws_ep['start'].iloc[0], pos_sws_ep['end'].iloc[0] + 60*60*1000*1000))

		
		# if pre_sws_ep.tot_length()/1000/1000/60 > 30.0 and pos_sws_ep.tot_length()/1000/1000/60 > 30.0:			
		if pre_ep.tot_length()/1000/1000/60 > 3.0 and post_ep.tot_length()/1000/1000/60 > 3.0:			
			for hd in range(2):
				index = np.where(hd_info_neuron == hd)[0]
				allpop = all_pop[index].copy()			
				if allpop.shape[1] and allpop.shape[1] > 2:	
					eigen = compute_eigen(allpop)
					###############################################################################################################
					# SHANK LOOP
					###############################################################################################################				
					for shank in np.unique(shankIndex):
						index2 = np.where((shankIndex[index] == shank))[0]
						prepop = nts.TsdFrame(pre_pop[index2].copy())
						pospop = nts.TsdFrame(pos_pop[index2].copy())
						
						pre_score = compute_score(prepop.copy(), eigen[:,index2])
						pos_score = compute_score(pospop.copy(), eigen[:,index2])
						
						prerip_score = compute_rip_score(rip_tsd.restrict(pre_ep), pre_score, bins2)
						posrip_score = compute_rip_score(rip_tsd.restrict(post_ep), pos_score, bins2)

						a = pd.DataFrame(index = prerip_score.index.values, data = gaussFilt(prerip_score.mean(1).values, (5,)))
						b = pd.DataFrame(index = posrip_score.index.values, data = gaussFilt(posrip_score.mean(1).values, (5,)))

						tmp = pd.DataFrame(data = [[a.loc[-500:500].idxmax()[0], b.loc[-500:500].idxmax()[0]]], columns = ['pre', 'pos'])
						tsmax[hd] = tsmax[hd].append(tmp, ignore_index = True)
						sys.exit()
						# idx = session.split("/")[1] + "_" + str(shank)
						# premeanscore[hd][idx] = prerip_score.mean(0)
						# posmeanscore[hd][idx] = posrip_score.mean(0)

# 	return tsmax

# a = dview.map_sync(compute_pop_pca_shank, datasets)


tsmax = {i:pd.DataFrame(columns = ['pre', 'pos']) for i in range(2)}
for i in range(len(a)):
	for hd in range(2):
		tsmax[hd] = tsmax[hd].append(a[i][hd], ignore_index = True)


from pylab import *
plot(tsmax[0]['pos'], np.ones(len(tsmax[0]['pos'])), 'o')
plot(tsmax[0]['pos'].mean(), [1], '|', markersize = 10)
plot(tsmax[1]['pos'], np.zeros(len(tsmax[1]['pos'])), 'o')
plot(tsmax[1]['pos'].mean(), [0], '|', markersize = 10)
show()

sys.exit()

preshank = {i:pd.DataFrame() for i in range(2)}
posshank = {i:pd.DataFrame() for i in range(2)}
for i in range(len(a)):
	for j in range(2):
		preshank[j][a[i][0][j].columns] = a[i][0][j]
		posshank[j][a[i][1][j].columns] = a[i][1][j]


tspre = pd.DataFrame(columns = range(2))
tspos = pd.DataFrame(columns = range(2))
for i in range(2):
	pre = preshank[i]
	pos = posshank[i]
	pre = pre - pre.mean(0)	
	pos = pos - pos.mean(0)	
	pre = pre / pre.std(0)	
	pos = pos / pos.std(0)
	pre = pre.loc[-500:500]			
	pos = pos.loc[-500:500]
	# pre = pre.loc[:,pre.max()>1]
	# pos = pos.loc[:,pos.max()>1]
	tspre[i] = pre.idxmax()
	tspos[i] = pos.idxmax()

from pylab import *
plot(tspos[0], np.ones(len(tspos[0])), 'o')
plot(tspos[0].mean(), [1], '|')
plot(tspos[1], np.zeros(len(tspos[1])), 'o')
plot(tspos[1].mean(), [0], '|')



sys.exit()

figure()
subplot(121)
plot(premeanscore)
subplot(122)
plot(posmeanscore)
show()

sys.exit()
store = pd.HDFStore("../figures/figures_articles/figure3/pca_analysis.h5")
for i in range(2):	
	store.put(str(i)+'pre_rip', premeanscore[i]['rip'])
	store.put(str(i)+'pos_rip', posmeanscore[i]['rip'])
	store.put(str(i)+'pre_rem', premeanscore[i]['rem'])
	store.put(str(i)+'pos_rem', posmeanscore[i]['rem'])
store.close()



# a = dview.map_sync(compute_population_correlation, datasets[0:15])
# for i in range(len(a)):
# 	if type(a[i]) is dict:
# 		s = list(a[i].keys())[0]
# 		premeanscore.loc[s] = a[i][s]['pre']
# 		posmeanscore.loc[s] = a[i][s]['pos']

from pylab import *
titles = ['non hd', 'hd']
figure()
for i in range(2):
	subplot(1,3,i+1)
	
	times = premeanscore[i]['rip'].columns.values
	# for s in premeanscore[i]['rip'].index.values:		
	# 	plot(times, premeanscore[i]['rip'].loc[s].values, linewidth = 0.3, color = 'blue')
	# 	plot(times, posmeanscore[i]['rip'].loc[s].values, linewidth = 0.3, color = 'red')
	plot(times, gaussFilt(premeanscore[i]['rip'].mean(0).values, (1,)), label = 'pre', color = 'blue', linewidth = 2)
	plot(times, gaussFilt(posmeanscore[i]['rip'].mean(0).values, (1,)),label = 'post', color = 'red', linewidth = 2)
	legend()
	title(titles[i])
subplot(1,3,3)
bar([1,2], [premeanscore[0]['rem'].mean(0)['mean'], premeanscore[1]['rem'].mean(0)['mean']])
bar([3,4], [posmeanscore[0]['rem'].mean(0)['mean'], posmeanscore[1]['rem'].mean(0)['mean']])
xticks([1,2], ['non hd', 'hd'])
xticks([3,4], ['non hd', 'hd'])


			
show()

figure()
subplot(121)
times = premeanscore[0]['rip'].columns.values
for s in premeanscore[0]['rip'].index.values:		
	print(s)
	plot(times, premeanscore[0]['rip'].loc[s].values, linewidth = 1, color = 'blue')
plot(premeanscore[0]['rip'].mean(0))
subplot(122)	
for s in posmeanscore[0]['rip'].index.values:		
	plot(times, posmeanscore[0]['rip'].loc[s].values, linewidth = 1, color = 'red')
plot(posmeanscore[0]['rip'].mean(0))
show()



