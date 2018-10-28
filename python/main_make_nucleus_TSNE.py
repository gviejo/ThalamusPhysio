	#!/usr/bin/env python

'''
	File name: main_make_mucleus_TSNE.py
	Author: Guillaume Viejo
	Date created: 28/09/2017    
	Python Version: 3.5.2


'''

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import neuroseries as nts
import sys
import scipy.ndimage.filters as filters
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from functools import reduce
from multiprocessing import Pool
import h5py as hd
from scipy.stats import zscore
from sklearn.manifold import TSNE, SpectralEmbedding
from skimage import filters

###############################################################################################################
# LOADING DATA
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (20,)).transpose())
swr = swr.drop(swr.columns[swr.isnull().any()].values, axis = 1)

swr_mod = swr.loc[0]
# waveforms loading
# waveforms = loadWaveforms(data_directory, datasets)
# waveforms.to_hdf("../data/waveforms.h5", mode = 'w', format = 'fixed')
# waveforms = pd.read_hdf("../data/waveforms.h5")
###############################################################################################################
# THETA MODULATION
###############################################################################################################
theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta 					= pd.DataFrame(	index = theta_ses['rem'], 
									columns = ['phase', 'pvalue', 'kappa'],
									data = theta_mod['rem'])


##############################################################################################################
# which hd neurons + burstiness + ISI
##############################################################################################################
hd_index = []
burst = []
index = []
for m in ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']:
	sessions 		= [n.split("/")[1] for n in datasets if m in n]
	for s in sessions:
		generalinfo 		= scipy.io.loadmat(data_directory+m+"/"+s+'/Analysis/GeneralInfo.mat')		
		shankStructure 		= loadShankStructure(generalinfo)
		spikes,shank		= loadSpikeData(data_directory+m+"/"+s+'/Analysis/SpikeData.mat', shankStructure['thalamus'])						
		wake_ep 		= loadEpoch(data_directory+m+'/'+s, 'wake')
		sleep_ep 		= loadEpoch(data_directory+m+'/'+s, 'sleep')
		sws_ep 			= loadEpoch(data_directory+m+'/'+s, 'sws')
		rem_ep 			= loadEpoch(data_directory+m+'/'+s, 'rem')
		sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
		sws_ep 			= sleep_ep.intersect(sws_ep)	
		rem_ep 			= sleep_ep.intersect(rem_ep)
		# hd
		hd_info 			= scipy.io.loadmat(data_directory+m+'/'+s+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		hd_info_neuron		= np.array([hd_info[n] for n in spikes.keys()])		
		for n in np.where(hd_info[list(spikes.keys())])[0]:
			hd_index.append(s+'_'+str(n))
# 		# # burst
# 		for n in spikes.keys():
# 			index.append(s+'_'+str(n))
# 			burst.append([computeBurstiness(spikes[n], ep) for ep in [wake_ep, rem_ep, sws_ep]])
				

# burst = pd.DataFrame(data=np.array(burst), index=index, columns=['wak', 'rem', 'sws'])
# burst.to_hdf("/mnt/DataGuillaume/MergedData/BURSTINESS.h5", 'w')
hd_index = np.array(hd_index)

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']

firing_rate = pd.HDFStore("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")['firing_rate']
fr_index = firing_rate.index.values[((firing_rate > 1.0).sum(1) == 3).values]


###############################################################################################################
# autocorr loading
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")

autocorr_wak = store_autocorr['wake']
autocorr_rem = store_autocorr['rem']
autocorr_sws = store_autocorr['sws']

# sws_index = autocorr_sws.columns.values[((autocorr_sws.loc[0.5:1.0]<0.4).all()).values]
# wak_index = autocorr_wak.columns.values[((autocorr_wak.loc[0.5:1.0]<0.4).all()).values]
# rem_index = autocorr_rem.columns.values[((autocorr_rem.loc[0.5:1.0]<0.4).all()).values]

# neurons = np.intersect1d(np.intersect1d(wak_index, rem_index), sws_index)
# neurons = np.intersect1d(np.intersect1d(rem_index, sws_index), fr_index)

# 1. starting at 2
autocorr_wak = store_autocorr['wake'].loc[0.5:]
autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
autocorr_sws = 	store_autocorr['sws'].loc[0.5:]
# autocorr_wak.loc[0.0] = 0.0
# autocorr_rem.loc[0.0] = 0.0
# autocorr_sws.loc[0.0] = 0.0

# sws_index = autocorr_sws.columns.values[((autocorr_sws.loc[0.5:1.0]<0.4).all()).values]

# 2. ISI 2 to 50
# store_isi = pd.HDFStore("/mnt/DataGuillaume/MergedData/ISI_ALL.h5")
# isi_wak = store_isi['wake'][2:30]
# isi_rem = store_isi['rem'][2:30]
# isi_sws = store_isi['sws'][2:30]

 # # # 2. greater than 0
# autocorr_wak = autocorr_wak.drop(autocorr_wak.columns[autocorr_wak.apply(lambda col: (col == 0).sum() >= 4)], axis = 1)
# autocorr_rem = autocorr_rem.drop(autocorr_rem.columns[autocorr_rem.apply(lambda col: (col == 0).sum() >= 4)], axis = 1)
# autocorr_sws = autocorr_sws.drop(autocorr_sws.columns[autocorr_sws.apply(lambda col: (col == 0).sum() >= 4)], axis = 1)
# # # # # 3. lower than 200 
autocorr_wak = autocorr_wak.drop(autocorr_wak.columns[autocorr_wak.apply(lambda col: col.max() > 100.0)], axis = 1)
autocorr_rem = autocorr_rem.drop(autocorr_rem.columns[autocorr_rem.apply(lambda col: col.max() > 100.0)], axis = 1)
autocorr_sws = autocorr_sws.drop(autocorr_sws.columns[autocorr_sws.apply(lambda col: col.max() > 100.0)], axis = 1)
# # 4. gauss filt
autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)


autocorr_wak = autocorr_wak[2:100]
autocorr_rem = autocorr_rem[2:100]
autocorr_sws = autocorr_sws[2:100]

# isi_wak = isi_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
# isi_rem = isi_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)
# isi_sws = isi_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.0)

# 6 combining all 
neurons = np.intersect1d(np.intersect1d(autocorr_wak.columns, autocorr_rem.columns), autocorr_sws.columns)
neurons = np.intersect1d(neurons, fr_index)
# neurons = fr_index
# neurons = autocorr_sws.columns

# autocorr = pd.concat([autocorr_wak[neurons],autocorr_rem[neurons],autocorr_sws[neurons],isi_wak[neurons],isi_rem[neurons],isi_sws[neurons]], ignore_index = False)
# autocorr = pd.concat([autocorr_sws[neurons]], ignore_index = False)
autocorr = pd.concat([autocorr_sws[neurons],autocorr_rem[neurons],autocorr_wak[neurons]], ignore_index = False)
# autocorr = autocorr_sws[neurons]
# autocorr = pd.concat([autocorr_rem.loc[2.5:20,neurons],autocorr_sws.loc[2.5:20,neurons]], ignore_index = False)
# autocorr = autocorr.apply(zscore)

if autocorr.isnull().any().any(): autocorr = autocorr.dropna(axis = 1, how = 'any')


# data = np.hstack((pca_wak, pca_rem, pca_sws))

data = autocorr.values.T

# data = autocorr_sws[neurons].values.T

neurons = autocorr.columns

####################################################################################
# TSNE
####################################################################################
n = 10
TSNE, divergence = makeAllTSNE(data, n)

for i in range(n):
	tmp = pd.DataFrame(index = neurons, data = TSNE[i].T)
	figure()	
	scatter(tmp[0], tmp[1], s = 10)
	scatter(tmp.loc[hd_index,0], tmp.loc[hd_index,1], s = 3)
	show()


tsne = pd.DataFrame(index = neurons, data = TSNE[0].T)




####################################################################################
# K MEANS
####################################################################################
from sklearn.cluster import KMeans
n_clusters = 2
km = KMeans(n_clusters = n_clusters).fit(data)

tsne['cluster'] = km.labels_
tsne['theta'] = theta.loc[tsne.index.values]['pvalue'] < 0.05
tsne['hd'] = 0
tsne.loc[np.intersect1d(hd_index, tsne.index.values), 'hd'] = 1


scatter(tsne[0], tsne[1], c = tsne['cluster']);show()




tsne.to_hdf("../figures/figures_articles/figure1/tsne.hdf5", key = 'space', mode = 'w')


######################################################################################
# LOADING THE MAP
######################################################################################
# carte = imread('../figures/mapping_to_align/paxino/carte_thalamus_paxino_page66.png')
# carte_61 = imread('../figures/mapping_to_align/allen/allen_61.png')
# carte_62 = imread('../figures/mapping_to_align/allen/GUillaumeThalamus2.png')
# carte_63 = imread('../figures/mapping_to_align/allen/allen_63.png')
# carte_64 = imread('../figures/mapping_to_align/allen/allen_64.png')
carte38_mouse17 = imread('../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
# carte = imread('../figures/mapping_to_align/paxino/paxino_64.png')
# bound_map = (-2768/1235,2853/1235,0,4010/1235) # map paxino
# bound_map_61 = (-3135/1764, 3106/1764, 0, 3867/1764)
# bound_map_62 = (-2847/1401, 2846/1401, 0, 3382/1401) # map 62
# bound_map_63 = (-4892/2199, 4913/2199, 0, 5655/2199) # map 63
# bound_map_64 = (-4555/2046, 4526/2046, 0, 5232/2046)
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
# bound_map = (-2072/1237, 2086/1237, 0, 2555/1237)


####################################################################################
# SHANK MAPPING from main_make_nucleus_set.py
####################################################################################
mapping_nucleus = cPickle.load(open("../data/maps/mapping_nucleus_allen.pickle", 'rb'))
neuron_to_shank = mapping_nucleus['neuron_to_shank']
neuron_to_channel = mapping_nucleus['neuron_to_channel']
session_shank = np.array([list(neuron_to_shank[n]) for n in neurons])
session_channel = np.array([neuron_to_channel[n] for n in neurons])
tsne['session'] = session_shank[:,0]
tsne['shank'] = 7 - session_shank[:,1]
tsne['nucleus'] = [mapping_nucleus['neuron_to_nucleus'][n] for n in neurons]

####################################################################################
# PLOT
####################################################################################
import matplotlib.gridspec as gridspec
from scipy.misc import imread
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

colors = ['red', 'green', 'blue', 'purple', 'orange', 'grey']
cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys']
markers = ['o', '^', '*', 's']

# Which embedding?
space = tsne


##################################################################################
# DOING IT BY HAND
##################################################################################
from skimage.filters import gaussian

space['x'] = np.nan
space['y'] = np.nan
ymax = {}
shanks_pos = {}
headdirs = {}
for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):	
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	x 			= data['x']
	y 			= data['y']*-1.0+np.max(data['y'])
	headdirs[m] = data['headdir']
	ymax[m] 	= y.max()
	shanks_pos[m] = (data['x'], data['y'])
	# neurons	
	for n in space.index[space.index.str.contains(m)]:
		space.loc[n, ['x', 'y']] = [x[space.loc[n, 'shank']],y[space.loc[n,'session']]]		
		# space.loc[n, 'y'] -= (4 - space.loc[n, 'channel'])*0.02 


shifts = np.array([	[-0.34, 0.56],
					[0.12, 0.6],
					[-0.35, 0.75],
					[-0.3, 0.5]
				])
angles = np.array([15.0, 10.0, 15.0, 20.0])

cartes = {	'Mouse12':carte38_mouse17,
			'Mouse17':carte38_mouse17,
			'Mouse20':carte38_mouse17,
			'Mouse32':carte38_mouse17 }
bounds = { 	'Mouse12':bound_map_38,
			'Mouse17':bound_map_38,
			'Mouse20':bound_map_38,
			'Mouse32':bound_map_38}

cPickle.dump({'shifts':shifts, 'angles':angles}, open("../figures/figures_articles/figure1/shifts.pickle", 'wb'))

fig_clusters = figure(figsize = (40,10))

for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
# for m in ['Mouse32']:
	# i = 3
	"""
	shank and hd position
	"""
	idx = space.index[space.index.str.contains(m)]
	subspace = space.loc[idx].copy()
	M = getRotationMatrix((ymax[m], 1.4), angles[i])
	xy1 = np.vstack((subspace['x'].values, subspace['y'].values, np.ones(len(subspace))))
	new_xy1 = M.dot(xy1).T
	xx, yy = np.meshgrid(shanks_pos[m][0], shanks_pos[m][1]*-1.0+ymax[m])
	xy_shank = np.vstack((xx.flatten(), yy.flatten(), np.ones(xx.size)))
	new_xy_shank = M.dot(xy_shank).T
	new_xy1 -= new_xy_shank.min(0)
	new_xy_shank -= new_xy_shank.min(0)
	subspace['x'] = new_xy1[:,0]
	subspace['y'] = new_xy1[:,1]

	"""
	cluster density
	"""
	cluster_density = np.zeros((n_clusters, len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	total_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	theta_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	neg_swr_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	pos_swr_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	frate_density = np.zeros((3, len(shanks_pos[m][1]), len(shanks_pos[m][0]))) # REM/WAKE | SWS/WAKE
	burst_density = np.zeros((3, len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	for n in subspace.index.values:
		cluster_density[subspace.loc[n,'cluster'],subspace.loc[n,'session'],subspace.loc[n,'shank']] += 1.0
		total_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += 1.0
		theta_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += float(theta.loc[n,'pvalue']<0.05)
		neg_swr_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += float(swr.loc[0,n]<np.percentile(swr.loc[0.0].values, 40))
		pos_swr_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += float(swr.loc[0,n]>np.percentile(swr.loc[0.0].values, 60))
	
	for ses in range(len(shanks_pos[m][1])):
		for sh in range(len(shanks_pos[m][0])):
			idx = subspace[np.logical_and(subspace['session'] == ses, subspace['shank'] == sh).values].index
			frate_density[0,ses,sh] = (firing_rate.loc[idx,'wake']/firing_rate.loc[idx,'sws']).mean(0)
			frate_density[1,ses,sh] = (firing_rate.loc[idx,'rem']/firing_rate.loc[idx,'sws']).mean(0)
			frate_density[2,ses,sh] = firing_rate.loc[idx,'sws'].mean(0)
			clu0_idx = np.intersect1d(idx, subspace.index[subspace['cluster'] == 0])
			burst_density[:,ses,sh] = burst.loc[clu0_idx].mean(0).values

	frate_density[np.isnan(frate_density)] = 0.0
	burst_density[np.isnan(burst_density)] = 0.0

	# clusters | theta | pos swr | neg swr | frate wake | frate rem | frate sws | burst_wake | burst_rem | burst_sws | count
	images = np.vstack((cluster_density,theta_density[np.newaxis,:,:],pos_swr_density[np.newaxis,:,:],neg_swr_density[np.newaxis,:,:]))
	images = images / total_density
	images[np.isnan(images)] = 0.0

	images = np.vstack((images, frate_density, burst_density))	
	new_images = []
	for k in range(len(images)):
		xnew, ynew, tmp = interpolate(images[k].copy(), shanks_pos[m][0], shanks_pos[m][1], 0.010)		
		tmp2 = gaussian(tmp, sigma = 15.0, mode = 'reflect')
		new_images.append(tmp2)
	new_images = np.array(new_images)

	_, h, w = new_images.shape
	rotated_images = np.zeros((len(new_images), h*3, w*3))*np.nan
	rotated_images[:,h:h*2,w:w*2] = new_images.copy() + 1.0
	for k in range(len(new_images)):
		rotated_images[k] = rotateImage(rotated_images[k], -angles[i])
		rotated_images[k][rotated_images[k] == 0.0] = np.nan
		rotated_images[k] -= 1.0

	tocrop = np.where(~np.isnan(rotated_images[-1]))
	rotated_images = rotated_images[:,tocrop[0].min()-1:tocrop[0].max()+1,tocrop[1].min()-1:tocrop[1].max()+1]
	xlength, ylength = getXYshapeofRotatedMatrix(shanks_pos[m][0].max(), shanks_pos[m][1].max(), angles[i])

	bound = (shifts[i][0],xlength+shifts[i][0],shifts[i][1],ylength+shifts[i][1])
	subspace['x'] += shifts[i][0]
	subspace['y'] += shifts[i][1]
	new_xy_shank += shifts[i]
	
	space.loc[subspace.index] = subspace

	################################################################################
	# SAVING DATA FOR FIGURE ARTICLE
	################################################################################
	subspace.to_hdf("../figures/figures_articles/figure1/subspace_"+m+".hdf5", key = 'subspace', mode = 'w')
	
	data_to_save = {'rotated_images':rotated_images,
					'new_xy_shank':new_xy_shank,
					'bound':bound
					}

	cPickle.dump(data_to_save, open("../figures/figures_articles/figure1/rotated_images_"+m+".pickle", 'wb'))
	

	######################################################################################	
	for k in range(n_clusters):
		if m == 'Mouse12': 
			ax = fig_clusters.add_subplot(4,n_clusters,k+1)	
		elif m == 'Mouse17': 
			ax = fig_clusters.add_subplot(4,n_clusters,n_clusters+k+1)
		elif m == 'Mouse20':
			ax = fig_clusters.add_subplot(4,n_clusters,2*n_clusters+k+1)
		elif m == 'Mouse32':
			ax = fig_clusters.add_subplot(4,n_clusters,3*n_clusters+k+1)
		ax.imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
		ax.imshow(rotated_images[k], extent = bound, alpha = 0.8, aspect = 'equal', cmap = cmaps[k], vmin = 0.0, vmax = 0.6)
		ax.scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 0.5, c = 'black')				
		ax.scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = headdirs[m].flatten()*8, c = 'red')
		# idx = np.intersect1d(hd_index, space.index[space.index.str.contains(m)])
		# scatter(subspace.loc[idx, 'x'], subspace.loc[idx, 'y'], s = 4)		
		ax.set_xlim(-2.5, 2.5)
		ax.set_ylim(0, 3.0)
	######################################################################################
	figure(figsize = (40,10))
	'''
	theta
	'''
	subplot(3,3,1)
	imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
	imshow(rotated_images[n_clusters], extent = bound, alpha = 0.8, aspect = 'equal')
	scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 0.5, c = 'black')				
	scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = headdirs[m].flatten()*8, c = 'red')
	xlim(-2.5, 2.5)
	ylim(0, 3.0)
	title("theta")
	'''
	pos swr
	'''
	subplot(3,3,2)
	imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
	imshow(rotated_images[n_clusters+1], extent = bound, alpha = 0.8, aspect = 'equal')
	scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 0.5, c = 'black')				
	scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = headdirs[m].flatten()*8, c = 'red')
	xlim(-2.5, 2.5)
	ylim(0, 3.0)
	title("positive swr")
	'''
	neg swr
	'''
	subplot(3,3,3)
	imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
	imshow(rotated_images[n_clusters+2], extent = bound, alpha = 0.8, aspect = 'equal')
	scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 0.5, c = 'black')				
	scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = headdirs[m].flatten()*8, c = 'red')
	xlim(-2.5, 2.5)
	ylim(0, 3.0)
	title("negative swr")
	'''
	firing rate
	'''
	for j, ep, lab in zip(range(3),['wake', 'rem', 'sws'], ['wake/sws', 'rem/sws', 'sws']):
		subplot(3,3,j+4)
		imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
		imshow(rotated_images[n_clusters+3+j], extent = bound, alpha = 0.8, aspect = 'equal')
		scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 0.5, c = 'black')				
		scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = headdirs[m].flatten()*8, c = 'red')
		xlim(-2.5, 2.5)
		ylim(0, 3.0)
		title(ep)
		ylabel(lab)
	'''
	burstiness distribution
	'''
	for j, ep in zip(range(3),['wake', 'rem', 'sws']):
		subplot(3,3,j+7)
		imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
		imshow(rotated_images[n_clusters+6+j], extent = bound, alpha = 0.8, aspect = 'equal')
		scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 0.5, c = 'black')				
		scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = headdirs[m].flatten()*8, c = 'red')
		xlim(-2.5, 2.5)
		ylim(0, 3.0)
		title(ep)
		ylabel("mean burstiness")

	##########################################################################################
	# SHANKS POSITION
	##########################################################################################
	figure()
	imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
	scatter(new_xy_shank[:,0], new_xy_shank[:,1])
	scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = headdirs[m].flatten()*8, c = 'red')


show()



space.to_hdf("../figures/figures_articles/figure1/space.hdf5", key = 'space', mode = 'w')



#########################################################################################
# COMPARING LENGTH OF AUTOCORR FOR TSNE
#########################################################################################
count = pd.DataFrame(columns = [0,1])
for c in np.arange(10,250, 10):
	autocorr_wak = store_autocorr['wake']
	autocorr_rem = store_autocorr['rem']
	autocorr_sws = store_autocorr['sws']
	autocorr_wak = store_autocorr['wake'].loc[0.5:]
	autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
	autocorr_sws = 	store_autocorr['sws'].loc[0.5:]
	autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
	autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
	autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
	autocorr = pd.concat([autocorr_sws[2:c][neurons],autocorr_rem[2:c][neurons],autocorr_wak[2:c][neurons]], ignore_index = False)
	data = autocorr.values.T
	n_clusters = 2
	tmp = np.zeros((100,2))
	for i in range(100):		
		km = KMeans(n_clusters = n_clusters).fit(data)
		for k in range(2):
			n_in_cluster = neurons[km.labels_ == k]
			tmp[i,k] = len(np.intersect1d(n_in_cluster, hd_index))	
	count.loc[c] = np.mean(tmp, 0)



























sys.exit()






##############################################################################
# FINDING AXE OF VARIANCE
##############################################################################
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from evolutionary_search import maximize, optimize

center = space[['x', 'y']].mean(0).values
xy = space.loc[neurons, ['x', 'y']].values.copy()
# centering neurons position
xypos = xy - center

data = autocorr[neurons].values.T
pca = PCA(n_components = 2)
compo = pca.fit_transform(data)


gradient = []
def findaxis(rho):
	a = np.tan(rho)
	b = 0.0
	newxline = a * (xypos[:,1] + (xypos[:,0] / a) - b) / (a**2. + 1)
	newyline = a * newxline + b
	xyline = np.vstack((newxline, newyline)).T
	distto0 = np.sqrt(np.power(newxline, 2.0) + np.power(newyline - b, 2.0))
	score = pearsonr(compo[:,0], distto0)[0]
	gradient.append(score)
	return score


axismap0 = dict({'rho':np.arange(0, 2*np.pi, 0.001)})				

best_params, best_score, score_results, history, logbook = maximize(findaxis, axismap0, generations_number = 2000, verbose = True)
bestrho = best_params['rho']


###############################################################################
# THETA AND BURSTINESS
###############################################################################
figure()
subplot(131)
scatter(space[~space['theta']][0], space[~space['theta']][1], c = 'grey')
scatter(space[space['theta']][0], space[space['theta']][1], c = 'red')
scatter(space.loc[hd_index][0], space.loc[hd_index][1], s = 5, c= 'white')
title('theta')
subplot(132)
thr = 12
bu = burst.loc[space.index.values]
scatter(space.loc[bu[bu['sws']<thr].index.values, 0], space.loc[bu[bu['sws']<thr].index.values, 1], c = 'grey')
scatter(space.loc[bu[bu['sws']>thr].index.values, 0], space.loc[bu[bu['sws']>thr].index.values, 1], c = 'green')
scatter(space.loc[hd_index][0], space.loc[hd_index][1], s = 5, c= 'white')
title('burstiness')
subplot(133)
scatter(space.loc[bu.index.values,0], space.loc[bu.index.values,1], c=bu['sws'])
scatter(space.loc[hd_index][0], space.loc[hd_index][1], s = 5, c= 'white')
title('burstiness')

show()

sys.exit()


###################################################################################
# CLuster + Mapping per mouse + count of nucleus
###################################################################################
figure()

allheaddir = []

gs = gridspec.GridSpec(n_clusters,7)
for i in np.unique(space['cluster']):	
	ax = subplot(gs[i,0])	
	scatter(space[space['cluster'] == i][0], space[space['cluster'] == i][1], c = colors[i], s = 5)
	scatter(space[space['cluster'] != i][0], space[space['cluster'] != i][1], c = 'grey', s = 5)	
	scatter(space.loc[hd_index][0], space.loc[hd_index][1], s = 1, c= 'black')
	for j, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
		subspace = space[space.index.str.contains(m)]
		group = subspace[subspace['cluster'] == i]
		data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
		headdir 	= data['headdir']
		x 			= data['x']
		y 			= data['y']
		
		subspace['y'] = y[subspace['session']]
		subspace['x'] = x[subspace['shank']]
		xx, yy = np.meshgrid(x, y)
		# head direction
		xnew, ynew, newheaddir = interpolate(headdir.copy(), x, y, 0.01)
		allheaddir.append(pd.DataFrame(data = newheaddir.copy(), index = ynew, columns = xnew))

		newheaddir[newheaddir < np.percentile(newheaddir, 90)] = np.nan

		# density
		groupmap = np.zeros((len(y), len(x)))
		for k in range(len(group)):
			groupmap[int(group.iloc[k]['session']), int(group.iloc[k]['shank'])] += 1
		groupmap /= np.max(groupmap)		
		newx, newy, newgroupmap = interpolate(groupmap.copy(), x, y, 0.01)
		newgroupmap[newgroupmap<np.percentile(newgroupmap, 50)] = np.nan

		ax = subplot(gs[i,j+1])		
		contour(newheaddir, extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'Greys', origin = 'upper', alpha = 0.5)
		imshow(newgroupmap, extent = (newx[0], newx[-1], newy[-1], newy[0]), cmap = cmaps[i])
		gca().set_aspect('equal')

	# counting nucleus	
	nuc = np.array(space[space['cluster'] == i]['nucleus'])
	allnuc, count = np.unique(nuc, return_counts = True)	
	ax = subplot(gs[i,-2:])
	bar(np.arange(len(allnuc)), count)
	xticks(np.arange(len(allnuc)), allnuc, fontsize = 6)

#################################################################################
# ISOLINES OF ALL CLUSTER TOGETHER PER MOUSE
#################################################################################
groupmaps = {i:[] for i in range(n_clusters)}
hdmaps = []
figure(figsize = (15,20))
for j, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
	subspace = space[space.index.str.contains(m)]
	# ax = subplot(gs[j+1,0])
	ax = subplot(2,2,j+1)
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	headdir 	= data['headdir']
	x 			= data['x']
	y 			= data['y']
	subspace['y'] = y[subspace['session']]
	subspace['x'] = x[subspace['shank']]
	xx, yy = np.meshgrid(x, y)
	# head direction
	# for optimization of shifts, points with < 0.5 are set to 0
	headdir2 = headdir.copy()
	headdir2[headdir2<0.5] = 0.0
	xnew, ynew, newheaddir2 = interpolate(headdir2.copy(), x, y, 0.01)
	hdmaps.append(np.copy(newheaddir2))
	xnew, ynew, newheaddir = interpolate(headdir.copy(), x, y, 0.01)
	newheaddir[newheaddir < np.percentile(newheaddir, 90)] = np.nan
	imshow(newheaddir, extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'Greys', origin = 'upper', alpha = 0.5)
	# contour(newheaddir, extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'Greys', origin = 'upper', alpha = 0.5)

	for i in np.unique(subspace['cluster']):
		group = subspace[subspace['cluster'] == i]
		# density
		groupmap = np.zeros((len(y), len(x)))
		for k in range(len(group)):
			groupmap[int(group.iloc[k]['session']), int(group.iloc[k]['shank'])] += 1
		groupmap /= np.max(groupmap)		
		newx, newy, newgroupmap = interpolate(groupmap.copy(), x, y, 0.01)
		groupmaps[i].append(pd.DataFrame(data = newgroupmap.copy(), index = newy, columns = newx))
		# newgroupmap[newgroupmap<np.percentile(newgroupmap, 50)] = np.nan			
		contour(newgroupmap, extent = (newx[0], newx[-1], newy[-1], newy[0]), cmap = cmaps[i], origin = 'upper')		
	# gca().set_aspect('equal')
	title(m)




sys.exit()

###############################################################################
# PLOT OF AUTOCORRELOGRAM PER CLUSTER
###############################################################################
figure()
for i in range(n_clusters):	
	idx = space[space['cluster'] == i].index.values
	subplot(3,n_clusters,i+1)	
	plot(autocorr_wak[idx], color = 'grey', alpha = 0.5)
	plot(autocorr_wak[idx].mean(1), color = 'black')
	ylabel('wake')
	subplot(3,n_clusters,i+1+n_clusters)
	plot(autocorr_rem[idx], color = 'grey', alpha = 0.5)
	plot(autocorr_rem[idx].mean(1), color = 'black')
	ylabel('rem')
	subplot(3,n_clusters,i+1+2*n_clusters)	
	plot(autocorr_sws[idx], color = 'grey', alpha = 0.5)
	plot(autocorr_sws[idx].mean(1), color = 'black')
	ylabel('sws')















sys.exit()









# 	# shanks
# 	xx, yy = np.meshgrid(shanks_pos[m][0], shanks_pos[m][1])
# 	shank_pos = np.vstack((xx.flatten(),yy.flatten(),np.ones(len(xx.flatten()))))
# 	new_shank_pos = M.dot(shank_pos)[0:2]
# 	new_shank_pos = new_shank_pos.T
# 	new_shank_pos += shifts[i]
	
# 	for n, i, j in zip(subspace.index, yindex, xindex):
# 		cluster_density[subspace.loc[n,'cluster'], i-1, j-1] += 1
# 		total_density[i-1,j-1] += 1
# 		theta_density[i-1,j-1] += theta.loc[n, 'kappa']

# 	theta_density = theta_density/total_density
# 	theta_density[np.isnan(theta_density)] = 0.0

# 	new_cluster_density = cluster_density / (total_density + 1.0)	

# 	figure(figsize = (40,10))
# 	for j in range(len(cluster_density)):
# 		tmp = new_cluster_density[j].copy()
# 		tmp = filters.gaussian(tmp, 1.0)		
# 		# tmp[tmp < 0.05] = np.nan		
# 		subplot(2,3,j+1)	
# 		imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
# 		imshow(tmp, extent = (xbins[0], xbins[-1], ybins[-1], ybins[0]), alpha = 0.8, aspect = 'equal', cmap = cmaps[j], vmin = 0, vmax = 1.0)
# 		scatter(new_shank_pos[:,0], new_shank_pos[:,1], s = 3, c = 'black')
# 		idx = np.intersect1d(hd_index, space.index[space.index.str.contains(m)])
# 		scatter(subspace.loc[idx, 'x'], subspace.loc[idx, 'y'], s = 5, c = 'red')
# 		xlim(-2.5, 2.5)
# 		ylim(0, 3.0)
# 	subplot(2,3,6)
# 	imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
# 	imshow(filters.gaussian(theta_density, 1.0), extent = (xbins[0], xbins[-1], ybins[-1], ybins[0]), alpha = 0.8, aspect = 'equal', cmap = 'Greys')
# 	idx = np.intersect1d(hd_index, space.index[space.index.str.contains(m)])
# 	scatter(subspace.loc[idx, 'x'], subspace.loc[idx, 'y'], s = 5)
# 	xlim(-2.5, 2.5)
# 	ylim(0, 3.0)

# 	sys.exit()

# show()

# sys.exit()




figure()
for k, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
	idx = space.index[space.index.str.contains(m)]
	subspace = space.loc[idx].copy()
	M = getRotationMatrix((ymax[m], 1.4), angles[k])
	xy1 = np.vstack((subspace['x'].values, subspace['y'].values, np.ones(len(subspace))))
	new_xy1 = M.dot(xy1)
	subspace['x'] = new_xy1[0] + shifts[k,0]
	subspace['y'] = new_xy1[1] + shifts[k,1]
	xbins = np.arange(-1.0, 2.0, 0.1)
	ybins = np.arange(-0.2, 3.0, 0.1)[::-1]
	cluster_density = np.zeros((n_clusters, len(ybins)-1, len(xbins)-1))
	total_density = np.zeros((len(ybins)-1, len(xbins)-1))
	theta_density = np.zeros((len(ybins)-1, len(xbins)-1))
	xindex = np.digitize(subspace['x'].values.astype('float64'), xbins)
	yindex = np.digitize(subspace['y'].values.astype('float64'), ybins)	
	for n, i, j in zip(subspace.index, yindex, xindex):
		cluster_density[subspace.loc[n,'cluster'], i-1, j-1] += 1
		total_density[i-1,j-1] += 1

	new_cluster_density = cluster_density / (total_density + 1.0)	

	subplot(2,2,k+1)
	imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
	for j in range(len(cluster_density)):		
		tmp = new_cluster_density[j].copy()
		tmp = filters.gaussian(tmp, 1.0)		
		tmp[tmp<0.1] = np.nan
		imshow(tmp, extent = (xbins[0], xbins[-1], ybins[-1], ybins[0]), alpha = 0.8, aspect = 'equal', cmap = cmaps[j], interpolation = 'bilinear')
		idx = np.intersect1d(hd_index, space.index[space.index.str.contains(m)])
		scatter(subspace.loc[idx, 'x'], subspace.loc[idx, 'y'], s = 5)
	xlim(-2.5, 2.5)
	ylim(0, 3.0)

show()








#################################################################################
# POSITIONS OF SHANK WITH HD ON THE MAP
#################################################################################
# figure()
# # subplot(121)
# # imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
# # for m in ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']:
# # 	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
# # 	headdir 	= data['headdir']
# # 	tmp = positions_shanks[m]
# # 	scatter(positions_shanks[m][:,:,0], positions_shanks[m][:,:,1], s = 2)
# # 	scatter(tmp[:,:,0], tmp[:,:,1], s = headdir*50., label = m)
# # legend()
# # subplot(122)
# imshow(carte_63, extent = bound_map_63, aspect = 'equal', interpolation = 'bilinear')
# for m in ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']:

# 	idx = np.intersect1d(hd_index, space.index[space.index.str.contains(m)])
# 	scatter(space.loc[idx,'x'], space.loc[idx, 'y'], s = 10)
# legend()

# show()






# #################################################################################
# # MAXIMIZE INDEX OF SIMILARITY 
# #################################################################################
from scipy.optimize import minimize
from skimage.measure import compare_ssim
import skimage.transform as tf
from skimage.filters import gaussian
import itertools
from evolutionary_search import maximize, optimize

gradient = []

hdlmaps = np.zeros((4,183*3,141*3))
for n in range(4): # mouse		
	hdlmaps[n,183:183+hdmaps[n].shape[0],141:141*2] = np.copy(hdmaps[n])
	# hdlmaps[n] = gaussian(hdlmaps[n], sigma = 15)
grouplmaps = np.zeros((n_clusters,4,183*3,141*3))
for c in groupmaps.keys(): # cluster
	for n in range(4): # mouse
		grouplmaps[c,n,183:183+groupmaps[c][n].shape[0],141:141*2] = np.copy(groupmaps[c][n].values)
		grouplmaps[c,n] = gaussian(grouplmaps[c,n], sigma = 15)


def func(x12, y12, t12, x17, y17, t17, x20, y20, t20, x32, y32, t32):
	xyshifts = np.array([ 	[x12, y12, t12],
							[x17, y17, t17],
							[x20, y20, t20],
							[x32, y32, t32]
						])	
	score_sim = np.zeros(len(groupmaps.keys())+1)
	# groupmaps are normalized 
	tmp = grouplmaps.copy()
	for c in range(n_clusters): # cluster
		shifted = []
		for n in range(4):
			tmp2 = tf.rotate(tmp[c,n], -1.0*xyshifts[n,2])
			tform = tf.SimilarityTransform(translation = (xyshifts[n,0], xyshifts[n,1]))
			tmp3 = tf.warp(tmp2, tform.inverse)			
			shifted.append(tmp3)
		sim = []
		for a, b in itertools.product(shifted, shifted):
			sim.append(compare_ssim(a, b))
		score_sim[c] = np.sum(sim)

	# + head direction map
	tmp = hdlmaps.copy()
	shifted = []
	for n in range(4):		
		tmp2 = tf.rotate(tmp[n], -1.0*xyshifts[n,2])
		tform = tf.SimilarityTransform(translation = (xyshifts[n,0], xyshifts[n,1]))
		tmp3 = tf.warp(tmp2, tform.inverse)		
		shifted.append(tmp3)
	sim = []
	for a, b in itertools.product(shifted, shifted):
		sim.append(compare_ssim(a, b))
	
	score_sim[-1] = np.sum(sim)
	weights = np.array(list(np.ones(n_clusters)*0.001)+[0.995])
	score = np.sum(score_sim*weights)
	gradient.append(score)
	return score

	



x0 = {	'x12': np.arange(-50, 50, 1.0),
		'y12': np.arange(-50, 50, 1.0),
		't12': np.arange(5, 25, 1.0),
		# 's12': [1.0],
		'x17': np.arange(-50, 50, 1.0),
		'y17': np.arange(-50, 50, 1.0),
		't17': np.arange(5, 25, 1.0),
		# 's17': [1.0],
		'x20': np.arange(-50, 50, 1.0),
		'y20': np.arange(-50, 50, 1.0),
		't20': np.arange(5, 25, 1.0),
		# 's20': [1.0],
		'x32': np.arange(-50, 50, 1.0),
		'y32': np.arange(-50, 50, 1.0),
		't32': np.arange(5, 25, 1.0)
		# 's32': [1.0]
	}


optimize.compile()
best_params, best_score, score_results, history, logbook = maximize(func, x0, population_size = 20, generations_number = 200, verbose = True, n_jobs = 8)

angles = np.array([best_params['t12'],best_params['t17'],best_params['t20'],best_params['t32']])
shifts = np.array([	[best_params['x12'],best_params['y12']],
					[best_params['x17'],best_params['y17']],
					[best_params['x20'],best_params['y20']],
					[best_params['x32'],best_params['y32']]])
# scales = np.array([best_params['s12'],best_params['s17'],best_params['s20'],best_params['s32']])

shifts *= (1.4/141)

sys.exit()

#################################################################################
# MINIMIZE DISTANCE BETWEEN POINTS
#################################################################################
from scipy.optimize import minimize
from skimage.measure import compare_ssim
import skimage.transform as tf
from skimage.filters import gaussian
import itertools
from evolutionary_search import maximize, optimize


space['x'] = np.nan
space['y'] = np.nan
allheaddirs = []
for j, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):	
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	x 			= data['x']
	y 			= data['y']*-1.0
	headdir 	= data['headdir']
	allheaddirs.append(pd.DataFrame(index = y, columns = x, data = headdir))
	for n in space.index[space.index.str.contains(m)]:
		space.loc[n, ['x', 'y']] = [x[space.loc[n, 'shank']],y[space.loc[n,'session']]]
		# channel 4 is zero
		# space.loc[n, 'y'] -= -1.0*(4 - space.loc[n, 'channel'])*0.02 
	

gradient = []

# def func(x12, y12, t12, x17, y17, t17, x20, y20, t20, x32, y32, t32):
# 	xyshifts = np.array([ 	[x12, y12, t12],
# 							[x17, y17, t17],
# 							[x20, y20, t20],
# 							[x32, y32, t32]
# 						])
# 	new_positions = space[['x', 'y']].copy()
	
# 	scores = np.zeros(1+n_clusters)

# 	for j, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
# 		idx = space.index[space.index.str.contains(m)]
# 		xy1 = space.loc[idx, ['x', 'y']].values
# 		xy1 += xyshifts[j,0:2]
# 		xy1 = np.vstack((xy1.T, np.ones(len(xy1))))
# 		M = getRotationMatrix((space['y'][space.index.str.contains(m)].min(), 1.4), xyshifts[j,2])
# 		xy1 = M.dot(xy1)
# 		new_positions.loc[idx, 'x'] = xy1[0]
# 		new_positions.loc[idx, 'y'] = xy1[1]
	
# 	min_dist = 0.5
# 	# headdirections | removing neurons too far from main cluster of the same mouse
# 	tokeep = []
# 	for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
# 		idx = np.intersect1d(new_positions.index[new_positions.index.str.contains(m)], hd_index)
# 		center = new_positions.loc[idx].mean(0).values
# 		dist = np.sqrt(np.sum(np.power(new_positions.loc[idx].values - center, 2.0), 1))
# 		tokeep.append(idx[dist < min_dist])
# 	tokeep = np.hstack(tokeep)
# 	tmp = new_positions.loc[tokeep].values
# 	dist = np.sqrt(np.power(np.vstack(tmp[:,0]) - tmp[:,0], 2.0) + np.power(np.vstack(tmp[:,1]) - tmp[:,1], 2.0))
# 	scores[0] = dist.sum()

# 	# # clusters
# 	# for n in range(n_clusters):
# 	# 	idx = space.index[space['cluster'] == n]
# 	# 	tmp = new_positions.loc[idx].values
# 	# 	dist = np.sqrt(np.power(np.vstack(tmp[:,0]) - tmp[:,0], 2.0) + np.power(np.vstack(tmp[:,1]) - tmp[:,1], 2.0))		
# 	# 	scores[n+1] = np.sum(dist)

# 	gradient.append(np.sum(scores))
# 	return -np.sum(scores)

def func(x12, y12, t12, x17, y17, t17, x20, y20, t20, x32, y32, t32):
	xyshifts = np.array([ 	[x12, y12, t12],
							[x17, y17, t17],
							[x20, y20, t20],
							[x32, y32, t32]
						])

	xyall = []
	for i in range(4):
		xx, yy = np.meshgrid(allheaddirs[i].columns.values, allheaddirs[i].index.values)
		xy = np.vstack((xx[(allheaddirs[i]>0.34).values], yy[(allheaddirs[i]>0.34).values])).T
		xy += xyshifts[i,0:2]		
		xyall.append(xy)
	xyall = np.vstack(xyall)
	dist = np.sqrt(np.power(np.vstack(xyall[:,0]) - xyall[:,0], 2.0) + np.power(np.vstack(xyall[:,1]) - xyall[:,1], 2.0))
	gradient.append(dist.sum())
	return -dist.sum()


x0 = {	'x12': np.arange(-2, 2, 0.1),
		'y12': np.arange(-2, 2, 0.1),
		# 't12': [0],
		't12': np.arange(10, 20, 1.0),
		# 's12': [1.0],
		'x17': np.arange(-2, 2, 0.1),
		'y17': np.arange(-2, 2, 0.1),
		# 't17': [0],
		't17': np.arange(10, 20, 1.0),
		# 's17': [1.0],
		'x20': np.arange(-2, 2, 0.1),
		'y20': np.arange(-2, 2, 0.1),
		# 't20': [0],
		't20': np.arange(10, 20, 1.0),
		# 's20': [1.0],
		'x32': np.arange(-2, 2, 0.1),
		'y32': np.arange(-2, 2, 0.1),
		# 't32': [0]
		't32': np.arange(10, 20, 1.0)
		# 's32': [1.0]
	}



best_params, best_score, score_results, history, logbook = maximize(func, x0, generations_number = 1000, verbose = True, n_jobs = 1)

angles = np.array([best_params['t12'],best_params['t17'],best_params['t20'],best_params['t32']])
shifts = np.array([	[best_params['x12'],best_params['y12']],
					[best_params['x17'],best_params['y17']],
					[best_params['x20'],best_params['y20']],
					[best_params['x32'],best_params['y32']]])

# angles = np.ones(4)*15.0

sys.exit()

# for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
# 	idx = np.intersect1d(new_positions.index[new_positions.index.str.contains(m)], tokeep)
# 	scatter(new_positions.loc[idx, 'x'], new_positions.loc[idx, 'y'], label = m)
# legend()
# show()


# 	xy1 = new_positions.loc[idx, ['x', 'y']].values
# 	xy1 += shifts[i,0:2]
# 	xy1 = np.vstack((xy1.T, np.ones(len(xy1))))
# 	M = getRotationMatrix((space['y'][space.index.str.contains(m)].min(), 1.4), angles[i])
# 	xy1 = M.dot(xy1)	
# 	scatter(xy1[0], xy1[1], label = m)
# legend()
# show()

######################################################################################
# POSITION OF SHANKS AND NEURONS WITH THE COST MINIMIZATION
######################################################################################
newxcost = {}
newycost = {}
positions_shanks = {}
space['x'] = np.nan
space['y'] = np.nan
common_offset = (0.54, 2.6)

for j, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):	
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	x 			= data['x']
	y 			= data['y']*-1.0	
	newxcost[m] = x + shifts[j,0]
	newycost[m] = y + shifts[j,1]
	M = getRotationMatrix((data['y'].max(), data['x'].max()), angles[j])
	# neurons	
	for n in space.index[space.index.str.contains(m)]:
		space.loc[n, ['x', 'y']] = [x[space.loc[n, 'shank']],y[space.loc[n,'session']]]
		# channel 4 is zero
		# space.loc[n, 'y'] -= -1.0*(4 - space.loc[n, 'channel'])*0.02 
		space.loc[n, 'x'] += shifts[j,0]
		space.loc[n, 'y'] += shifts[j,1]
	xy1 = np.vstack((space[space.index.str.contains(m)]['x'].values, space[space.index.str.contains(m)]['y'].values, np.ones(space.index.str.contains(m).sum())))
	new_xy1 = M.dot(xy1)
	new_xy1[0] += common_offset[0]
	new_xy1[1] += common_offset[1]
	for i, n in enumerate(space[space.index.str.contains(m)].index.values):
		space.loc[n, ['x', 'y']] = new_xy1[:,i]
	# shanks
	xx, yy = np.meshgrid(newxcost[m], newycost[m])
	shank_pos = np.vstack((xx.flatten(),yy.flatten(),np.ones(len(xx.flatten()))))
	new_shank_pos = M.dot(shank_pos)[0:2]
	new_shank_pos = new_shank_pos.T
	new_shank_pos[:,0] += common_offset[0]
	new_shank_pos[:,1] += common_offset[1]
	new_shank_pos = new_shank_pos.reshape(len(y), len(x), 2)
	positions_shanks[m] = new_shank_pos	




# ######################################################################################
# # ROTATING THE HEAD DIRECTION DENSITIES
# #####################################################################################
# def getXYshapeofRotatedMatrix(x, y, n):
# 	angle = angles[n]
# 	return (x * np.cos(angle * np.pi /180.) + y * np.sin(angle * np.pi / 180.),
# 			y * np.cos(angle * np.pi /180.) + x * np.sin(angle * np.pi / 180.)			)


# for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
# 	# head direction
# 	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
# 	headdir 	= data['headdir']
# 	x 		 	= newxcost[m]
# 	y 			= newycost[m]
# 	xnew, ynew, newheaddir = interpolate(headdir.copy(), x, y, 0.01)
# 	# newheaddir[newheaddir < np.percentile(newheaddir, 90)] = np.nan
# 	rotateheaddir = np.zeros((newheaddir.shape[0]*2,newheaddir.shape[1]*2))*np.nan
# 	rotateheaddir[rotateheaddir.shape[0]//4:rotateheaddir.shape[0]//4+newheaddir.shape[0],rotateheaddir.shape[1]//4:rotateheaddir.shape[1]//4+newheaddir.shape[1]] = newheaddir.copy()+1.0
# 	rotateheaddir = rotateImage(rotateheaddir, -angles[i])
# 	rotateheaddir[rotateheaddir == 0.] = np.nan
# 	rotateheaddir -= 1.0
# 	tocrop = np.where(~np.isnan(rotateheaddir))
# 	rotateheaddir = rotateheaddir[tocrop[0].min()-1:tocrop[0].max()+1,tocrop[1].min()-1:tocrop[1].max()+1]
# 	# rotateheaddir[rotateheaddir < 0.5] = 0.0
# 	xlength, ylength = getXYshapeofRotatedMatrix(data['x'].max(), data['y'].max(), i)
# 	xylengths[m] = (xlength, ylength)
# 	hd_densities[m] = rotateheaddir

###################################################################################
# GROUPING CLUSTER FOR ALL MICE ON THE MAP
###################################################################################
xbins = np.arange(-0.6, 2.0, 0.1)
ybins = np.arange(-0.2, 3.0, 0.1)[::-1]
xpos = xbins[0:-1] + (xbins[1] - xbins[0])/2.
ypos = ybins[0:-1] + (ybins[1] - ybins[0])/2.
cluster_density = np.zeros((n_clusters, len(ybins)-1, len(xbins)-1))
total_density = np.zeros((len(ybins)-1, len(xbins)-1))
xindex = np.digitize(space['x'].values.astype('float64'), xbins)
yindex = np.digitize(space['y'].values.astype('float64'), ybins)
for n, j, i in zip(space.index, xindex, yindex):
	cluster_density[space.loc[n,'cluster'], i-1, j-1] += 1
	total_density[i-1,j-1] += 1

for i in range(len(cluster_density)):
	cluster_density[i] = cluster_density[i] / total_density
	cluster_density[i][np.isnan(cluster_density[i])] = 0.0

new_cluster_density = []
for n in range(len(cluster_density)):
	tmp2 = np.copy(cluster_density[n])
	# tmp2[tmp2 <= 1.0] = 0.0
	tmp = filters.gaussian(tmp2, 1.0)
	# tmp[tmp <= 4*np.median(tmp)] = 0.0 
	new_cluster_density.append(tmp)

new_cluster_density = np.array(new_cluster_density)


figure()
for n in range(n_clusters):
	subplot(2,3,n+1)
	# figure()
	imshow(carte_62, extent = bound_map_62, aspect = 'equal', interpolation = 'bilinear')
	imshow(new_cluster_density[n], extent = (xbins[0], xbins[-1], ybins[-1], ybins[0]), alpha = 0.8, aspect = 'equal', cmap = cmaps[n])
	xlim(-2.5, 2.5)
	ylim(0, 2.7)
	title(str(n))


show()





##################################################################################
# TO LABEL SHANKS WITH A NUCLEUS AUTOMATICALY
##################################################################################
for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):	
	xbins = np.arange(-1.0, 2.0, 0.1)
	ybins = np.arange(-0.2, 3.0, 0.1)[::-1]
	xpos = xbins[0:-1] + 0.05
	ypos = ybins[0:-1] - 0.05
	figure()
	imshow(cartes[m], extent = bounds[m], aspect = 'equal', interpolation = 'bilinear')
	xx, yy = np.meshgrid(xpos, ypos)
	a = 0
	for x, y in zip(xx.flatten(), yy.flatten()):
		text(x=x, y=y, s = str(a))
		a+=1
	xlim(-2.5, 2.5)
	ylim(0, 3.0)
	show()


	# cluster_density = np.zeros((n_clusters, len(ybins)-1, len(xbins)-1))
	# total_density = np.zeros((len(ybins)-1, len(xbins)-1))
	# theta_density = np.zeros((len(ybins)-1, len(xbins)-1))
	# xindex = np.digitize(subspace['x'].values.astype('float64'), xbins)
	# yindex = np.digitize(subspace['y'].values.astype('float64'), ybins)

	# for n, i, j in zip(subspace.index, yindex, xindex):
	# 	cluster_density[subspace.loc[n,'cluster'], i-1, j-1] += 1
	# 	total_density[i-1,j-1] += 1
	# 	theta_density[i-1,j-1] += theta.loc[n, 'kappa']

	# theta_density = theta_density/total_density
	# theta_density[np.isnan(theta_density)] = 0.0

	# new_cluster_density = cluster_density / (total_density + 1.0)	

	# figure(figsize = (40,10))
	# for j in range(len(cluster_density)):
	# 	tmp = new_cluster_density[j].copy()
	# 	tmp = filters.gaussian(tmp, 1.0)		
	# 	# tmp[tmp < 0.05] = np.nan		
	# 	subplot(2,3,j+1)	
		
	# 	imshow(tmp, extent = (xbins[0], xbins[-1], ybins[-1], ybins[0]), alpha = 0.8, aspect = 'equal', cmap = cmaps[j])
	# 	idx = np.intersect1d(hd_index, space.index[space.index.str.contains(m)])
	# 	scatter(subspace.loc[idx, 'x'], subspace.loc[idx, 'y'], s = 5)


show()






################################################################################
# DENSITIES OF CLUSTER ON MAPS
###############################################################################




fig = figure(figsize = (30, 10))
ax = subplot(3,3,1)
imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
for j, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	headdir 	= data['headdir']
	tmp = positions_shanks[m]
	scatter(tmp[:,:,0], tmp[:,:,1], s = headdir*30., label = m, marker = markers[j], color = 'black', alpha = 0.7)

legend()
xlim(-3, 3)
ylim(0, 3.5)
subplot(3,3,2)
for i in np.unique(space['cluster']):		
	scatter(space[space['cluster'] == i][0], space[space['cluster'] == i][1], c = colors[i], s = 5)	
subplot(3,3,3)
imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
for n in range(n_clusters):	
	contour(new_cluster_density[n], extent = (xpos[0], xpos[-1], ypos[0], ypos[-1]), alpha = 1., aspect = 'equal', cmap = cmaps[n])
xlim(-3, 3)
ylim(0, 3.5)
subplot(3,3,4)
imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
# plot(np.arange(-1, 3, 0.1), (np.sin(bestrho)/np.cos(bestrho))*np.arange(-1, 3, 0.1) + bestoffset)
legend()
xlim(-3, 3)
ylim(0, 3.5)
for n in range(n_clusters):
	subplot(3,3,n+5)
	# figure()
	imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
	imshow(new_cluster_density[n], extent = (xpos[0], xpos[-1], ypos[-1], ypos[0]), alpha = 0.8, aspect = 'equal', cmap = cmaps[n], interpolation = 'bilinear')
	xlim(-3, 3)
	ylim(0, 3.5)

show()























################################################################################
# SAVING DATA FOR FIGURE ARTICLE
################################################################################
space['hd'] = 0
space.loc[hd_index, 'hd'] = 1
space.to_hdf("../figures/figures_articles/figure1/space.hdf5", key = 'space', mode = 'w')
autocorr.to_hdf("../figures/figures_articles/figure1/autocorr.hdf5", key = 'autocorr', mode = 'w')
cPickle.dump({'shifts':shifts, 'angles':angles, 'scales':scales}, open("../figures/figures_articles/figure1/shifts.pickle", 'wb'))
cPickle.dump({'cluster_dens':new_cluster_density}, open("../figures/figures_articles/figure1/cluster_density.pickle", 'wb'))
cPickle.dump({'pos_shank':positions_shanks}, open("../figures/figures_articles/figure1/positions_shanks.pickle", 'wb'))


################################################################################
# BEST AXIS
################################################################################
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

data = autocorr.values.T
pca = PCA(n_components = 2)
compo = pca.fit_transform(data)
x2bins = np.arange(0.0, 2.4, 0.4)
y2bins = np.arange(0.6, 3.4, 0.4)[::-1]
x2pos = x2bins[0:-1] + (x2bins[1] - x2bins[0])/2.
y2pos = y2bins[0:-1] + (y2bins[1] - y2bins[0])/2.
pcdensity = np.zeros((len(y2pos), len(x2pos)))
x2index = np.digitize(space['x'].values.astype('float64'), x2bins)
y2index = np.digitize(space['y'].values.astype('float64'), y2bins)
for i in range(len(y2bins)-1):
	for j in range(len(x2bins)-1):
		idx = np.where((x2index == j+1) & (y2index == i+1))[0]
		if len(idx):
			pcdensity[i,j] = np.mean(compo[idx,0])

xvect, yvect = np.gradient(pcdensity, edge_order = 1)
bestrho = np.arctan(np.sum(yvect)/np.sum(xvect))
bestoffset = 0.0

X2, Y2 = np.meshgrid(x2pos, y2pos)
X2 = X2.flatten()
Y2 = Y2.flatten()
A = np.c_[X2, Y2, np.ones(X2.shape[0])]
C, _, _, _ = scipy.linalg.lstsq(A, pcdensity.flatten())
Z = C[0]*X2 + C[1]*Y2 + C[2]

fig = figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X2.reshape(pcdensity.shape), Y2.reshape(pcdensity.shape), Z.reshape(pcdensity.shape), rstride=1, cstride=1, alpha=0.2)
ax.plot_surface(X2.reshape(pcdensity.shape), Y2.reshape(pcdensity.shape), pcdensity, rstride=1, cstride=1, alpha=0.2)
# ax.scatter(X2, Y2, pcdensity.flatten(), c='r', s=50)
ax.axis('equal')
ax.axis('tight')
show()

subplot(121)
imshow(pcdensity, extent = (x2bins.min(), x2bins.max(), y2bins.min(), y2bins.max()))
[axhline(i) for i in y2bins]
[axvline(i) for i in x2bins]
scatter(space['x'], space['y'])
subplot(122)
imshow(pcdensity, extent = (x2bins.min(), x2bins.max(), y2bins.min(), y2bins.max()))
quiver(x2pos, y2pos, xvect, yvect)
alpha = 0.005
arrow(np.mean(x2pos), np.mean(y2pos), np.sum(xvect)*alpha, np.sum(yvect)*alpha, head_width = 0.1, linewidth = 2)

show()

figure()
for i, n in enumerate(space.index):
	# annotate(space.loc[n, 'nucleus'], (space.loc[n,'x'], compo[i,0]), color = colors[space.loc[n,'cluster']])
	plot(space.loc[n,'x'], compo[i,0], 'o', color = colors[space.loc[n,'cluster']])


for i in range(5):
	scatter(compo[klu == i,0], compo[klu == i,1], color = colors[i], label = str(i))
legend()


scatter(space['x'], compo[:,0], c = space['cluster'])

scatter(compo[:,0], compo[:,1], c = space['cluster'])

# gradient2 = []
# # xypos centered on the middle bins
# xypos = np.meshgrid(x2pos - (x2pos.min()+(x2pos.max()-x2pos.min())/2), y2pos - (y2pos.min() + ((y2pos.max() - y2pos.min())/2)))
# xypos = np.vstack((xypos[0].flatten(), xypos[1].flatten())).T

# def findaxis(rho):
# 	a = np.tan(rho)
# 	b = 0.0
# 	newxline = a * (xypos[:,1] + (xypos[:,0] / a) - b) / (a**2. + 1)
# 	newyline = a * newxline + b
# 	xyline = np.vstack((newxline, newyline)).T
# 	distto0 = np.sqrt(np.power(newxline, 2.0) + np.power(newyline - b, 2.0))
# 	score = pearsonr(compo[:,0], distto0)[0]
# 	gradient2.append(score)
# 	return score


# axismap0 = dict({'rho':np.arange(0, 2*np.pi, 0.01),
# 				'b':np.arange(0.001, 0.5, 0.1)})		

# best_params, best_score, score_results, history, logbook = maximize(findaxis, axismap0, generations_number = 2000, verbose = True)
# bestrho = best_params['rho']
# bestoffset = best_params['b']

# bb = np.arange(0.0, 3.1, 1.0)
# rho_ = np.arange(0, 2*np.pi, 0.01)
# allscore = np.zeros((len(bb),len(rho_)))
# subplot(111,projection = 'polar')
# for i, b in enumerate(bb):
# 	for j, r in enumerate(rho_):
# 		allscore[i,j] = findaxis(r, b)
# 	plot(rho_, allscore[i], label = str(b))
# legend()
# show()


fig = figure(figsize = (30, 10))
ax = subplot(3,3,1)
imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
for j, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	headdir 	= data['headdir']
	tmp = positions_shanks[m]
	scatter(tmp[:,:,0], tmp[:,:,1], s = headdir*30., label = m, marker = markers[j], color = 'black', alpha = 0.7)

legend()
xlim(-3, 3)
ylim(0, 3.5)
subplot(3,3,2)
for i in np.unique(space['cluster']):		
	scatter(space[space['cluster'] == i][0], space[space['cluster'] == i][1], c = colors[i], s = 5)	
subplot(3,3,3)
imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
for n in range(n_clusters):	
	contour(new_cluster_density[n], extent = (xpos[0], xpos[-1], ypos[0], ypos[-1]), alpha = 1., aspect = 'equal', cmap = cmaps[n])
xlim(-3, 3)
ylim(0, 3.5)
subplot(3,3,4)
imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
# plot(np.arange(-1, 3, 0.1), (np.sin(bestrho)/np.cos(bestrho))*np.arange(-1, 3, 0.1) + bestoffset)
legend()
xlim(-3, 3)
ylim(0, 3.5)
for n in range(n_clusters):
	subplot(3,3,n+5)
	# figure()
	imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
	imshow(new_cluster_density[n], extent = (xpos[0], xpos[-1], ypos[-1], ypos[0]), alpha = 0.8, aspect = 'equal', cmap = cmaps[n], interpolation = 'bilinear')
	xlim(-3, 3)
	ylim(0, 3.5)

# savefig("../figures/mapping_to_align/figure_clustering_3.pdf", bbox_inches = 'tight', dpi = 400)



sys.exit()


figure()
subplot(121)
imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
plot(np.arange(-3, 3, 0.1), (fittedaxis[0]/fittedaxis[1])*np.arange(-3, 3, 0.1) + (fittedaxis[2]/fittedaxis[1]))
legend()
xlim(-3, 3)
ylim(0, 3.5)
subplot(122)
scatter(compo[:,0], disttocenter)
xlabel('components')
ylabel('map')
legend()
show()









####################################################################################
# COLOR GRADIENTS DV ML
####################################################################################
fig = figure()
ax = subplot(131)
imshow(np.tile(np.abs(np.linspace(-1, 1, 1000))[np.newaxis], (10,1)), extent = (-0.5, 0.5, 3.0, 3.05), aspect = 'equal', cmap = 'summer')
imshow(np.tile(np.abs(np.linspace(1, 0, 1000))[np.newaxis], (10,1)).T, extent = (2, 2.05, 1, 2), aspect = 'equal', cmap = 'winter')
imshow(carte, extent = bound_map, aspect = 'equal', interpolation = 'bilinear')
for m in hd_densities.keys():
	imshow(hd_densities[m], extent = (0+offsets[m][0], xylengths[m][0]+offsets[m][0], 0+offsets[m][1], xylengths[m][1]+offsets[m][1]), cmap = 'BuGn', alpha = 0.6, aspect = 'equal', interpolation = 'bilinear')
# scatter(new_shank_pos[:,:,0], new_shank_pos[:,:,1], c = 'black', marker = '.', s = 2)
xlim(-2.5, 2.5)
ylim(0, 3.25)
# MEDIAL LATERAL GRADIENT
ax = subplot(132)
scatter(space[0], space[1], c = gradientcolors['ml'], cmap = 'summer')
title("ml")
# DORSO VENTRAL GRADIENT
ax = subplot(133)
scatter(space[0], space[1], c = gradientcolors['dv'], cmap = 'winter')
title("dv")



