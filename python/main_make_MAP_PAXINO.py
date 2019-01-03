	#!/usr/bin/env python

'''
	File name: main_make_MAP_PAXINO.py
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
from skimage.filters import gaussian


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

swr_powr = np.sqrt(np.power(swr[-500:500], 2.0).sum(0))

swr_powr.to_hdf("../figures/figures_articles/figure2/power_ripples_2.h5", 'rippower')




###############################################################################################################
# THETA MODULATION
###############################################################################################################
# theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
# theta 					= pd.DataFrame(	index = theta_ses['rem'], 
# 									columns = ['phase', 'pvalue', 'kappa'],
# 									data = theta_mod['rem'])

theta2 = pd.read_hdf("/mnt/DataGuillaume/MergedData/THETA_THAL_mod_2.h5")

theta = theta2['rem']
theta.rename(columns={'pval':'pvalue'}, inplace=True)



######################################################################################
# LOADING THE MAP
######################################################################################
carte38_mouse17 = imread('../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)


####################################################################################
# SHANK MAPPING from main_make_nucleus_set.py
####################################################################################
mapping_nucleus = cPickle.load(open("../data/maps/mapping_nucleus_allen.pickle", 'rb'))
neuron_to_shank = mapping_nucleus['neuron_to_shank']
neuron_to_channel = mapping_nucleus['neuron_to_channel']

neurons = np.intersect1d(theta.index.values, swr_mod.index.values)
session_shank = np.array([list(neuron_to_shank[n]) for n in neurons])
session_channel = np.array([neuron_to_channel[n] for n in neurons])


space = pd.DataFrame(index = neurons, columns = ['session', 'shank'])
space['session'] = session_shank[:,0]
space['shank'] = 7 - session_shank[:,1]

sys.exit()
##################################################################################
# DOING IT BY HAND
##################################################################################
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

for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
# for i, m in zip([1], ['Mouse17']):
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

	total_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	theta_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	neg_swr_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	pos_swr_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))
	neu_swr_density = np.zeros((len(shanks_pos[m][1]), len(shanks_pos[m][0])))

	p40 = np.percentile(swr.loc[0.0].values, 40)
	p60 = np.percentile(swr.loc[0.0].values, 60)
	for n in subspace.index.values:		
		total_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += 1.0
		theta_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += float(theta.loc[n,'pvalue']<0.01)
		neg_swr_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += float(swr.loc[0,n]<p40)
		pos_swr_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += float(swr.loc[0,n]>p60)
		neu_swr_density[subspace.loc[n,'session'],subspace.loc[n,'shank']] += float(np.logical_and(swr.loc[0,n]>p40, swr.loc[0,n]<p60))

	print(m, p40, p60)


	# clusters | theta | pos swr | neg swr | frate wake | frate rem | frate sws | burst_wake | burst_rem | burst_sws | count
	images = np.vstack((theta_density[np.newaxis,:,:],pos_swr_density[np.newaxis,:,:],neg_swr_density[np.newaxis,:,:],neu_swr_density[np.newaxis,:,:]))
	images = images / total_density
	images[np.isnan(images)] = 0.0
	
	new_images = []
	for k in range(len(images)):
		xnew, ynew, tmp = interpolate(images[k].copy(), shanks_pos[m][0], shanks_pos[m][1], 0.010)				
		tmp2 = gaussian(tmp, sigma = 10.0, mode = 'reflect')		
		tmp3 = softmax(tmp2, 10.0)
		new_images.append(tmp3)
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

	################################################################################
	# SAVING DATA FOR FIGURE ARTICLE
	################################################################################
	# store = pd.HDFStore('../figures/figures_articles/figure2/modulation_theta2_swr_'+m+'.h5', 'a')
	# _, h, w = rotated_images.shape
	# for i, n in zip(range(len(rotated_images)), ['theta', 'pos_swr', 'neg_swr', 'neu_swr']):
	# 	tmp = pd.DataFrame(	index = np.linspace(bound[3], bound[2], h), 
	# 						columns = np.linspace(bound[0], bound[1], w), 
	# 						data = rotated_images[i])
	# 	store[n] = tmp
	# store.close()





