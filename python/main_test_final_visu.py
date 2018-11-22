import ternary
import numpy as np
import pandas as pd
from functions import *
import sys
from functools import reduce
from sklearn.manifold import *
from sklearn.cluster import *
from pylab import *
import _pickle as cPickle
from skimage.filters import gaussian

############################################################################################################
# LOADING DATA
############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
burstiness 				= pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
lambdaa  = pd.read_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_AUTOCORR.h5")[('rem', 'b')]
lambdaa = lambdaa[np.logical_and(lambdaa>0.0,lambdaa<30.0)]



theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta 					= pd.DataFrame(	index = theta_ses['rem'], 
									columns = ['phase', 'pvalue', 'kappa'],
									data = theta_mod['rem'])
rippower 				= pd.read_hdf("../figures/figures_articles/figure2/power_ripples_2.h5")
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
swr_phase = pd.read_hdf("/mnt/DataGuillaume/MergedData/SWR_PHASE.h5")

# SWR MODULATION
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
swr = swr.loc[-200:200]

# AUTOCORR FAST
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")
autocorr_wak = store_autocorr['wake'].loc[0.5:]
autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
autocorr_sws = 	store_autocorr['sws'].loc[0.5:]
autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_wak = autocorr_wak[2:150]
autocorr_rem = autocorr_rem[2:150]
autocorr_sws = autocorr_sws[2:150]

# HISTOGRAM THETA
theta_hist = pd.read_hdf("/mnt/DataGuillaume/MergedData/THETA_THAL_HISTOGRAM_2.h5")
theta_hist = theta_hist.rolling(window = 5, win_type='gaussian', center = True, min_periods=1).mean(std=1.0)
theta_wak = theta_hist.xs(('wak'), 1, 1)
theta_rem = theta_hist.xs(('rem'), 1, 1)

# AUTOCORR LONG
store_autocorr2 = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG.h5")
autocorr2_wak = store_autocorr2['wak'].loc[0.5:]
autocorr2_rem = store_autocorr2['rem'].loc[0.5:]
autocorr2_sws = store_autocorr2['sws'].loc[0.5:]
autocorr2_wak = autocorr2_wak.rolling(window = 100, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 10.0)
autocorr2_rem = autocorr2_rem.rolling(window = 100, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 10.0)
autocorr2_sws = autocorr2_sws.rolling(window = 100, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 10.0)
autocorr2_wak = autocorr2_wak[2:2000]
autocorr2_rem = autocorr2_rem[2:2000]
autocorr2_sws = autocorr2_sws[2:2000]

############################################################################################################
# WHICH NEURONS
############################################################################################################
firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
fr_index = firing_rate.index.values[((firing_rate >= 1.0).sum(1) == 3).values]
# neurons = reduce(np.intersect1d, (burstiness.index.values, theta.index.values, rippower.index.values, fr_index))
neurons = reduce(np.intersect1d, (autocorr_sws.columns, autocorr2_rem.columns, theta_rem.columns, swr.columns, lambdaa.index.values))

# neurons = np.array([n for n in neurons if 'Mouse17' in n])

nucleus = ['AD', 'AM', 'AVd', 'AVv', 'VA', 'LDvl', 'CM']
# neurons = np.intersect1d(neurons, mappings.index[mappings['nucleus'].isin(nucleus)])

###########################################################################################################
# AVERAGING OVER SHANKS
###########################################################################################################

shank_swr 			= pd.DataFrame(index = swr.index)
shank_autoc_rem 	= pd.DataFrame(index = autocorr_rem.index)
shank_autoc_wak 	= pd.DataFrame(index = autocorr_wak.index)
shank_autoc_sws 	= pd.DataFrame(index = autocorr_sws.index)
shank_autoc2_rem 	= pd.DataFrame(index = autocorr2_rem.index)
shank_autoc2_wak 	= pd.DataFrame(index = autocorr2_wak.index)
shank_thetahist_wak = pd.DataFrame(index = theta_hist.index)
shank_thetahist_rem = pd.DataFrame(index = theta_hist.index)
animals = []
index = []
nuc = []
for m in ['Mouse12','Mouse17','Mouse20','Mouse32']:
	groups = mappings.loc[neurons][mappings.loc[neurons].index.str.contains(m)].groupby(['shank','session']).groups
	for k in groups.keys():
		# if len(groups[k])>=3:
			shank_swr[(m,k[0],k[1])] 	= swr[groups[k]].mean(1)
			shank_autoc_rem[(m,k[0],k[1])] 			= autocorr_rem[groups[k]].mean(1)
			shank_autoc_wak[(m,k[0],k[1])] 			= autocorr_wak[groups[k]].mean(1)
			shank_autoc_sws[(m,k[0],k[1])] 			= autocorr_sws[groups[k]].mean(1)
			shank_autoc2_rem[(m,k[0],k[1])] 			= autocorr2_rem[groups[k]].mean(1)
			shank_autoc2_wak[(m,k[0],k[1])] 			= autocorr2_wak[groups[k]].mean(1)
			shank_thetahist_wak[(m,k[0],k[1])] 		= theta_hist[groups[k]].xs('wak',1,1).mean(1)
			shank_thetahist_rem[(m,k[0],k[1])] 		= theta_hist[groups[k]].xs('rem',1,1).mean(1)
			animals.append(m)
			index.append((m,k[0],k[1]))
			nucnuc = np.unique(mappings.loc[groups[k],'nucleus'])
			if len(nucnuc)>1: sys.exit()
			nuc.append(nucnuc[0])

shank_info 			= pd.DataFrame(index = index, columns = ['mouse', 'nucleus'])
shank_info['mouse'] = animals
shank_info['nucleus'] = nuc
############################################################################################################
# STACKING DIMENSIONS
############################################################################################################
from sklearn.decomposition import PCA


pc_short_rem = PCA(n_components=3).fit_transform(shank_autoc_rem.values.T)
pc_short_wak = PCA(n_components=3).fit_transform(shank_autoc_wak.values.T)
pc_short_sws = PCA(n_components=3).fit_transform(shank_autoc_sws.values.T)
# pc_short_rem = np.log((pc_short_rem - pc_short_rem.min(axis = 0))+1)
# pc_short_wak = np.log((pc_short_wak - pc_short_wak.min(axis = 0))+1)
# pc_short_sws = np.log((pc_short_sws - pc_short_sws.min(axis = 0))+1)
pc_long_wak = PCA(n_components=3).fit_transform(shank_autoc2_wak.values.T)
pc_long_rem = PCA(n_components=3).fit_transform(shank_autoc2_rem.values.T)
# pc_long = np.log((pc_long - pc_long.min(axis=0))+1) 
# pc_long = np.log((pc_long - pc_long.min(axis=0))+1) 
# pc_long = np.log(lambdaa.loc[neurons].values[:,np.newaxis])
# pc_theta = np.hstack([np.cos(theta.loc[neurons,'phase']).values[:,np.newaxis],np.sin(theta.loc[neurons,'phase']).values[:,np.newaxis],np.log(theta.loc[neurons,'kappa'].values[:,np.newaxis])])
# pc_theta = np.hstack([np.log(theta.loc[neurons,'kappa'].values[:,np.newaxis])])
pc_thta_rem = PCA(n_components = 3).fit_transform(shank_thetahist_rem.values.T)
pc_thta_wak = PCA(n_components = 3).fit_transform(shank_thetahist_wak.values.T)
# pc_theta = PCA(n_components=3).fit_transform(theta_rem[neurons].values.T)
# pc_theta = np.log((pc_theta - pc_theta.min(axis = 0))+1)
# pc_swr   = np.hstack([np.log(rippower.loc[neurons].values[:,np.newaxis])])
pc_swr 	 = PCA(n_components=3).fit_transform(shank_swr.values.T)
# pc_swr 	 = np.log((pc_swr - pc_swr.min(axis = 0))+1)
# pc_theta -= pc_theta.min(axis = 0)
# pc_swr 	 -= pc_swr.min(axis = 0)
# pc_theta = np.log(pc_theta+1)
# pc_swr 	 = np.log(pc_swr+1)

data = np.hstack([pc_short_rem, pc_short_sws, pc_short_wak, pc_long_wak, pc_long_rem, pc_thta_rem, pc_thta_wak, pc_swr])

# data = np.vstack([
# 					shank_swr.values,
# 					shank_autoc_rem.values,
# 					shank_autoc_wak.values,
# 					shank_autoc_sws.values,
# 					shank_autoc2_rem.values,
# 					shank_autoc2_wak.values,
# 					shank_thetahist_rem.values,
# 					shank_thetahist_wak.values
# 				]).T



tsne = TSNE(n_components = 2, perplexity = 6).fit_transform(data)
# tsne = LocallyLinearEmbedding(n_neighbors=20,n_components=2).fit_transform(data)
# tsne = Isomap(n_neighbors=20,n_components=2).fit_transform(data)
# tsne = SpectralEmbedding(n_components=2,n_neighbors=50).fit_transform(data)
allnucleus = list(np.unique(shank_info['nucleus'].values))
label = np.array([allnucleus.index(n) for n in shank_info['nucleus'].values])

scatter(tsne[:,0], tsne[:,1], c= label)
show()

#########################################################################################
# CLUSTERING
#########################################################################################
# klu = KMeans(n_clusters=5).fit(data).labels_
n_clusters = 4
klu = AgglomerativeClustering(n_clusters=n_clusters).fit(data).labels_

shank_info['AC'] = klu

# names = mappings.loc[neurons, 'nucleus'].copy()
# nuccolors = names.replace(to_replace=np.unique(names), value=np.arange(len(np.unique(names))))

scatter(tsne[:,0], tsne[:,1], c = klu)







#########################################################################################
# PROJECTING IN SPACE
#########################################################################################
imageklu 	= {}
shanks_pos = {}
ymax = {}
imagehd = {}
for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):	
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	x 			= data['x']
	y 			= data['y']*-1.0+np.max(data['y'])		
	ymax[m] 	= y.max()
	shanks_pos[m] = (data['x'], data['y'])
	imageklu[m] = np.zeros((len(np.unique(klu)),len(data['y']),len(x)))
	imagehd[m] = np.zeros((len(data['y']),len(data['x'])))
	for n in mappings[mappings.index.str.contains(m)].index:
		imagehd[m][mappings.loc[n,'session'],mappings.loc[n,'shank']] += mappings.loc[n,'hd']
	imagehd[m] /= (data['total'] + 1)

	for j, k in enumerate(np.unique(klu)):
		idx = np.where((mappings['AC'] == k).values & mappings['AC'].index.str.contains(m))[0]
		for n in mappings.iloc[idx].index:
			imageklu[m][j,mappings.loc[n,'session'],mappings.loc[n,'shank']] += 1
		imageklu[m][j] /= (data['total'] + 1)

carte38_mouse17 = imread('../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
shifts = np.array([	[-0.34, 0.56],
					[0.12, 0.6],
					[-0.35, 0.75],
					[-0.3, 0.5]
				])
angles = np.array([15.0, 10.0, 15.0, 20.0])

mapcolors = ['Blues', 'Reds', 'Greens', 'Oranges', 'Greys', 'Magenta', 'Yellow']
colors = ['blue', 'red', 'green', 'orange', 'grey', 'magenta', 'yellow']

figure()
gs = GridSpec(4,n_clusters)
for i, m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):	
	for j, k in enumerate(np.unique(klu)):
		image = imageklu[m][j].copy()
		xnew, ynew, tmp = interpolate(image, shanks_pos[m][0], shanks_pos[m][1], 0.010)		
		tmp2 = gaussian(tmp, sigma = 15.0, mode = 'reflect')
		
		h, w = tmp2.shape
		tmp3 = np.zeros((h*3, w*3))*np.nan
		tmp3[h:h*2,w:w*2] = tmp2.copy() + 1.0		
		tmp3 = rotateImage(tmp3, -angles[i])
		tmp3[tmp3 == 0.0] = np.nan
		tmp3 -= 1.0

		tocrop = np.where(~np.isnan(tmp3))
		tmp3 = tmp3[tocrop[0].min()-1:tocrop[0].max()+1,tocrop[1].min()-1:tocrop[1].max()+1]
		xlength, ylength = getXYshapeofRotatedMatrix(shanks_pos[m][0].max(), shanks_pos[m][1].max(), angles[i])

		bound = (shifts[i][0],xlength+shifts[i][0],shifts[i][1],ylength+shifts[i][1])

		subplot(gs[i,j])		
		imshow(carte38_mouse17, extent = bound_map_38, aspect = 'equal', interpolation = 'bilinear')
		imshow(tmp3, extent = bound, alpha = 0.8, aspect = 'equal', cmap = mapcolors[j])
		ylabel(m)		
		xlim(-2.5, 2.5)
		ylim(0, 3.0)



figure()
subplot(3,3,1)
scatter(tsne[:,0], tsne[:,1], 30, c = nuccolors)
scatter(tsne[mappings.loc[neurons,'hd']==1][:,0], tsne[mappings.loc[neurons,'hd']==1][:,1], 2, color = 'white')
title("nucleus")

subplot(3,3,2)
scatter(tsne[:,0], tsne[:,1], 30, c = np.log(theta.loc[neurons,'kappa'].values))
title("theta")

subplot(3,3,3)
scatter(tsne[:,0], tsne[:,1], 30, c = np.log(rippower.loc[neurons].values))
title("rip power")

subplot(3,3,4)
scatter(tsne[:,0], tsne[:,1], 30, c = np.log(burstiness.loc[neurons,'sws']))
title("burstiness")

subplot(3,3,5)
scatter(tsne[:,0], tsne[:,1], 30, c = np.log(lambdaa.loc[neurons]))
title("slow")

subplot(3,3,6)
scatter(tsne[:,0], tsne[:,1], 30, c = theta.loc[neurons,'pvalue'].values)
title("theta")

subplot(3,3,7)
scatter(tsne[:,0], tsne[:,1], 30, c = theta.loc[neurons,'phase'].values)
title("phase theta")

subplot(3,3,8)
scatter(tsne[:,0], tsne[:,1], 30, c = swr.loc[0.0, neurons].values)
title("ripple 0 ")

subplot(3,3,9)
for i in np.arange(n_clusters):
	scatter(tsne[klu==i,0], tsne[klu==i,1], 30, c = colors[i])
title("KMEANS")


figure()
for i, n in enumerate(nucleus):
	subplot(2,4,i+1)
	title(n)
	idx = np.where(mappings.loc[neurons,'nucleus'] == n)[0]
	idx2 = np.where(mappings.loc[neurons,'nucleus'] != n)[0]
	scatter(tsne[idx2,0], tsne[idx2,1], 30, c = 'grey')
	scatter(tsne[idx,0], tsne[idx,1], 40, c = colors[i])
	

show()
