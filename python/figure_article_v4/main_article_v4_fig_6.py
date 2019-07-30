#!/usr/bin/env python

import sys
sys.path.append("../")
import numpy as np
import pandas as pd

import scipy.io
from functions import *

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
import os
from scipy.misc import imread


space = pd.read_hdf("../../figures/figures_articles_v2/figure1/space.hdf5")

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
burst = burst.loc[space.index]


# autocorr = pd.read_hdf("../../figures/figures_articles_v2/figure1/autocorr.hdf5")
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")

# carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
# carte38_mouse17_2 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
# bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
# cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

carte_adrien = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_ALL-01.png')
carte_adrien2 = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_Contour-01.png')
bound_adrien = (-398/1254, 3319/1254, -(239/1254 - 20/1044), 3278/1254)

# carte_adrien2[:,:,-1][carte_adrien2[:,:,-1]<150] = 0.0

tmp = cPickle.load(open("../../figures/figures_articles_v2/figure1/shifts.pickle", 'rb'))
angles = tmp['angles']
shifts = tmp['shifts']


hd_index = space.index.values[space['hd'] == 1]

neurontoplot = [np.intersect1d(hd_index, space.index.values[space['cluster'] == 1])[0],
				burst.loc[space.index.values[space['cluster'] == 0]].sort_values('sws').index[3],
				burst.sort_values('sws').index.values[-20]]






###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*0.5         # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.set_xticks([])
	ax.set_yticks([])
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)


import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	# "text.usetex": True,                # use LaTeX to write all text
	# "font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 7,
	"ytick.labelsize": 7,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.8,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}    
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

colors = ['red', 'green', 'blue', 'purple', 'orange']
cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']
markers = ['o', '^', '*', 's']

fig = figure(figsize = figsize(1.0))#, tight_layout=True)

outergs = gridspec.GridSpec(1,3, figure = fig, hspace = 0.4, wspace = 0.4)



# SPEFICIC TO SLOW DYNAMIC
###############################
# SPECIFIC TO SLOW DYNAMIC
count_nucl = pd.DataFrame(columns = ['12', '17','20', '32'])

for m in ['12', '17','20', '32']:
	subspace = pd.read_hdf("../../figures/figures_articles/figure1/subspace_Mouse"+m+".hdf5")	
	nucleus = np.unique(subspace['nucleus'])		
	total = [np.sum(subspace['nucleus'] == n) for n in nucleus]
	count_nucl[m] = pd.Series(index = nucleus, data = total)	
nucleus = list(count_nucl.dropna().index.values)
nucleus.remove('sm')

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
lambdaa  = pd.read_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_AUTOCORR.h5")

hd_neurons = mappings[mappings['hd'] == 1].index.values

lambdaa_nucleus = pd.DataFrame(	index = nucleus, 
								columns = pd.MultiIndex.from_product([['wak', 'rem', 'sws'], ['mean', 'sem']], 
								names = ['episode', 'mean-sem']))
for n in nucleus:
	tmp = lambdaa.loc[mappings.index[mappings['nucleus'] == n]].dropna()
	tmp = tmp.xs(('b'), 1, 1)
	tmp = tmp[((tmp>0.0).all(1) & (tmp<3.0).all(1))]
	for e in ['wak', 'rem', 'sws']:
		lambdaa_nucleus.loc[n,(e,'mean')] = tmp[e].mean(skipna=True)
		lambdaa_nucleus.loc[n,(e,'sem')] = tmp[e].sem(skipna=True)

data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta 					= pd.DataFrame(	index = theta_ses['rem'], 
									columns = ['phase', 'pvalue', 'kappa'],
									data = theta_mod['rem'])


# gs = gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec = outergs[2,:], width_ratios=[0.005,1,1,1], wspace = 0.4)#, hspace = 0.5, wspace = 0.2)
###################################################################################################
# A. AUTOCORRELOGRAM EXEMPLE
###################################################################################################
autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG.h5")

axI = Subplot(fig, outergs[0,0])
fig.add_subplot(axI)
simpleaxis(axI)

def func(x, a, b, c):
	return a*np.exp(-(1./b)*x) + c


labels = ['HD', 'non-HD', 'non-HD']
# colors = ['green', 'blue', 'red']

colors = ['red', 'black', 'gray']

# hd = 'Mouse12-120807_28'
# iad = 'Mouse12-120817_52'
# nhd = 'Mouse12-120819_3'
###########
index = mappings[np.logical_and(mappings['hd'] == 1, mappings['nucleus'] == 'AD')].index.values
best = (lambdaa.loc[index,('wak','b')] - lambdaa_nucleus.loc['AD', ('wak', 'mean')]).dropna().abs().sort_values().index.values
hd = best[0]

index = mappings[np.logical_and(mappings['hd'] == 0, mappings['nucleus'] == 'AVd')].index.values
best = (lambdaa.loc[index,('wak','b')] - lambdaa_nucleus.loc['AVd', ('wak', 'mean')]).dropna().abs().sort_values().index.values
nhd = best[0]

index = mappings[np.logical_and(mappings['hd'] == 0, mappings['nucleus'] == 'IAD')].index.values
best = (lambdaa.loc[index,('wak','b')] - lambdaa_nucleus.loc['IAD', ('wak', 'mean')]).dropna().abs().sort_values().index.values
iad = best[0]

hd = neurontoplot[0]	

for i, n in enumerate([hd, nhd, iad]):	
	tmp = autocorr['wak'][n].copy()
	tmp.loc[0] = 0.0
	tmp = tmp.loc[tmp.loc[0.1:25.0].argmax():]
	tmp2 = tmp.rolling(window = 100, win_type='gaussian', center=True, min_periods=1).mean(std=5.0)
	tmp3 = tmp2 - tmp2.min()
	tmp3 = tmp3 / tmp3.max()

	tmp3 = tmp3.loc[:2500]

	plot(tmp3.index.values*1e-3, tmp3.values, label = labels[i], color = colors[i])
	x = tmp3.index.values*1e-3
	y = func(x, *lambdaa.loc[n, 'wak'].values)
	if i == 2:
		plot(x, y, '--', color = 'grey', label = "Exp. fit \n " r"$y = a \ exp(-t/\tau)$")
	else:
		plot(x, y, '--', color = 'grey')

# show()


legend(edgecolor = None, facecolor = None, frameon = False, bbox_to_anchor=(0.3, 1.1), bbox_transform=axI.transAxes)
xlabel("Time lag (s)")
ylabel("Autocorrelation (a.u)")
locator_params(nbins = 4)

axI.text(-0.3, 1.0, "a", transform = axI.transAxes, fontsize = 10, fontweight='bold')

###################################################################################################
# B. WAKE TAU VS REM TAU
###################################################################################################
axJ = Subplot(fig, outergs[0,1])
fig.add_subplot(axJ)
simpleaxis(axJ)

x = lambdaa[('wak', 'b')].values
y = lambdaa['rem', 'b'].values
tlim = 5.0
idx = (x>=0.0) * (x<tlim) * (y>=0.0) * (y<tlim)
alln = lambdaa.index.values[idx]
slope, intercept, rvalue, pvalue, stderr  = scipy.stats.linregress(x[idx], y[idx])
print("REM, WAKE ",str(rvalue),str(pvalue))
scatter(x[idx], y[idx], s= 4, alpha = 0.5, color = 'grey', linewidth = 0)
# hd neurons
hdn = np.intersect1d(lambdaa.index.values[idx], hd_neurons) 
scatter(lambdaa.loc[hdn,('wak','b')], lambdaa.loc[hdn,('rem','b')], s = 4, color = 'red', linewidth = 0)

xx = np.arange(0, tlim, 1)
plot(xx, slope*xx + intercept, '--', color = 'green')
xlim(0, tlim)
ylim(0, tlim)

xlabel(r"Wake decay time $\tau$ (s)")
ylabel(r"REM decay time $\tau$ (s)")

axJ.text(0.25, 1.0, "r="+str(np.round(rvalue, 3))+" (p<0.001)", transform = axJ.transAxes, fontsize = 8)
axJ.text(-0.3, 1.0, "b", transform = axJ.transAxes, fontsize = 10, fontweight='bold')

###################################################################################################
# C. BURSTINESS VS LAMBDA
###################################################################################################
axK = Subplot(fig, outergs[0,2])
fig.add_subplot(axK)
simpleaxis(axK)

# firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
# fr_index = firing_rate.index.values[((firing_rate[['wake', 'rem']] > 1.0).sum(1) == 2).values]
from scipy.stats import pearsonr

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
idx = lambdaa['rem']['b'].index

# correlation during wake
df = pd.concat([burst['sws'].loc[idx], lambdaa['wak']['b'].loc[idx]], axis = 1).rename(columns={'sws':'burst','b':'lambda'})
df = df[np.logical_and(df['burst']<25,df['lambda']<3)]
a, b = pearsonr(df['burst'].values, df['lambda'].values)
print('wake', a, b)
# correlation during rem
df2 = pd.concat([burst['sws'].loc[idx], lambdaa['rem']['b'].loc[idx]], axis = 1).rename(columns={'sws':'burst','b':'lambda'})
df2 = df2[np.logical_and(df2['burst']<25,df2['lambda']<3)]
df2 = df2[df2['lambda'] > 0.0]
c, d = pearsonr(df2['burst'].values, df2['lambda'].values)
print('rem', c, d)


scatter(df2['burst'].values, df2['lambda'].values, 4, color = 'grey', alpha = 0.8, edgecolors = 'none')
scatter(df2.loc[hdn,'burst'].values, df2.loc[hdn,'lambda'].values, 4, color = 'red', alpha = 1, edgecolors = 'none')

slope, intercept, rvalue, pvalue, stderr  = scipy.stats.linregress(df2['burst'].values, df2['lambda'].values)

xx = np.arange(0, gca().get_xlim()[1], 1)
yy = slope*xx + intercept
plot(xx[yy>0], yy[yy>0], '--', color = 'green')

xlabel("NREM burst index")
ylabel(r"REM decay time $\tau$ (s)")

yticks([0,1,2,3],['0','1','2','3'])

xlim(0,)
ylim(0,)

axK.text(0.25, 1.0, "r="+str(np.round(c, 3))+" (p<0.001)", transform = axK.transAxes, fontsize = 8)
axK.text(-0.3, 1.0, "c", transform = axK.transAxes, fontsize = 9 ,fontweight='bold')


###################################################################################################
# D. BURSTINESS VS LAMBDA
###################################################################################################
# axK = Subplot(fig, outergs[1,0])
# fig.add_subplot(axK)
# simpleaxis(axK)

# swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
# nbins 					= 400
# binsize					= 5
# times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
# swr_mod 					= pd.DataFrame(	columns = swr_ses, 
# 										index = times,
# 										data = gaussFilt(swr_mod, (1,)).transpose())
# swr_mod = swr_mod.drop(swr_mod.columns[swr_mod.isnull().any()].values, axis = 1)
# swr_mod = swr_mod.loc[-300:300]
# swr_mod = swr_mod[alln]


# from sklearn.decomposition import PCA
# pc = PCA(n_components = 2).fit_transform(swr_mod[alln].values.T)

# scatter(lambdaa.loc[alln,('rem', 'b')], pc[:,0], s = 4)




subplots_adjust(bottom = 0.2, top = 0.93, right = 0.98, left = 0.08)

savefig("../../figures/figures_articles_v4/figart_6.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v4/figart_6.pdf &")

# # theta
# ax1 = subplot(337)

# noaxis(ax1)
# tmp = rotated_images[2]
# imshow(tmp, extent = bound, alpha = 0.9, aspect = 'equal', cmap = 'viridis')
# imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
# title('Theta', fontsize = 5)

# # pos swr
# ax1 = subplot(338)
# noaxis(ax1)
# tmp = rotated_images[3]
# imshow(tmp, extent = bound, alpha = 0.9, aspect = 'equal', cmap = 'viridis')
# imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
# title('Positive SWR\nmodulation', fontsize = 5)

# # neg swr
# ax1 = subplot(339)
# noaxis(ax1)
# tmp = rotated_images[4]
# imshow(tmp, extent = bound, alpha = 0.9, aspect = 'equal', cmap = 'viridis')
# imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
# title('Negative SWR\nmodulation', fontsize = 5)

