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

# specific to mouse 17
subspace = pd.read_hdf("../../figures/figures_articles_v2/figure1/subspace_Mouse17.hdf5")

data = cPickle.load(open("../../figures/figures_articles_v2/figure1/rotated_images_Mouse17.pickle", 'rb'))
rotated_images = data['rotated_images']
new_xy_shank = data['new_xy_shank']
bound = data['bound']

data 		= cPickle.load(open("../../data/maps/Mouse17.pickle", 'rb'))
x 			= data['x']
y 			= data['y']*-1.0+np.max(data['y'])
headdir 	= data['headdir']

xy_pos = new_xy_shank.reshape(len(y), len(x), 2)


# XGB score
mean_score = pd.read_hdf('/mnt/DataGuillaume/MergedData/'+'SCORE_XGB.h5')




###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean        # height in inches
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
	"axes.labelsize": 10,               # LaTeX default is 10pt font.
	"font.size": 10,
	"legend.fontsize": 10,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 1,
	"axes.linewidth"        : 2,
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

fig = figure(figsize = figsize(1.6))#, tight_layout=True)

outergs = gridspec.GridSpec(2,1, figure = fig, height_ratios = [1, 0.6])


#############################################
# D. TSNE
#############################################
gs = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec = outergs[0,:])

# axD = fig.add_subplot(outergs[2,1])
axD = Subplot(fig, gs[:,0])
fig.add_subplot(axD)
noaxis(axD)
sc = scatter(space[space['cluster'] == 0][1]*-1.0, space[space['cluster'] == 0][0]*-1, s = 110, c = burst['sws'][space['cluster'] == 0].values, edgecolor = 'none', alpha = 0.8, label = '_nolegend_')
# hd
scatter(space[space['cluster'] == 1][1]*-1.0, space[space['cluster'] == 1][0]*-1, s = 110, facecolor = 'red', edgecolor = 'none', alpha = 0.8, label = 'Cluster 1')
scatter(space[space['hd'] == 1][1]*-1.0, space[space['hd'] == 1][0]*-1, s = 30, marker = 'o', facecolor = 'white', edgecolor = 'black', linewidth = 0.5, label = 'HD neuron', alpha = 0.8)

# xlim(-90, 100)
# ylim(-300,80)

# legend
handles, labels = axD.get_legend_handles_labels()
axD.legend(handles[::-1], labels[::-1], 
			fancybox=False, 
			framealpha =0, 
			fontsize = 16, 
			loc = 'lower left', 
			bbox_to_anchor=(-0.14, 0.5),
			handletextpad=0.05)

#axD.text(1, 0.54, "Cluster 2", transform = axD.transAxes, fontsize = 16)



# surrounding examples
scatter(space.loc[neurontoplot,1]*-1.0, space.loc[neurontoplot,0]*-1.0, s = 25, facecolor = 'none', edgecolor = 'grey', linewidths = 2)
txts = ['a', 'b', 'c']
xyoffset = [[-10, 12], [14, 12], [8, 20]]
for i, t in zip(range(3), txts):
	x, y = (space.loc[neurontoplot[i],1]*-1, space.loc[neurontoplot[i],0]*-1)
	annotate(t, xy=(x, y), 
				xytext=(x+np.sign(x)*xyoffset[i][0],  y+np.sign(y)*xyoffset[i][1]), 
				fontsize = 16,
				arrowprops=dict(facecolor='black',								
								arrowstyle="->",
								connectionstyle="arc3")
				)


#colorbar	
cax = inset_axes(axD, "17%", "5%",
                   bbox_to_anchor=(0.3, 0.1, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burst index', labelpad = -40, fontsize = 16)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("Cluster 2", fontsize = 16, pad = -35.0)



# #############################################################
# # E MAPS
# #############################################################
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
neurons_ = mappings.index[np.where(mappings['hd'] == 1)[0]]
neurons_ = neurons_[np.where((firing_rate.loc[neurons_]>2.0).all(axis=1))[0]]
# EXample autocorr
axD = Subplot(fig, outergs[1,:])


# ###### REM ##########################################
# cax0 = inset_axes(axD, "40%", "40%",bbox_to_anchor=(-0.1, 0.6, 1, 1),
# 					bbox_transform=axD.transAxes, loc = 'lower left')
# noaxis(cax0)
# # cax0.spines['bottom'].set_visible(True)
# autocorr = store_autocorr['rem'][neurons_]
# autocorr.loc[0.0] = 0.0
# fr = firing_rate.loc[neurons_, 'rem'].sort_values()
# idx = np.arange(0, len(fr), 6)[0:-1]
# cm = get_cmap('Reds')
# cNorm = matplotlib.colors.Normalize(vmin = 0.0, vmax = fr.iloc[idx].max())
# scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap = cm)
# for n in fr.index[idx]:
# 	tmp = autocorr.loc[-100:100,n]
# 	tmp /= np.mean(tmp.loc[-100:-50])
# 	cax0.plot(tmp.loc[-50:50], color = scalarMap.to_rgba(fr.loc[n]))



# ######## SWS #########################################
# cax00 = inset_axes(axD, "40%", "40%",bbox_to_anchor=(-0.1, -0.1, 1, 1),
# 					bbox_transform=axD.transAxes, loc = 'lower left')
# noaxis(cax00)
# cax00.spines['bottom'].set_visible(True)
# xticks([-50, 0, 50], fontsize = 16)
# cax00.tick_params(axis='x', which='major', pad=1)
# xlabel("Time (ms)", labelpad=0.1, fontsize = 16)
# autocorr = store_autocorr['sws'][neurons_]
# autocorr.loc[0.0] = 0.0
# # fr = firing_rate.loc[neurons_, 'sws'].sort_values()
# # idx = np.arange(0, len(fr), 6)[0:-1]
# cm = get_cmap('Reds')
# cNorm = matplotlib.colors.Normalize(vmin = 0.0, vmax = fr.iloc[idx].max())
# scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap = cm)
# for n in fr.index[idx]:
# 	tmp = autocorr.loc[-100:100,n]
# 	tmp /= np.mean(tmp.loc[-100:-50])
# 	cax00.plot(tmp.loc[-50:50], color = scalarMap.to_rgba(fr.loc[n]))

# # cax0.text('REM', 0.5, 0.5, fontsize = 7, transform = cax0.transAxes)
# # cax00.text('NREM', 0.5, 0.5, fontsize = 7, transform = cax00.transAxes)
# cax0.text(0.05, 0.1, "REM", transform = cax0.transAxes, fontsize = 16)
# cax00.text(0.05, 0.1, "NREM", transform = cax00.transAxes, fontsize = 16)



# ###### color bar firing rate
# cax = inset_axes(axD, "1.5%", "15%",
#                    bbox_to_anchor=(0.3, 0.35, 1, 1),
#                    bbox_transform=axD.transAxes, 
#                    loc = 'lower left')

# cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm = cNorm, orientation = 'vertical')
# # cax.set_title("Rate (Hz)", fontsize = 7, pad = 3)
# cb.set_label("Rate (Hz)", fontsize = 16)
# # cax0.text(0.48,1.3, 'Rate (Hz)', transform = cax0.transAxes, fontsize = 7)
# cb.ax.xaxis.set_tick_params(pad = 1)
# cb.ax.yaxis.set_ticks_position('left')
# cb.ax.tick_params(labelsize=6) 



######## cluster 1 HD ################################
cax1 = inset_axes(axD, "160%", "160%",
                   bbox_to_anchor=(-0.7, -0.25, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')
tmp = rotated_images[1]
tmp[tmp<0.0] = 0.0
cax1.imshow(tmp, extent = bound, alpha = 1, aspect = 'equal', cmap = 'Reds')
cax1.imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
noaxis(cax1)
cax1.patch.set_facecolor('none')
#colorbar	
cax = inset_axes(cax1, "30%", "5%",
                   bbox_to_anchor=(0.7, 0.1, 1, 1),
                   bbox_transform=cax1.transAxes, 
                   loc = 'lower left')
cb = matplotlib.colorbar.ColorbarBase(cax, cmap='Reds', orientation = 'horizontal', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Density', fontsize = 16)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 4, pad = 2.5)
# suF1.text(-0.06, 1.15, "E", transform = suF1.transAxes, fontsize = 9)



####### cluster 2 burstiness #########################
cax2 = inset_axes(axD, "160%", "160%",
                   bbox_to_anchor=(0.0, -0.25, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')
tmp = rotated_images[-1]
cax2.imshow(tmp, extent = bound, alpha = 1, aspect = 'equal', cmap = 'viridis')
cax2.imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
noaxis(cax2)
cax2.patch.set_facecolor('none')
# title("Cluster 2")
#colorbar	
cax = inset_axes(cax2, "30%", "5%",
                   bbox_to_anchor=(0.76, 0.1, 1, 1),
                   bbox_transform=cax2.transAxes, 
                   loc = 'lower left')

cb = matplotlib.colorbar.ColorbarBase(cax, cmap='viridis', orientation = 'horizontal', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burstiness', fontsize = 16)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 4, pad = 2.5)
# cax2.text(-0.05, 1.01, "E", transform = suD0.transAxes, fontsize = 7)
axD.text(-0.0, 0.405, "e", transform = axD.transAxes, fontsize = 10, fontweight='bold')



subplots_adjust(bottom = 0.05, top = 0.95, right = 0.98, left = 0.2, hspace = 0.4)


#savefig("../../figures/figures_articles_v3/figart_1.pdf", dpi = 900, facecolor = 'white')
savefig(r"../../../Dropbox (Peyrache Lab)/Talks/fig_talk_9.png", dpi = 300, facecolor = 'white')

#os.system("evince ../../figures/figures_articles_v3/figart_1.pdf &")

