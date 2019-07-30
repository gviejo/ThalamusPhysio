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
	fig_height = fig_width*golden_mean*1         # height in inches
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

outergs = gridspec.GridSpec(2,2, figure = fig, height_ratios = [1.0,0.6], hspace = 0.5)

#############################################
# A. Exemples of autocorrelogram
#############################################
from matplotlib.patches import FancyArrowPatch, ArrowStyle, ConnectionPatch, Patch
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx

# suC = fig.add_subplot(3,2,3)
gsC = gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outergs[0,0], hspace = 0.2, wspace = 0.5)
axC = {}

labels = ['a\nHD', 'b\nNon-bursty', 'c\nBursty']
titles = ['Wake', 'REM', 'NREM']

viridis = get_cmap('viridis')
cNorm = colors.Normalize(vmin=burst['sws'].min(), vmax = burst['sws'].max())
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap = viridis)


colors = ['red', scalarMap.to_rgba(burst.loc[neurontoplot[1], 'sws']), scalarMap.to_rgba(burst.loc[neurontoplot[2], 'sws'])] 

for i in range(3):
	for l,j in enumerate(['wake', 'rem', 'sws']):
		tmp = store_autocorr[j][neurontoplot[i]]
		tmp[0] = 0.0
		tmp = tmp.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
		tmp[0] = 0.0
		tmp = tmp[-80:80]		
		axC[(i,l)] = subplot(gsC[i,l])		
		simpleaxis(axC[(i,l)])
		plot(tmp, color = colors[i], label = labels[i], linewidth = 1.5)
		if i in [0,1]:
			axC[(i,l)].set_xticks([])
			axC[(i,l)].set_xticklabels([])
		if i == 0:
			axC[(i,l)].set_title(titles[l], fontsize = 7)
		if l == 0:
			# leg = legend(fancybox = False, framealpha = 0, loc='lower left', bbox_to_anchor=(-1, 0))
			# axB[(i,l)].set_ylabel(labels[i], labelpad = 2.)
			axC[(i,l)].text(-0.51, 0.5, labels[i], transform = axC[(i,l)].transAxes, ha = 'center', va = 'center', fontsize = 7, rotation = 'vertical')
		if i == 2:
			axC[(i,l)].set_xlabel("Time (ms)", labelpad = 0.1)
		if i == 0 and l == 0:
			axC[(i,l)].text(-0.5, 1.2, "a", transform = axC[(i,l)].transAxes, fontsize = 10, fontweight='bold')

		



#############################################
# B. TSNE
#############################################
axD = Subplot(fig, outergs[0,1])
fig.add_subplot(axD)
noaxis(axD)
sc = scatter(space[space['cluster'] == 0][1]*-1.0, space[space['cluster'] == 0][0]*-1, s = 28, c = burst['sws'][space['cluster'] == 0].values, edgecolor = 'none', alpha = 0.8, label = '_nolegend_')
# hd
scatter(space[space['cluster'] == 1][1]*-1.0, space[space['cluster'] == 1][0]*-1, s = 28, facecolor = 'red', edgecolor = 'none', alpha = 0.8, label = 'Cluster 1')
scatter(space[space['hd'] == 1][1]*-1.0, space[space['hd'] == 1][0]*-1, s = 5, marker = 'o', facecolor = 'white', edgecolor = 'black', linewidth = 0.5, label = 'HD neuron', alpha = 0.8)

# xlim(-90, 100)
ylim(-120,80)

# legend
handles, labels = axD.get_legend_handles_labels()
axD.legend(handles[::-1], labels[::-1], 
			fancybox=False, 
			framealpha =0, 
			fontsize = 9, 
			loc = 'lower left', 
			bbox_to_anchor=(-0.08, 0.0),
			handletextpad=0.05)
title("t-SNE of auto-correlograms", fontsize = 9)
axD.text(0.5, -0.13, "Cluster 2", transform = axD.transAxes, fontsize = 9)

# arrows to map
import matplotlib.patches as mpatches
el = mpatches.Ellipse((-40, -130), 0.3, 0.4, angle=-30, alpha=0.2)
axD.add_artist(el)
annotate("", xy=(-15, -140), xytext=(-25, -110), color = 'grey',
            arrowprops=dict(arrowstyle="fancy", #linestyle="dashed",
                            color="0.5",
                            patchB=el,
                            shrinkB=5,
                            connectionstyle="arc3,rad=0.3",
                            ))

el2 = mpatches.Ellipse((50, -130), 0.3, 0.4, angle=-30, alpha=0.2)
axD.add_artist(el)
annotate("", xy=(50, -130), xytext=(60, -100), color = 'grey',
            arrowprops=dict(arrowstyle="fancy", #linestyle="dashed",
                            color="0.5",
                            patchB=el,
                            shrinkB=5,
                            connectionstyle="arc3,rad=-0.3",
                            ))


# surrounding examples
scatter(space.loc[neurontoplot,1]*-1.0, space.loc[neurontoplot,0]*-1.0, s = 25, facecolor = 'none', edgecolor = 'grey', linewidths = 2)
txts = ['a', 'b', 'c']
xyoffset = [[-10, 12], [14, 12], [8, 20]]
for i, t in zip(range(3), txts):
	x, y = (space.loc[neurontoplot[i],1]*-1, space.loc[neurontoplot[i],0]*-1)
	annotate(t, xy=(x, y), 
				xytext=(x+np.sign(x)*xyoffset[i][0],  y+np.sign(y)*xyoffset[i][1]), 
				fontsize = 8,
				arrowprops=dict(facecolor='black',								
								arrowstyle="->",
								connectionstyle="arc3")
				)


#colorbar	
cax = inset_axes(axD, "17%", "4%",
                   bbox_to_anchor=(0.5, 0.0, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burst index', labelpad = -22, fontsize = 8)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 9, pad = 2.5)


axD.text(-0.0, 1.05, "b", transform = axD.transAxes, fontsize = 10, fontweight='bold')

#############################################################
# c MAPS
#############################################################
gs = gridspec.GridSpecFromSubplotSpec(2,6,subplot_spec = outergs[1,:], width_ratios = [0.2, 0.0, 0.3, 0.3, 0.15, 0.3])

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
neurons_ = mappings.index[np.where(mappings['hd'] == 1)[0]]
neurons_ = neurons_[np.where((firing_rate.loc[neurons_]>2.0).all(axis=1))[0]]

##################################
cax0 = Subplot(fig, gs[0,0])
fig.add_subplot(cax0)
noaxis(cax0)
# cax0.spines['bottom'].set_visible(True)
autocorr = store_autocorr['rem'][neurons_]
autocorr.loc[0.0] = 0.0
fr = firing_rate.loc[neurons_, 'rem'].sort_values()
idx = np.arange(0, len(fr), 6)[0:-1]
cm = get_cmap('Reds')
cNorm = matplotlib.colors.Normalize(vmin = 0.0, vmax = fr.iloc[idx].max())
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap = cm)
for n in fr.index[idx]:
	tmp = autocorr.loc[-100:100,n]
	tmp /= np.mean(tmp.loc[-100:-50])
	cax0.plot(tmp.loc[-50:50], color = scalarMap.to_rgba(fr.loc[n]))
cax0.text(-0.4, 1.10, "c", transform = cax0.transAxes, fontsize = 10, fontweight='bold')

##################################
cax00 = Subplot(fig, gs[1,0])
fig.add_subplot(cax00)
noaxis(cax00)
cax00.spines['bottom'].set_visible(True)
xticks([-50, 0, 50], fontsize = 6)
cax00.tick_params(axis='x', which='major', pad=1)
xlabel("Time (ms)", labelpad=0.1, fontsize = 6)
autocorr = store_autocorr['sws'][neurons_]
autocorr.loc[0.0] = 0.0
# fr = firing_rate.loc[neurons_, 'sws'].sort_values()
# idx = np.arange(0, len(fr), 6)[0:-1]
cm = get_cmap('Reds')
cNorm = matplotlib.colors.Normalize(vmin = 0.0, vmax = fr.iloc[idx].max())
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap = cm)
for n in fr.index[idx]:
	tmp = autocorr.loc[-100:100,n]
	tmp /= np.mean(tmp.loc[-100:-50])
	cax00.plot(tmp.loc[-50:50], color = scalarMap.to_rgba(fr.loc[n]))


cax0.text(0.75, 0.1, "REM", transform = cax0.transAxes, fontsize = 8)
cax00.text(0.75, 0.1, "NREM", transform = cax00.transAxes, fontsize = 8)

# color bar firing rate
cax = inset_axes(axD, "10%", "50%",
                   bbox_to_anchor=(-0.32, -0.5, 1, 1),
                   bbox_transform=cax0.transAxes, 
                   loc = 'lower left')

cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm = cNorm, orientation = 'vertical')
# cax0.text(0.48,1.3, 'Rate (Hz)', transform = cax0.transAxes, fontsize = 7)
cb.ax.set_title("Rate (Hz)", fontsize = 7)
cb.ax.xaxis.set_tick_params(pad = 1)
cb.ax.yaxis.set_ticks_position('left')
cb.ax.tick_params(labelsize=6) 


########### cluster 1 HD ##########
cax1 = Subplot(fig, gs[:,2])
fig.add_subplot(cax1)
noaxis(cax1)
tmp = rotated_images[1]
tmp[tmp<0.0] = 0.0
cax1.imshow(tmp, extent = bound, alpha = 1, aspect = 'equal', cmap = 'Reds')
cax1.imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
noaxis(cax1)
cax1.patch.set_facecolor('none')
title('Cluster 1')
#colorbar	
cax = inset_axes(cax1, "30%", "5%",
                   bbox_to_anchor=(0.7, -0.1, 1, 1),
                   bbox_transform=cax1.transAxes, 
                   loc = 'lower left')
cb = matplotlib.colorbar.ColorbarBase(cax, cmap='Reds', orientation = 'horizontal', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Density', labelpad = -24)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 4, pad = 2.5)
# suF1.text(-0.06, 1.15, "E", transform = suF1.transAxes, fontsize = 9)
cax1.text(-0.0, 1.05, "d", transform = cax1.transAxes, fontsize = 10, fontweight='bold')

########### cluster 2 burstiness ######
cax2 = Subplot(fig, gs[:,3])
fig.add_subplot(cax2)
noaxis(cax2)
tmp = rotated_images[-1]
cax2.imshow(tmp, extent = bound, alpha = 1, aspect = 'equal', cmap = 'viridis')
cax2.imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
noaxis(cax2)
cax2.patch.set_facecolor('none')
title('Cluster 2')
# title("Cluster 2")
#colorbar	
cax = inset_axes(cax2, "30%", "5%",
                   bbox_to_anchor=(0.76, -0.1, 1, 1),
                   bbox_transform=cax2.transAxes, 
                   loc = 'lower left')

cb = matplotlib.colorbar.ColorbarBase(cax, cmap='viridis', orientation = 'horizontal', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burstiness', labelpad = -24)
cb.ax.xaxis.set_tick_params(pad = 1)

###########################################################################
# D SCORES XGB HD/NOHD
###########################################################################
store = pd.HDFStore("../../figures/figures_articles_v2/figure1/score_XGB_HDNOHD.h5", 'r')
score = store['score']
shuff = store['shuff']
store.close()
a = (score.mean(1)-shuff.mean(1))/(1.0 - shuff.mean(1))


axE = Subplot(fig, gs[:,-1])
fig.add_subplot(axE)
simpleaxis(axE)
# semilogx(score.loc[0:3000], '+-', markersize = 2, color = 'firebrick', linewidth = 1.5)
semilogx(a.loc[0:1000], 'o-', markersize = 3, color = 'firebrick', linewidth = 1.5)
axhline(0.5, linewidth = 1.0, color = 'black', alpha = 0.8, linestyle = '--')
# ylim(-0.1,1)
# title("HD classification")
xlabel("Time (ms)", labelpad = 0.1)
ylabel("Classification \n score", multialignment='center')
axE.text(0.5, 0.3, 'HD/no-HD', transform = axE.transAxes, fontsize = 9)
axE.text(-0.4, 1.05, 'e', transform = axE.transAxes, fontsize = 10, fontweight='bold')
axE.annotate('6 ms', xy=(6, a.loc[6]), 
				xytext=(2, a.loc[6]+0.3), 
				fontsize = 8,
				arrowprops=dict(facecolor='black',								
								arrowstyle="->",
								connectionstyle="arc3")
				)
# axE.text(-0.0, 1.05, "d", transform = axE.transAxes, fontsize = 10, fontweight='bold')


subplots_adjust(bottom = 0.1, top = 0.95, right = 0.98, left = 0.07, hspace = 0.1)

savefig("../../figures/figures_articles_v4/figart_4.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v4/figart_4.pdf &")
