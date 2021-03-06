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
	fig_height = fig_width*golden_mean*1.5         # height in inches
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

outergs = gridspec.GridSpec(3,3, figure = fig, height_ratios = [1.0,1.5,0.8], hspace = 0.3)

#############################################
# A. HISTOLOGY
#############################################
gs = gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec = outergs[0,:], width_ratios=[0.6,0.7,0.1,1.2],wspace = 0.1)
axA = fig.add_subplot(gs[0,0])
# noaxis(axA)
histo = imread("../../data/histology/Mouse17/Mouse17_2_Slice7_Thalamus_Dapi_2.png")
imshow(histo, interpolation = 'bilinear',aspect= 'equal')
text(2500.0, 600.0, "Shanks", rotation = -10, color = 'white', fontsize = 9)
text(1600.0, 1900.0, "AD", color = 'red', fontsize = 8)
xticks([], [])
yticks([], [])
axA.text(-0.07, 1.04, "a", transform = axA.transAxes, fontsize = 10, fontweight='bold')

#############################################
# B. MAP AD + HD
#############################################
def show_labels(ax):
	ax.text(0.68,	1.09,	"AM", 	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(1.26,	1.26,	"VA",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.92,	2.05,	"AVd",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'), rotation = 50)
	ax.text(1.14,	1.72,	"AVv",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(1.28,	2.25,	"LD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.42,	2.17,	"sm",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.20,	1.89,	"MD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(-0.06,	1.58,	"PV",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.4,	1.5,	"IAD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'), rotation = 52)
	return


suB = fig.add_subplot(gs[0,1])
# imshow(carte38_mouse17, extent = bound_map_38, interpolation = 'bilinear', aspect = 'equal')
# imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
imshow(carte_adrien, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
i = 1
m = 'Mouse17'


tmp2 = headdir
tmp2[tmp2<0.05] = 0.0
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 1.5, color = 'black', marker = '.', 
	alpha = 1.0, linewidths = 0.5, label = 'shank positions')
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = tmp2*7., label = 'HD positions',
	color = 'red', marker = 'o', alpha = 0.6)


plot([2.2,2.2],[0,1], '-', linewidth = 1.3, color = 'black')
suB.text(2.25, 0.5, "1 mm", rotation = -90)

show_labels(suB)

leg = legend(loc = 'lower left', fontsize = 7, framealpha=1.0, bbox_to_anchor=(0.0, -0.09)) #, title = 'HD recording sites', )

noaxis(suB)
leg.get_title().set_fontsize(7)
leg.get_frame().set_facecolor('white')

annotate('Antero-dorsal (AD)', xy=(0.9,2.4), xytext=(0.9,2.7), xycoords='data', textcoords='data',
arrowprops=dict(facecolor='black',
	shrink=0.05,
	headwidth=3,
	headlength=2,
	width=0.3),
fontsize = 7, ha = 'center', va = 'bottom')

suB.text(-0.04, 1.03, "b", transform = suB.transAxes, fontsize = 10, fontweight = 'bold')

#############################################
# C. Exemples of autocorrelogram
#############################################
from matplotlib.patches import FancyArrowPatch, ArrowStyle, ConnectionPatch, Patch
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx

# suC = fig.add_subplot(3,2,3)
gsC = gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=gs[0,3], hspace = 0.2, wspace = 0.5)
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
			axC[(i,l)].text(-0.8, 1.12, "c", transform = axC[(i,l)].transAxes, fontsize = 9, fontweight='bold')





#############################################
# D. TSNE
#############################################
gs = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec = outergs[1,:], width_ratios=[0.66,0.34], hspace = 0.6, wspace = 0.25)

# axD = fig.add_subplot(outergs[2,1])
axD = Subplot(fig, gs[:,0])
fig.add_subplot(axD)
noaxis(axD)
sc = scatter(space[space['cluster'] == 0][1]*-1.0, space[space['cluster'] == 0][0]*-1, s = 28, c = burst['sws'][space['cluster'] == 0].values, edgecolor = 'none', alpha = 0.8, label = '_nolegend_')
# hd
scatter(space[space['cluster'] == 1][1]*-1.0, space[space['cluster'] == 1][0]*-1, s = 28, facecolor = 'red', edgecolor = 'none', alpha = 0.8, label = 'Cluster 1')
scatter(space[space['hd'] == 1][1]*-1.0, space[space['hd'] == 1][0]*-1, s = 5, marker = 'o', facecolor = 'white', edgecolor = 'black', linewidth = 0.5, label = 'HD neuron', alpha = 0.8)

xlim(-90, 100)
ylim(-300,80)

# legend
handles, labels = axD.get_legend_handles_labels()
axD.legend(handles[::-1], labels[::-1], 
			fancybox=False, 
			framealpha =0, 
			fontsize = 9, 
			loc = 'lower left', 
			bbox_to_anchor=(0.1, 0.47),
			handletextpad=0.05)
title("t-SNE of auto-correlograms", fontsize = 9)
axD.text(0.75, 0.54, "Cluster 2", transform = axD.transAxes, fontsize = 10)

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
cax = inset_axes(axD, "17%", "2%",
                   bbox_to_anchor=(0.75, 0.62, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burst index', labelpad = -24, fontsize = 8)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 9, pad = 2.5)


axD.text(-0.0, 1.01, "d", transform = axD.transAxes, fontsize = 10, fontweight='bold')

#############################################################
# E MAPS
#############################################################
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
neurons_ = mappings.index[np.where(mappings['hd'] == 1)[0]]
neurons_ = neurons_[np.where((firing_rate.loc[neurons_]>2.0).all(axis=1))[0]]
# EXample autocorr
cax0 = inset_axes(axD, "24%", "18%",bbox_to_anchor=(0.02, 0.14, 1, 1),
					bbox_transform=axD.transAxes, loc = 'lower left')
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


cax00 = inset_axes(axD, "24%", "18%",bbox_to_anchor=(0.02, -0.05, 1, 1),
					bbox_transform=axD.transAxes, loc = 'lower left')
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

# cax0.text('REM', 0.5, 0.5, fontsize = 7, transform = cax0.transAxes)
# cax00.text('NREM', 0.5, 0.5, fontsize = 7, transform = cax00.transAxes)
cax0.text(0.05, 0.1, "REM", transform = cax0.transAxes, fontsize = 6)
cax00.text(0.05, 0.1, "NREM", transform = cax00.transAxes, fontsize = 6)

# color bar firing rate
cax = inset_axes(axD, "1.5%", "9%",
                   bbox_to_anchor=(0.1, 0.35, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')

cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm = cNorm, orientation = 'vertical')
# cax.set_title("Rate (Hz)", fontsize = 7, pad = 3)
# cb.set_label("Rate (Hz)", orientation = 'horizontal')
cax0.text(0.48,1.3, 'Rate (Hz)', transform = cax0.transAxes, fontsize = 7)
cb.ax.xaxis.set_tick_params(pad = 1)
cb.ax.yaxis.set_ticks_position('left')
cb.ax.tick_params(labelsize=6) 
# cluster 1 HD
cax1 = inset_axes(axD, "50%", "47%",
                   bbox_to_anchor=(0.2, -0.05, 1, 1),
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
                   bbox_to_anchor=(0.7, -0.05, 1, 1),
                   bbox_transform=cax1.transAxes, 
                   loc = 'lower left')
cb = matplotlib.colorbar.ColorbarBase(cax, cmap='Reds', orientation = 'horizontal', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Density', labelpad = -24)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 4, pad = 2.5)
# suF1.text(-0.06, 1.15, "E", transform = suF1.transAxes, fontsize = 9)

# cluster 2 burstiness
cax2 = inset_axes(axD, "50%", "47%",
                   bbox_to_anchor=(0.5, -0.05, 1, 1),
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
                   bbox_to_anchor=(0.76, -0.05, 1, 1),
                   bbox_transform=cax2.transAxes, 
                   loc = 'lower left')

cb = matplotlib.colorbar.ColorbarBase(cax, cmap='viridis', orientation = 'horizontal', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burstiness', labelpad = -24)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 4, pad = 2.5)
# cax2.text(-0.05, 1.01, "E", transform = suD0.transAxes, fontsize = 7)
axD.text(-0.0, 0.405, "e", transform = axD.transAxes, fontsize = 10, fontweight='bold')




#############################################
# F. GRadient
#############################################
mean_burst = pd.DataFrame(columns = ['12', '17','20', '32'])
count_nucl = pd.DataFrame(columns = ['12', '17','20', '32'])

for m in ['12', '17','20', '32']:
	subspace = pd.read_hdf("../../figures/figures_articles_v2/figure1/subspace_Mouse"+m+".hdf5")	
	nucleus = np.unique(subspace['nucleus'])
	mean_burstiness = [burst.loc[subspace.index, 'sws'][subspace['nucleus'] == nu].mean() for nu in nucleus]
	mean_burst[m] = pd.Series(index = nucleus, data = mean_burstiness)	
	total = [np.sum(subspace['nucleus'] == n) for n in nucleus]
	count_nucl[m] = pd.Series(index = nucleus, data = total)
# nucleus = ['AD', 'LDvl', 'AVd', 'MD', 'AVv', 'IAD', 'CM', 'AM', 'VA', 'Re']
nucleus = list(count_nucl.dropna().index.values)

# mean all
meanall = pd.DataFrame(index = nucleus, columns = ['mean','sem'])
for n in nucleus:
	tmp = burst[space['nucleus'] == n]
	# if len(tmp)>20:
	meanall.loc[n,'mean'] = tmp.mean(0)['sws']
	meanall.loc[n,'sem'] = tmp.sem(0)['sws']

meanall = meanall.sort_values('mean')

# axF = fig.add_subplot(3,4,11)
# axG = Subplot(fig, gs[1:,3])
axG = Subplot(fig, gs[0,1])
fig.add_subplot(axG)
simpleaxis(axG)
# mean_burst = mean_burst.loc[nucleus]
# mean_burst[0] = np.arange(len(nucleus))
# for i, m in enumerate(['17', '12','20', '32']):	
# 	tmp = mean_burst[[m,0]].dropna()
# 	plot(tmp[m], tmp[0], 'o', label = str(i+1), markersize = 2, linewidth = 1)
# plot(mean_burst.mean(1).values, mean_burst[0].values, 'o-', label = 'Mean', markersize = 2, linewidth = 1, color = 'black')
x, s = (meanall['mean'].values.astype('float32'), meanall['sem'].values.astype('float32'))
# plot(np.arange(len(nucleus)), x, 'o-', markersize = 2, linewidth = 1.5, color = 'black')
# fill_between(np.arange(len(nucleus)), x-s, x+s, color = 'grey', alpha = 0.5)

bar(np.arange(len(nucleus)), x, yerr = s, 
	linewidth = 1, color = 'none', edgecolor = 'black')

leg = legend(frameon=False, bbox_to_anchor = (0.9, 1.15))
# leg.set_title("Mouse", prop={'size':5})
xticks(np.arange(len(nucleus)), nucleus)
ylabel("NREM \n Burst index",  multialignment='center')
xlabel("Nuclei", labelpad = 0.1)
# annotate(s='', xy = (4,25), xytext=(6,25), arrowprops=dict(arrowstyle='<->'))
# text(2.5, 24.5, 'Dorsal')
# text(6, 24.5, 'Ventral')
# axG.invert_yaxis()
axG.text(-0.35, 1.05, "f", transform = axG.transAxes, fontsize = 10, fontweight='bold')


###########################################################################
# G SCORES XGB HD/NOHD
###########################################################################
# count = pd.read_hdf("../../figures/figures_articles_v2/figure1/count_time.h5")
# score = pd.read_hdf("../../figures/figures_articles_v2/figure1/score_logreg.h5")
store = pd.HDFStore("../../figures/figures_articles_v2/figure1/score_XGB_HDNOHD.h5", 'r')
score = store['score']
shuff = store['shuff']
store.close()
a = (score.mean(1)-shuff.mean(1))/(1.0 - shuff.mean(1))


# axE = Subplot(fig, gs[1,5])
axE = Subplot(fig, gs[1,1])
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
axE.text(-0.35, 1.05, 'g', transform = axE.transAxes, fontsize = 10, fontweight='bold')
axE.annotate('6 ms', xy=(6, a.loc[6]), 
				xytext=(2, a.loc[6]+0.3), 
				fontsize = 8,
				arrowprops=dict(facecolor='black',								
								arrowstyle="->",
								connectionstyle="arc3")
				)

# axE.semilogx(count.mean(1).loc[0:3000], 'o-', markersize = 2, color = 'black', linewidth = 1.5)
# axE2 = axE.twinx()
# axE2.spines['top'].set_visible(False)
# axE2.set_ylabel("Logit score for HD classification", rotation = -90, labelpad = 10)
# axE2.yaxis.label.set_color('firebrick')
# axE2.tick_params(axis='y', labelcolor='firebrick')

# ylabel("Clustering specificity of HD cells")

###########################################################################
# H SCORE NUCLEUS
###########################################################################
# axH = Subplot(fig, gs[2,5])
axH = Subplot(fig, gs[2,1])
fig.add_subplot(axH)
simpleaxis(axH)
axH.text(-0.35, 1.05, 'h', transform = axH.transAxes, fontsize = 10, fontweight='bold')
xlabel("Nuclei", labelpad = 0.1)
ylabel("Classification\n score")
# title("Classification", pad = 1.0)
tmp = mean_score[('score', 'auto', 'mean')]
tmp2 = mean_score[('shuffle', 'auto', 'mean')]
tmp3 = (tmp-tmp2)/(1-tmp2)
tmp3 = tmp3.sort_values(ascending=False)
order = tmp3.index.values
# tmp2 = mean_score[('shuffle', 'swr', 'mean')].sort_values()
# bar(np.arange(len(tmp)), tmp2.values, linewidth = 1, color = 'none', edgecolor = 'black', linestyle = '--')
bar(np.arange(len(tmp3)), tmp3.values, yerr = mean_score.loc[order,('score','swr','sem')], 
	linewidth = 1, color = 'none', edgecolor = 'black')
xticks(np.arange(mean_score.shape[0]), order)#, rotation = 45)
# axhline(1/8, linestyle = '--', color = 'black', linewidth = 0.5)
# yticks([0, 0.2,0.4], [0, 20,40])


# ##########################################################################
# # ANOVAS
# #########################################################################
# neurons = burst['sws'].index
# # neurons = space[space.loc[neurons,'nucleus'].isin(nucleus)].index
# mouses = pd.DataFrame(index = neurons, columns = ['animal'], data = [n.split("-")[0] for n in neurons])
# totest = pd.concat([burst.loc[neurons,'sws'],space.loc[neurons,'nucleus'],mouses], axis = 1)
# groups1 = totest.groupby("nucleus").groups
# groups2 = totest.groupby("animal").groups

# mean1 = pd.DataFrame(index = groups1.keys(),
# 					data = np.array([[np.mean(totest.loc[groups1[k],'sws']) for k in groups1.keys()],
# 							[np.var(totest.loc[groups1[k],'sws']) for k in groups1.keys()]]).T,
# 					columns = ['mean', 'var']) 
# mean2 = pd.DataFrame(index = groups2.keys(),
# 					data = np.array([[np.mean(totest.loc[groups2[k],'sws']) for k in groups2.keys()],
# 							[np.var(totest.loc[groups2[k],'sws']) for k in groups2.keys()]]).T,
# 					columns = ['mean', 'var']) 


# import scipy.stats as stats
# f1 = stats.f_oneway(*[totest.loc[groups1[n],'sws'] for n in nucleus])
# f2 = stats.f_oneway(*[totest.loc[groups2[n],'sws'] for n in groups2.keys()])

# k1 = stats.kruskal(*[totest.loc[groups1[n],'sws'] for n in nucleus])
# k2 = stats.kruskal(*[totest.loc[groups2[n],'sws'] for n in groups2.keys()])

# SPEFICIF TO SLOW DYNAMIC
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
# nucleus = np.unique(mappings['nucleus'])
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


gs = gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec = outergs[2,:], width_ratios=[0.005,1,1,1], wspace = 0.4)#, hspace = 0.5, wspace = 0.2)
###################################################################################################
# I. AUTOCORRELOGRAM EXEMPLE
###################################################################################################
autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG.h5")

axI = Subplot(fig, gs[0,1])
fig.add_subplot(axI)
simpleaxis(axI)

def func(x, a, b, c):
	return a*np.exp(-(1./b)*x) + c


labels = ['AD (HD)', 'AVd', 'IAD']
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
xlabel("Time lag (s)", labelpad = 0.5)
ylabel("Autocorrelation (a.u)")
locator_params(nbins = 4)

axI.text(-0.30, 1.10, "i", transform = axI.transAxes, fontsize = 9, fontweight='bold')

###################################################################################################
# J. LAMBDA AUTOCORRELOGRAM / NUCLEUS
###################################################################################################
axJ = Subplot(fig, gs[0,2])
fig.add_subplot(axJ)
simpleaxis(axJ)

order = lambdaa_nucleus[('wak', 'mean')].sort_values().index

labels = ['Wake', 'REM']

mks = ['o-', 'D-']

colors = ['white', 'grey']

offset = 0

for i, ep in enumerate(['wak', 'rem']):
	m = lambdaa_nucleus.loc[order,(ep,'mean')].values.astype('float32')
	s = lambdaa_nucleus.loc[order,(ep,'sem')].values.astype('float32')
	# plot(m, np.arange(len(order)), mks[i], color = colors[i], label = labels[i], markersize = 3, linewidth = 1)
	# fill_betweenx(np.arange(len(order)), m+s, m-s, color = colors[i], alpha = 0.3)
	m = m[::-1]
	s = s[::-1]
	bar(np.arange(len(order))+offset, m, width = 0.4, yerr = s, linewidth = 1, edgecolor = 'black', color = colors[i], label = labels[i])
	offset+=0.4

legend(edgecolor = None, facecolor = None, frameon = False)
xticks(np.arange(len(order)), order[::-1])
xlabel("Nuclei", labelpad = 0.5)	
ylabel(r"Decay time $\tau$ (s)", labelpad = 0.5)
locator_params(axis = 'y', nbins = 4)

axJ.text(-0.3, 1.10, "j", transform = axJ.transAxes, fontsize = 9, fontweight='bold')

###################################################################################################
# K. BURSTINESS VS LAMBDA
###################################################################################################
axK = Subplot(fig, gs[0,3])
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


scatter(df2['burst'].values, df2['lambda'].values, 4, color = 'black', alpha = 1.0, edgecolors = 'none')
xlabel("NREM burst index", labelpad = 0.5)
ylabel(r"REM decay time $\tau$ (s)")

yticks([0,1,2,3],['0','1','2','3'])


axK.text(0.25, 1.0, "r="+str(np.round(c, 3))+" (p<0.001)", transform = axK.transAxes, fontsize = 8)
axK.text(-0.3, 1.10, "k", transform = axK.transAxes, fontsize = 9 ,fontweight='bold')



subplots_adjust(bottom = 0.05, top = 0.97, right = 0.98, left = 0.01, hspace = 0.1)

savefig("../../figures/figures_articles_v3/figart_1.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v3/figart_1.pdf &")

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

