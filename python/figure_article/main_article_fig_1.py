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


space = pd.read_hdf("../../figures/figures_articles/figure1/space.hdf5")

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
burst = burst.loc[space.index]


# autocorr = pd.read_hdf("../../figures/figures_articles/figure1/autocorr.hdf5")
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")

# carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
# carte38_mouse17_2 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
# bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
# cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

carte_adrien = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_ALL-01.png')
carte_adrien2 = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_Contour-01.png')
bound_adrien = (-398/1254, 3319/1254, -(239/1254 - 20/1044), 3278/1254)

# carte_adrien2[:,:,-1][carte_adrien2[:,:,-1]<150] = 0.0

tmp = cPickle.load(open("../../figures/figures_articles/figure1/shifts.pickle", 'rb'))
angles = tmp['angles']
shifts = tmp['shifts']


hd_index = space.index.values[space['hd'] == 1]

neurontoplot = [np.intersect1d(hd_index, space.index.values[space['cluster'] == 1])[0],
				burst.loc[space.index.values[space['cluster'] == 0]].sort_values('sws').index[3],
				burst.sort_values('sws').index.values[-20]]

# specific to mouse 17
subspace = pd.read_hdf("../../figures/figures_articles/figure1/subspace_Mouse17.hdf5")

data = cPickle.load(open("../../figures/figures_articles/figure1/rotated_images_Mouse17.pickle", 'rb'))
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

# sys.exit()

###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.45         # height in inches
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

outergs = gridspec.GridSpec(2,3, figure = fig, height_ratios = [1.0,1.5], hspace = 0.25)

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
axA.text(-0.0, 1.10, "A", transform = axA.transAxes, fontsize = 10)

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
fontsize = 7, ha = 'left', va = 'bottom')

suB.text(-0.0, 1.18, "B", transform = suB.transAxes, fontsize = 10)

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
			axC[(i,l)].text(-0.15, 1.10, "C", transform = axC[(i,l)].transAxes, fontsize = 9)





#############################################
# D. TSNE
#############################################
gs = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec = outergs[1,:], width_ratios=[0.66,0.34], hspace = 0.5, wspace = 0.2)

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
axD.legend(handles[::-1], labels[::-1], fancybox=False, framealpha =0, fontsize = 9, loc = 'lower left', bbox_to_anchor=(-0.01, 0.51))
title("TSNE Anterior thalamus", fontsize = 9)
axD.text(0.75, 0.54, "Cluster 2", transform = axD.transAxes, fontsize = 10)

# arrows to map
import matplotlib.patches as mpatches
el = mpatches.Ellipse((-40, -130), 0.3, 0.4, angle=-30, alpha=0.2)
axD.add_artist(el)
annotate("", xy=(-50, -130), xytext=(-60, -100), color = 'grey',
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
                   bbox_to_anchor=(0.75, 0.6, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burst index', labelpad = -24, fontsize = 8)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 9, pad = 2.5)


axD.text(-0.0, 1.01, "D", transform = axD.transAxes, fontsize = 10)

#############################################################
# E MAPS
#############################################################

# cluster 1 HD
cax1 = inset_axes(axD, "50%", "47%",
                   bbox_to_anchor=(0.0, -0.05, 1, 1),
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
                   bbox_to_anchor=(0.7, -0.05, 1, 1),
                   bbox_transform=cax2.transAxes, 
                   loc = 'lower left')

cb = matplotlib.colorbar.ColorbarBase(cax, cmap='viridis', orientation = 'horizontal', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burstiness', labelpad = -24)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 4, pad = 2.5)
# cax2.text(-0.05, 1.01, "E", transform = suD0.transAxes, fontsize = 7)
axD.text(-0.0, 0.405, "E", transform = axD.transAxes, fontsize = 10)




#############################################
# F. GRadient
#############################################
mean_burst = pd.DataFrame(columns = ['12', '17','20', '32'])
count_nucl = pd.DataFrame(columns = ['12', '17','20', '32'])

for m in ['12', '17','20', '32']:
	subspace = pd.read_hdf("../../figures/figures_articles/figure1/subspace_Mouse"+m+".hdf5")	
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
plot(np.arange(len(nucleus)), x, 'o-', markersize = 2, linewidth = 1.5, color = 'black')
fill_between(np.arange(len(nucleus)), x-s, x+s, color = 'grey', alpha = 0.5)
leg = legend(frameon=False, bbox_to_anchor = (0.9, 1.15))
# leg.set_title("Mouse", prop={'size':5})
xticks(np.arange(len(nucleus)), nucleus)
ylabel("NREM Burst index")
xlabel("Nuclei", labelpad = 0.1)
# annotate(s='', xy = (4,25), xytext=(6,25), arrowprops=dict(arrowstyle='<->'))
# text(2.5, 24.5, 'Dorsal')
# text(6, 24.5, 'Ventral')
# axG.invert_yaxis()
axG.text(-0.10, 1.05, "F", transform = axG.transAxes, fontsize = 10)


###########################################################################
# G SCORES XGB HD/NOHD
###########################################################################
# count = pd.read_hdf("../../figures/figures_articles/figure1/count_time.h5")
# score = pd.read_hdf("../../figures/figures_articles/figure1/score_logreg.h5")
store = pd.HDFStore("../../figures/figures_articles/figure1/score_XGB_HDNOHD.h5", 'r')
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
ylabel("Score")
axE.text(0.5, 0.3, 'HD/no-HD', transform = axE.transAxes, fontsize = 9)
axE.text(-0.10, 1.05, 'G', transform = axE.transAxes, fontsize = 10)
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
axH.text(-0.10, 1.05, 'H', transform = axH.transAxes, fontsize = 10)
xlabel("Nuclei", labelpad = 0.1)
ylabel("Score")
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

subplots_adjust(bottom = 0.05, top = 0.97, right = 0.98, left = 0.01, hspace = 0.1)

savefig("../../figures/figures_articles/figart_1.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_1.pdf &")

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

