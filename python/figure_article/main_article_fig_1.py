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

carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
carte38_mouse17_2 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)


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


###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.4          # height in inches
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
	"axes.labelsize": 6,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 6,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 6,
	"ytick.labelsize": 6,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.5,
	"ytick.major.size"      : 1.0,
	"xtick.major.size"      : 1.0
	}    
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

colors = ['red', 'green', 'blue', 'purple', 'orange']
cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']
markers = ['o', '^', '*', 's']

fig = figure(figsize = figsize(1.0), tight_layout=True)

outergs = gridspec.GridSpec(3,2, figure = fig)

#############################################
# A. HISTOLOGY
#############################################
axA = fig.add_subplot(outergs[0,0])
noaxis(axA)
histo = imread("../../data/histology/Mouse17/Mouse17_2_Slice7_Thalamus_Dapi.png")
text(5600.0, 500.0, "Shanks", rotation = -7, color = 'white', fontsize = 8)
imshow(histo, interpolation = 'bilinear', aspect= 'equal')
xticks([], [])
yticks([], [])
axA.text(-0.2, 0.95, "A", transform = axA.transAxes, fontsize = 9)

#############################################
# B. MAP AD + HD
#############################################
suB = fig.add_subplot(outergs[0,1])
imshow(carte38_mouse17, extent = bound_map_38, interpolation = 'bilinear', aspect = 'equal')

# for i,m in enumerate(['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']):
i = 1
m = 'Mouse17'


tmp2 = headdir
tmp2[tmp2<0.05] = 0.0
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = tmp2*5., label = 'HD positions',
	color = 'red', marker = 'o', alpha = 1.0)
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 1, color = 'black', marker = '.', 
	alpha = 1.0, linewidths = 0.5, label = 'shank positions')



leg = legend(loc = 'lower left', fontsize = 6, title = 'HD recording sites', framealpha=1.0)

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

suB.text(-0.12, 0.96, "B", transform = suB.transAxes, fontsize = 9)

#############################################
# C. Exemples of autocorrelogram
#############################################
from matplotlib.patches import FancyArrowPatch, ArrowStyle, ConnectionPatch, Patch
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx

# suC = fig.add_subplot(3,2,3)
gsC = gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=outergs[1,0], hspace = 0.2, wspace = 0.6)
axC = {}

labels = ['1\nHD', '2\nNon-bursty', '3\nBursty']
titles = ['Wake', 'REM', 'SWS']

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
		plot(tmp, color = colors[i], label = labels[i], linewidth = 1.0)
		if i in [0,1]:
			axC[(i,l)].set_xticks([])
			axC[(i,l)].set_xticklabels([])
		if i == 0:
			axC[(i,l)].set_title(titles[l], fontsize = 7)
		if l == 0:
			# leg = legend(fancybox = False, framealpha = 0, loc='lower left', bbox_to_anchor=(-1, 0))
			# axB[(i,l)].set_ylabel(labels[i], labelpad = 2.)
			axC[(i,l)].text(-0.5, 0.5, labels[i], transform = axC[(i,l)].transAxes, ha = 'center', va = 'center', fontsize = 7, rotation = 'vertical')
		if i == 2:
			axC[(i,l)].set_xlabel("Time (ms)", labelpad = 0.1)
		if i == 0 and l == 0:
			axC[(i,l)].text(-0.15, 1.15, "C", transform = axC[(i,l)].transAxes, fontsize = 9)





#############################################
# D. TSNE
#############################################
axD = fig.add_subplot(outergs[1,1])
noaxis(axD)
sc = scatter(space[space['cluster'] == 0][1]*-1.0, space[space['cluster'] == 0][0]*-1.0, s = 6, c = burst['sws'][space['cluster'] == 0].values, edgecolor = 'none', alpha = 1.0, label = '_nolegend_')
# hd
scatter(space[space['cluster'] == 1][1]*-1.0, space[space['cluster'] == 1][0]*-1.0, s = 6, facecolor = 'red', edgecolor = 'none', alpha = 1.0, label = 'Cluster 2')
scatter(space[space['hd'] == 1][1]*-1.0, space[space['hd'] == 1][0]*-1.0, s = 2, marker = 'o', facecolor = 'black', edgecolor = 'none', linewidth = 0.0, label = 'HD')
axD.legend(fancybox=False, framealpha =0, fontsize = 6, loc = 'lower left', bbox_to_anchor=(-0.05, -0.14))
title("TSNE Anterior thalamus", fontsize = 7)

# surrounding examples
scatter(space.loc[neurontoplot,1]*-1.0, space.loc[neurontoplot,0]*-1.0, s = 12, facecolor = 'none', edgecolor = 'black')
txts = ['1', '2', '3']
xyoffset = [[1, 5], [3, 2], [1, 6]]
for i, t in zip(range(3), txts):
	x, y = (space.loc[neurontoplot[i],1]*-1.0, space.loc[neurontoplot[i],0]*-1.0)
	annotate(t, xy=(x, y), xytext=(x+np.sign(x)*xyoffset[i][0],  y+np.sign(y)*xyoffset[i][1]))


#colorbar	
cax = inset_axes(axD, "30%", "4%",
                   bbox_to_anchor=(-0.04, 0.14, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burstiness', labelpad = -4)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("Cluster 1", fontsize = 6, pad = 2.5)

axD.text(-0.1, 1.01, "D", transform = axD.transAxes, fontsize = 9)




#############################################
# E. Burstiness 
#############################################
# suE = fig.add_subplot(3,1,3)
gs = gridspec.GridSpecFromSubplotSpec(2,5,subplot_spec = outergs[2,:], width_ratios=[1,1,0.1,1,1], height_ratios=[0.1,1], wspace = 0.5)
cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

# cluster 1 HD
suE1 = Subplot(fig, gs[1,0])
fig.add_subplot(suE1)
noaxis(suE1)
tmp = rotated_images[1]
tmp[tmp<0.0] = 0.0
imshow(tmp, extent = bound, alpha = 0.9, aspect = 'equal', cmap = 'Reds')
imshow(carte38_mouse17_2[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
ylabel('Cluster 2')
#colorbar	
cax = inset_axes(suE1, "5%", "30%",
                   bbox_to_anchor=(0.8, -0.2, 1, 1),
                   bbox_transform=suE1.transAxes, 
                   loc = 'lower left')
cb = matplotlib.colorbar.ColorbarBase(cax, cmap='Reds', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Density', labelpad = -20)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 4, pad = 2.5)
suE1.text(-0.06, 1.05, "E", transform = suE1.transAxes, fontsize = 9)

# cluster 2 burstiness
suE0 = Subplot(fig, gs[1,1])
fig.add_subplot(suE0)
noaxis(suE0)
tmp = rotated_images[-1]
imshow(tmp, extent = bound, alpha = 0.9, aspect = 'equal', cmap = 'viridis')
imshow(carte38_mouse17_2[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
ylabel("Cluster 1")
#colorbar	
cax = inset_axes(suE0, "5%", "30%",
                   bbox_to_anchor=(0.8, -0.2, 1, 1),
                   bbox_transform=suE0.transAxes, 
                   loc = 'lower left')

cb = matplotlib.colorbar.ColorbarBase(cax, cmap='viridis', ticks = [0, 1])
# cb = colorbar(sc, cax = cax, orientation = 'horizontal', ticks = [1, int(np.floor(burst['sws'].max()))])
cb.set_label('Burstiness', labelpad = -20)
cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("Cluster 2", fontsize = 4, pad = 2.5)
# suD0.text(-0.05, 1.01, "E", transform = suD0.transAxes, fontsize = 7)


#############################################
# F. GRadient
#############################################
mean_burst = pd.DataFrame(columns = ['12', '17','20', '32'])


for m in ['12', '17','20', '32']:
	subspace = pd.read_hdf("../../figures/figures_articles/figure1/subspace_Mouse"+m+".hdf5")	
	nucleus = np.unique(subspace['nucleus'])
	mean_burstiness = [burst.loc[subspace.index, 'sws'][subspace['nucleus'] == nu].mean() for nu in nucleus]
	mean_burst[m] = pd.Series(index = nucleus, data = mean_burstiness)	

nucleus = ['AD', 'LDvl', 'AVd', 'MD', 'AVv', 'IAD', 'CM', 'AM', 'VA', 'Re']

# mean all
meanall = pd.DataFrame(index = nucleus, columns = ['mean','sem'])
for n in nucleus:
	tmp = burst[space['nucleus'] == n]
	if len(tmp)>20:
		meanall.loc[n,'mean'] = tmp.mean(0)['sws']
		meanall.loc[n,'sem'] = tmp.sem(0)['sws']

meanall = meanall.dropna()
nucleus = meanall.index.values

# axF = fig.add_subplot(3,4,11)
axF = Subplot(fig, gs[1,3])
fig.add_subplot(axF)
simpleaxis(axF)
# mean_burst = mean_burst.loc[nucleus]
# mean_burst[0] = np.arange(len(nucleus))
# for i, m in enumerate(['17', '12','20', '32']):	
# 	tmp = mean_burst[[m,0]].dropna()
# 	plot(tmp[m], tmp[0], 'o', label = str(i+1), markersize = 2, linewidth = 1)
# plot(mean_burst.mean(1).values, mean_burst[0].values, 'o-', label = 'Mean', markersize = 2, linewidth = 1, color = 'black')
x, s = (meanall['mean'].values.astype('float32'), meanall['sem'].values.astype('float32'))
plot(x, np.arange(len(nucleus)), 'o-', label = 'Mean', markersize = 2, linewidth = 1, color = 'black')
fill_betweenx(np.arange(len(nucleus)), x-s, x+s, color = 'grey', alpha = 0.5)
leg = legend(frameon=False, bbox_to_anchor = (0.9, 1.15))
# leg.set_title("Mouse", prop={'size':5})
yticks(np.arange(len(nucleus)), nucleus)
xlabel("Burstiness SWS")
ylabel("Nucleus")
# annotate(s='', xy = (4,25), xytext=(6,25), arrowprops=dict(arrowstyle='<->'))
# text(2.5, 24.5, 'Dorsal')
# text(6, 24.5, 'Ventral')
axF.invert_yaxis()
axF.text(-0.1, 1.05, "F", transform = axF.transAxes, fontsize = 9)


#############################################################
# G TSNE NUCLEUS DENSITY
#############################################################
count = pd.read_hdf("../../figures/figures_articles/figure1/count_time.h5")

axG = Subplot(fig, gs[1,4])
fig.add_subplot(axG)
# axF = fig.add_subplot(3,4,12)
simpleaxis(axG)
ylabel("Clustering specificity of HD cells")
xlabel("Time (ms)")
axG.text(-0.1, 1.05, 'G', transform = axG.transAxes, fontsize = 9)
semilogx(count.mean(1), 'o-', markersize = 2, color = 'black', linewidth = 1)

# subplots_adjust(hspace = 0.9)


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

