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
from skimage.filters import gaussian
import hsluv


data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

path = '../../figures/figures_articles_v4/figure1/good_100ms_pickle/'
files = [f for f in os.listdir(path) if '.pickle' in f and 'Mouse' in f]


files.remove("Mouse32-140822.pickle")



###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.2          # height in inches
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
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D



fig = figure(figsize = figsize(1.0))

outergs = gridspec.GridSpec(3,4, figure = fig)

# #############################################
# # A. RINGS MANIFOLD
# #############################################
# for n, f in enumerate(np.sort(files)):
# 	data = cPickle.load(open('../../figures/figures_articles_v4/figure1/good_100ms_pickle/'+f, 'rb'))

# 	iwak		= data['swr'][0]['iwak']
# 	iswr		= data['swr'][0]['iswr']
# 	rip_tsd		= data['swr'][0]['rip_tsd']
# 	rip_spikes	= data['swr'][0]['rip_spikes']
# 	times 		= data['swr'][0]['times']
# 	wakangle	= data['swr'][0]['wakangle']
# 	neurons		= data['swr'][0]['neurons']
# 	tcurves		= data['swr'][0]['tcurves']
# 	irand 		= data['rnd'][0]['irand']
# 	iwak2 		= data['rnd'][0]['iwak2']

# 	tcurves = tcurves.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)

# 	H = wakangle.values/(2*np.pi)

# 	HSV = np.vstack((H*360, np.ones_like(H)*85, np.ones_like(H)*45)).T

# 	RGB = np.array([hsluv.hsluv_to_rgb(HSV[i]) for i in range(len(HSV))])

# 	if n < 4:
# 		subplot(outergs[0,n])
# 	else:
# 		subplot(outergs[1,n%4])

# 	gca().axis('off')		
# 	# gca().text(-0.1, 0.94, "c", transform = gca().transAxes, fontsize = 10, fontweight='bold')
# 	gca().set_aspect(aspect=1)
# 	for i in range(len(iswr)):
# 		scatter(iswr[i,:,0], iswr[i,:,1], c = 'lightgrey', marker = '.', alpha = 0.7, zorder = 2, linewidth = 0, s= 40)

# 	scatter(iwak[~np.isnan(H),0], iwak[~np.isnan(H),1], c = RGB[~np.isnan(H)], marker = '.', alpha = 0.5, zorder = 2, linewidth = 0, s= 40)

# 	if f == 'Mouse17-130129.pickle':		
# 		title(f.split(".")[0]+'\n(excluded)', pad = 0)
# 	else:
# 		title(f.split(".")[0])

# 	if n == 0:
# 		gca().text(-0.2, 1.1, "a", transform = gca().transAxes, fontsize = 10, fontweight='bold')

# 	if n == len(files)-1:
# 		ax = gca()
# 		# colorbar
# 		from matplotlib.colorbar import ColorbarBase
# 		colors = [hsluv.hsluv_to_hex([i,85,45]) for i in np.arange(0, 361)]
# 		cmap= matplotlib.colors.ListedColormap(colors)
# 		# cmap.set_under("hsluv")
# 		# cmap.set_over("w")''''
# 		cax = inset_axes(ax, "40%", "10%",
# 		                   bbox_to_anchor=(1.2, 0.5, 1, 1),
# 		                   bbox_transform=ax.transAxes, 
# 		                   loc = 'lower left')
# 		cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
# 		                                norm=matplotlib.colors.Normalize(vmin=0,vmax=360),
# 		                                orientation='horizontal')
# 		cb1.set_ticks([0,360])
# 		cb1.set_ticklabels(['0', r"$2\pi$"])
# 		cax.set_title("Wake", pad = 3)


#############################################
# B. UP/ DOWN
############################################
data2 = cPickle.load(open('../../figures/figures_articles_v4/figure1/UP_ONSET_SWR.pickle', 'rb'))
allswrrad = data2['radius']
allswrvel = data2['velocity']

tokeep = np.logical_and(allswrrad.columns.values >= 10000, allswrrad.columns.values <= 2*1e6)

allswrrad = allswrrad.iloc[:,tokeep]
allswrvel = allswrvel.iloc[:,tokeep]


allswrvel = allswrvel.rolling(window = 10, win_type='gaussian', center=True, min_periods=1).mean(std=2.0)

gsm = gridspec.GridSpecFromSubplotSpec(1,5, subplot_spec = outergs[2,:], wspace = 0.8, width_ratios = [0.05, 1, 1, 1, 1])

groups = []
bounds = []
for idx in np.array_split(np.arange(allswrrad.shape[1]),4):
	groups.append(allswrrad.iloc[:,idx].mean(1))
	bounds.append((str(int(allswrrad.columns[idx].min()/1000)), str(int(allswrrad.columns[idx].max()/1000))))
groups = pd.concat(groups, 1)
groups.columns = pd.Index(bounds)

ytic = np.unique(np.array(bounds).astype('int'))
ypos = [np.argmin(np.abs(allswrrad.columns.values - ytic[i]*1000)) for i in range(len(ytic))]



# MAPS RADIUS
subplot(gsm[0,1])
imshow(allswrrad.values.T, aspect = 'auto', cmap = 'jet')
xticks([0, 20, 40], [-500, 0, 500])
yticks(ypos, ytic)
xlabel("Time from SWRs (ms)")
ylabel("UP onset (ms)")
title("Radius")
gca().text(-0.5, 1.1, "b", transform = gca().transAxes, fontsize = 10, fontweight='bold')

# GROUPS RADIUS
subplot(gsm[0,2])
simpleaxis(gca())
for c in groups.columns:
	plot(groups[c])
ylabel("Radius (a.u.)")
xlabel("Time from SWRs (ms)")


# MAPS VELOCITY
subplot(gsm[0,3])
imshow(allswrvel.values.T, aspect = 'auto', cmap = 'jet', vmax = 5)
xticks([0, 40], [-487.5, 487.5])
yticks(ypos, ytic)
xlabel("Time from SWRs (ms)")
ylabel("UP onset (ms)")
title("Angular velocity")
gca().text(-0.5, 1.1, "c", transform = gca().transAxes, fontsize = 10, fontweight='bold')

# GROUPS VELOCITY
subplot(gsm[0,4])
simpleaxis(gca())
groups2 = []
bounds = []
for idx in np.array_split(np.arange(allswrvel.shape[1]),4):
	groups2.append(allswrvel.iloc[:,idx].mean(1))
	bounds.append((str(int(allswrvel.columns[idx].min()/1000)), str(int(allswrvel.columns[idx].max()/1000))))
groups2 = pd.concat(groups2, 1)
groups2.columns = pd.Index(bounds)
for c in groups2.columns:
	plot(groups2[c], label = c[0]+'-'+c[1]+' ms')
ylabel("Angular velocity")
xlabel("Time from SWRs (ms)")
legend(loc = 'lower left', fontsize = 7, framealpha=0.0, bbox_to_anchor=(0.05, 1.03)) #, title = 'HD recording sites', )


# subplots_adjust(left = 0.01, right = 0.98, top = 0.98, bottom = 0.1, wspace = 0.2, hspace = 0.9)
subplots_adjust(wspace = -0.1, left = 0.02, right = 0.98, bottom = 0.1, top = 0.95, hspace = 0.2)

# savefig("../../figures/figures_articles_v4/figart_supp_1.pdf", dpi = 100, facecolor = 'white')
savefig(r"/home/guillaume/Dropbox (Peyrache Lab)/Talks/figtalk_supp2.pdf", dpi = 200, facecolor = 'white')
# os.system("evince ../../figures/figures_articles_v4/figart_supp_1.pdf &")

