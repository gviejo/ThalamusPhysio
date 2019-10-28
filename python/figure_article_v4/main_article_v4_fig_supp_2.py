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


data = cPickle.load(open("../../figures/figures_articles_v4/figure1/decodage_bayesian.pickle", 'rb'))


swrvel = data['swrvel']
count = data['count']


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


fig = figure(figsize = figsize(1.0))#, tight_layout=True)

outergs = gridspec.GridSpec(1,2, figure = fig, hspace = 0.4, wspace = 0.4)


###################################################################################################
# A. AUTOCORRELOGRAM EXEMPLE
###################################################################################################
axI = Subplot(fig, outergs[0,0])
fig.add_subplot(axI)
simpleaxis(axI)
axhline(0, color = 'grey', linewidth = 0.5, alpha = 0.6)
plot(swrvel[count[count>10].index], linewidth = 1, color = 'grey')
plot(swrvel[count[count>10].index].mean(1), linewidth = 3, color = 'black')
xlabel("Time from SWRs (ms)")
ylabel("Angular velocity")
ylim(-0.3, 0.3)
title("# HD neurons > 10")

axI.text(-0.2, 1.0, "a", transform = axI.transAxes, fontsize = 10, fontweight='bold')

###################################################################################################
# B. WAKE TAU VS REM TAU
###################################################################################################
# axJ = Subplot(fig, outergs[0,1])
# fig.add_subplot(axJ)
# simpleaxis(axJ)
# axhline(0, color = 'grey', linewidth = 0.5, alpha = 0.6)
# plot(swrvel[count[count<10].index], linewidth = 1, color = 'grey')
# plot(swrvel[count[count<10].index].mean(1), linewidth = 3, color = 'black')
# ylim(-0.3, 0.3)
# title("10 > HD neurons > 5")
# xlabel("Time from SWRs (ms)")

# axJ.text(-0.3, 1.0, "b", transform = axJ.transAxes, fontsize = 10, fontweight='bold')

###################################################################################################
# C. BURSTINESS VS LAMBDA
###################################################################################################
axK = Subplot(fig, outergs[0,1])
fig.add_subplot(axK)
simpleaxis(axK)
axhline(0, color = 'grey', linewidth = 0.5, alpha = 0.6)
plot(swrvel[count[count>10].index], linewidth = 1, color = 'grey')
plot(swrvel[count[count<10].index], linewidth = 1, color = 'green')
plot(swrvel[count[count>10].index[0]], linewidth = 1, color = 'grey', label = ">10")
plot(swrvel[count[count<10].index[0]], linewidth = 1, color = 'green', label = "<10")
plot(swrvel.mean(1), linewidth = 3, color = 'black')
ylim(-0.3, 0.3)
title("# HD neurons > 5")
xlabel("Time from SWRs (ms)")
legend()

axK.text(-0.2, 1.0, "b", transform = axK.transAxes, fontsize = 10 ,fontweight='bold')





subplots_adjust(bottom = 0.2, top = 0.9, right = 0.98, left = 0.08)

savefig("../../figures/figures_articles_v4/figart_supp_2.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v4/figart_supp_2.pdf &")

