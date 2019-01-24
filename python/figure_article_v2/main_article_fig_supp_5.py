import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
import sys
sys.path.append("../")
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import matplotlib.cm as cm
import os


###############################################################################################################
# TO LOAD
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr_mod 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (20,)).transpose())
swr_mod = swr_mod.drop(swr_mod.columns[swr_mod.isnull().any()].values, axis = 1)
swr_mod = swr_mod.loc[-500:500]

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")


index = mappings.groupby(['nucleus']).groups

swr_MD = swr_mod[index['MD']]
swr_AD = swr_mod[index['AD']]

###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*0.5          # height in inches
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
import matplotlib.cm as cmx
import matplotlib.colors as colors
# colors = ['#444b6e', '#708b75', '#9ab87a']

fig = figure(figsize = figsize(1.0))
gs = gridspec.GridSpec(1,3, wspace = 0.3)

colors = ["#CA3242","#849FAD",  "#27647B", "#57575F"]
#########################################################################
# A. MD SWR
#########################################################################
subplot(gs[0,0])
simpleaxis(gca())
gca().text(-0.3, 1.0, "A", transform = gca().transAxes, fontsize = 9)

plot(swr_MD, color = colors[1], alpha = 0.7, linewidth = 0.8)
plot(swr_MD.mean(1), color = 'black')

xlabel("Time lag (ms)")
ylabel("z (a.u.)")
ylim(-0.61,0.76)
title("MD", pad = 0.0)


# #########################################################################
# # B. AD SWR
# #########################################################################
subplot(gs[0,1])
simpleaxis(gca())
gca().text(-0.3, 1.0, "B", transform = gca().transAxes, fontsize = 9)

plot(swr_AD, color = colors[0], alpha = 0.7, linewidth = 0.8)
plot(swr_AD.mean(1), color = 'black')

xlabel("Time lag (ms)")
ylabel("z (a.u.)")
ylim(-0.61,0.76)
title("AD", pad = 0.5)
#########################################################################
# C. hist z 50 ms
#########################################################################
subplot(gs[0,2])
simpleaxis(gca())
gca().text(-0.3, 1.0, "C", transform = gca().transAxes, fontsize = 9)
hist(swr_AD.loc[-50], label = 'AD', bins = 20, alpha = 0.7, color = colors[0], histtype='stepfilled', weights = np.ones(swr_AD.shape[1])/swr_AD.shape[1])
hist(swr_MD.loc[-50], label = 'MD', bins = 20, alpha = 0.7, color = colors[1], histtype='stepfilled', weights = np.ones(swr_MD.shape[1])/swr_MD.shape[1])
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'upper right')
yticks([0, 0.05, 0.1, 0.15], ['0', '5', '10', '15'])

ylabel("%")
xlabel("z (t=-50 ms)")
subplots_adjust(top = 0.93, bottom = 0.2, right = 0.96, left = 0.08)

savefig("../../figures/figures_articles_v2/figart_supp_5.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v2/figart_supp_5.pdf &")
