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
store = pd.HDFStore("../../figures/figures_articles_v2/figure6/determinant_corr_noSWS.h5", 'r')
det_all = store['det_all']
shufflings = store['shufflings']
store.close()


data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')


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
gs = gridspec.GridSpec(1,2, wspace = 0.3)


#########################################################################
# A. WAKE REM SWS INDIV
#########################################################################
subplot(gs[0,0])
simpleaxis(gca())
gca().text(-0.2, 1.0, "A", transform = gca().transAxes, fontsize = 9)

# colors = ['blue', 'red', 'green']
colors = ["#CA3242","#849FAD",  "#27647B", "#57575F"]
labels = ['WAKE', 'REM', 'NREM']
offset = [0.032, 0.032, 0.026]
for i, k in enumerate(['wak', 'rem', 'sws']):
	shuf, x = np.histogram(1-shufflings[k], bins = 100, weights = np.ones(len(shufflings[k]))/len(shufflings[k]))
	axvline(1-det_all[k], color = colors[i])
	# plot([1-det_all[k], 1-det_all[k]], [0, 0.032], color = colors[i], label = labels[i])
	plot(x[0:-1]+np.diff(x), shuf, color = colors[i], alpha = 0.7)
	# hist(, label = k, histtype='stepfilled', facecolor = 'None', edgecolor = colors[i])
	# gca().text(1-det_all[k], gca().get_ylim()[1], "p<0.001",fontsize = 7, ha = 'center', color = 'red')
	gca().text(1-det_all[k], offset[i], labels[i], ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor=colors[i]))
axvline(0.33, color = 'black', linestyle = '--')
gca().text(0.33, 0.032, 'ALL', ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor='black'))
ylim(0, 0.035)
xlabel(r"Total correlation $\rho^{2}$")
ylabel("Probability (%)")
yticks([0,0.01,0.02,0.03], ['0','1','2','3'])
# gca().text(-0.15, 1.0, "A", transform = gca().transAxes, fontsize = 9)
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.35, 0.6))	

# #########################################################################
# # B. WAKE REM ONLY
# #########################################################################
store = pd.HDFStore("../../figures/figures_articles_v2/figure6/determinant_corr_noSWS_shank_shuffled.h5", 'r')
det_all = store['det_all']
shufflings = store['shufflings']
store.close()

subplot(gs[0,1])
simpleaxis(gca())
gca().text(-0.2, 1.0, "B", transform = gca().transAxes, fontsize = 9)

# colors = ['blue', 'red', 'green']
labels = ['WAKE', 'REM', 'NREM']
offset = [0.15, 0.15, 0.12]
for i, k in enumerate(['wak', 'rem', 'sws']):
	shuf, x = np.histogram(1-shufflings[k], bins = 20, weights = np.ones(len(shufflings[k]))/len(shufflings[k]))
	axvline(1-det_all[k], color = colors[i])
	# plot([1-det_all[k], 1-det_all[k]], [0, 0.032], color = colors[i], label = labels[i])
	plot(x[0:-1]+np.diff(x), shuf, color = colors[i], alpha = 1)
	# hist(, label = k, histtype='stepfilled', facecolor = 'None', edgecolor = colors[i])
	# gca().text(1-det_all[k], gca().get_ylim()[1], "p<0.001",fontsize = 7, ha = 'center', color = 'red')
	gca().text(1-det_all[k], offset[i], labels[i], ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor=colors[i]))
axvline(0.33, color = 'black', linestyle = '--')
gca().text(0.33, 0.15, 'ALL', ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor='black'))
ylim(0, 0.16)
xlabel(r"Total correlation $\rho^{2}$")
ylabel("Probability (%)")
yticks([0,0.05,0.10,0.15], ['0','5','10','15'])
# gca().text(-0.15, 1.0, "A", transform = gca().transAxes, fontsize = 9)
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.35, 0.6))	

store = pd.HDFStore("../../figures/figures_articles_v2/figure6/determinant_corr.h5", 'r')
det_all = store['det_all']
shufflings = store['shufflings']
shuffl_shank = store['shuffling_shank']
store.close()

shuf, x = np.histogram(1-shuffl_shank, bins = 20, weights = np.ones(len(shuffl_shank))/len(shuffl_shank))
plot(x[0:-1]+np.diff(x), shuf, '--', color = colors[-1], alpha = 0.7)

# subplot(gs[0,1])
# simpleaxis(gca())
# axvline(1-det_all['wak-rem'], color = 'gray')
# hist(1-shufflings['wak-rem'], 100, color = 'gray', weights = np.ones(len(shufflings['wak-rem']))/len(shufflings['wak-rem']), label = 'Wake-REM', histtype='stepfilled')
# xlabel(r"Total correlation $\rho^{2}$")
# ylabel("Probability (%)")
# yticks([0,0.02,0.04], ['0','2','4'])
# # gca().text(1-det_all['wak-rem']-0.05, gca().get_ylim()[1], "p<0.001",fontsize = 7, ha = 'center', color = 'red')
# # legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.35, 0.6))
# axvline(0.33, color = 'black', linestyle = '--')
# gca().text(1-det_all['wak-rem'], 0.036, 'WAKE\nREM', ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor='gray'))
# gca().text(0.33, 0.039, 'ALL', ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor='black'))
# gca().text(-0.15, 1.0, "B", transform = gca().transAxes, fontsize = 9)


subplots_adjust(top = 0.93, bottom = 0.2, right = 0.96, left = 0.08)

savefig("../../figures/figures_articles_v2/figart_supp_3.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v2/figart_supp_3.pdf &")
