

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import matplotlib.cm as cm
import os

###############################################################################################################
# TO LOAD
###############################################################################################################
data = cPickle.load(open('../data/to_plot_examples.pickle', 'rb'))
Hcorr		=	data[	'Hcorr'		]
Hjiter		=	data[	'Hjitt'		]
Z			=	data[	'Z'			]
thmod 		= 	data[	'thmod'		]
theta		=	data[	'theta'		]

xt1 = np.arange(-0.5, 0.51, 0.010)
n_neurons = len(Hcorr)

neurons = [0, 1, 3]
neurons = [2,0,3]


# filter everything
# for i in neurons:
# 	Hcorr[i] = gaussFilt(Hcorr[i], (1,))
# 	Hjiter[i] = gaussFilt(Hjiter[i], (1,))
# 	Z[i] = gaussFilt(Z[i], (1,))

###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.4            # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)
	# ax.spines['left'].set_position('zero')
	# ax.spines['bottom'].set_position('zero')


import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 5,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 4,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 4,
	"ytick.labelsize": 4,
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
from mpl_toolkits.axes_grid.inset_locator import inset_axes

colors = ['#3d315b','#444b6e', '#708b75', '#9ab87a']


fig = figure(figsize = figsize(1))
gs = gridspec.GridSpec(20,3, wspace = 0.4, hspace = 0.6)


ax = subplot(gs[0,0])
for i,j in zip(neurons, range(len(neurons))):
	ax = subplot(gs[0:4,j])
	simpleaxis(ax)
	plot(xt1*1000., Hcorr[i], color = colors[i])
	plot(xt1*1000., Hjiter[i], ':', color = colors[i], label = 'Jitter')
	axvline(0, color = 'grey', alpha = 0.5)
	ylabel('Firing rate (Hz)')
	legend(edgecolor = None, facecolor = None, frameon = False)	
	title('Thalamic neuron')
	ax = subplot(gs[5:9,j])
	simpleaxis(ax)
	plot(xt1*1000., Z[i], color = colors[i])
	xlabel('Time from \n $\mathbf{Sharp\ Waves\ ripples}$ (ms)', fontsize = 8)
	ylabel('Modulation (a.u.)')
	axvline(0, color = 'grey', alpha = 0.5)


n = theta.shape[1]
bins = np.linspace(0, 2*np.pi+0.0001, n)

for i,j in zip(neurons,range(len(neurons))):
	ax = subplot(gs[12:,j])
	simpleaxis(ax)
	bar(bins, theta[i], bins[1]-bins[0]-0.05, color = colors[i])
	yt = gaussFilt(theta[i], (2,))
	plot(bins, yt, color = colors[i])	
	axvline(bins[np.argmax(yt)], color = 'red')
	ylabel("Theta spike count (a.u.)")
	xlabel("$\mathbf{Theta\ phase}$", fontsize = 8)
	xticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '$2\pi$'])





savefig("../figures/fig6.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../figures/fig6.pdf &")

