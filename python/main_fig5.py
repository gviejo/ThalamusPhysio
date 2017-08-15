

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
data = cPickle.load(open('../data/to_plot_corr_pop.pickle', 'rb'))
xt 		 	= data	['xt'	  ]
meanywak 	= data['meanywak']
meanyrem 	= data['meanyrem']
meanyrip 	= data['meanyrip']
toplot 	= data['toplot'  ]
varywak		= data['varywak']
varyrem		= data['varyrem']
varyrip		= data['varyrip']

###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.5            # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)


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
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
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
from mpl_toolkits.axes_grid.inset_locator import inset_axes


fig = figure(figsize = figsize(1))
# outer = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)#, height_ratios = [1,3])#, width_ratios = [1.6,0.7]) 
gs = gridspec.GridSpec(2,2, wspace = 0.35, hspace = 0.35)#, wspace = 0.4, hspace = 0.4)
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[0])

ax = subplot(gs[0,0])
axp = imshow(toplot['rem'][100:,100:], origin = 'lower', cmap = 'gist_heat')
ylabel("Theta Cycle")
xlabel("Theta Cycle")
title('REM SLEEP')
cbaxes = fig.add_axes([0.4, 0.5, 0.25, 0.01])
cb = colorbar(axp, cax = cbaxes, orientation = 'horizontal', cmap = 'gist_heat')
cbaxes.title.set_text('r')

ax = subplot(gs[0, 1])
imshow(toplot['wake'], interpolation = None, origin = 'lower', cmap = 'gist_heat')
ylabel('Theta Cycle')
xlabel('Theta Cycle')
title("WAKE")


ax = subplot(gs[1, 0])
imshow(toplot['rip'][0:200,0:200], origin = 'lower', cmap = 'gist_heat')
title('SHARP-WAVES RIPPLES')
xlabel('SWR')
ylabel('SWR')


ax = subplot(gs[1,1])
simpleaxis(ax)
# xtsym = np.array(list(xt[::-1]*-1.0)    +list(xt))
# meanywak = np.array(list(meanywak[::-1])+list(meanywak))
# meanyrem = np.array(list(meanyrem[::-1])+list(meanyrem))
# meanyrip = np.array(list(meanyrip[::-1])+list(meanyrip))
# varywak  = np.array(list(varywak[::-1]) +list(varywak))
# varyrem  = np.array(list(varyrem[::-1]) +list(varyrem))
# varyrip  = np.array(list(varyrip[::-1]) +list(varyrip))

colors = ['red', 'blue', 'green']
colors = ['#231123', '#af1b3f', '#ccb69b']

# plot(xtsym, meanywak, '-', color = colors[0], label = 'theta(wake)')
# plot(xtsym, meanyrem, '-', color = colors[1], label = 'theta(rem)')
# plot(xtsym, meanyrip, '-', color = colors[2], label = 'ripple')
# fill_between(xtsym, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
# fill_between(xtsym, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
# fill_between(xtsym, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)
plot(xt, meanyrem, '-', color = colors[1], label = 'REM')
plot(xt, meanywak, '-', color = colors[0], label = 'WAKE')
plot(xt, meanyrip, '-', color = colors[2], label = 'SWR')
fill_between(xt, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
fill_between(xt, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
fill_between(xt, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)

xt = xt[::-1]*-1.0
meanywak = meanywak[::-1]
meanyrem = meanyrem[::-1]
meanyrip = meanyrip[::-1]
varywak = varywak[::-1]
varyrem = varyrem[::-1]
varyrip = varyrip[::-1]
plot(xt, meanyrem, '-', color = colors[1])
plot(xt, meanywak, '-', color = colors[0])
plot(xt, meanyrip, '-', color = colors[2])
fill_between(xt, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
fill_between(xt, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
fill_between(xt, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)


axvline(0, linestyle = '--', color = 'grey')

# ylim(0.0, 0.3)

legend(edgecolor = None, facecolor = None, frameon = False)
xlabel('Time between events (s) ')
ylabel("r")


savefig("../figures/fig5.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../figures/fig5.pdf &")

