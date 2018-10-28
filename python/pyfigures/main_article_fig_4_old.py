

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
data = cPickle.load(open('../../data/to_plot_corr_pop.pickle', 'rb'))
xt 		 	= data	['xt'	  ]
meanywak 	= data['meanywak']
meanyrem 	= data['meanyrem']
meanyrip 	= data['meanyrip']
toplot 	= data['toplot'  ]
varywak		= data['varywak']
varyrem		= data['varyrem']
varyrip		= data['varyrip']

carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)

space = pd.read_hdf("../../figures/figures_articles/figure1/space.hdf5")

subspace = space[space.index.str.contains('Mouse17')]

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datasets = [s for s in datasets if 'Mouse17' in s]

to_show = 7

tocut_corrmatrix = {
	'wake':(275,475),
	'rem':(5330,5600),
	'sws':(600,750)}


mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
lambdaa  = pd.read_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_AUTOCORR.h5")
nucleus = np.unique(mappings['nucleus'])
lambdaa_nucleus = pd.DataFrame(	index = nucleus, 
								columns = pd.MultiIndex.from_product([['wak', 'rem', 'sws'], ['mean', 'sem']], 
								names = ['episode', 'mean-sem']))
for n in nucleus:
	tmp = lambdaa.loc[mappings.index[mappings['nucleus'] == n]]
	for e in ['wak', 'rem', 'sws']:
		lambdaa_nucleus.loc[n,(e,'mean')] = tmp[e].mean(skipna=True).loc['b']
		lambdaa_nucleus.loc[n,(e,'sem')] = tmp[e].sem(skipna=True).loc['b']


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

mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	# "text.usetex": True,                # use LaTeX to write all text
	# "font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 6,               # LaTeX default is 10pt font.
	"font.size": 6,
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
from mpl_toolkits.axes_grid.inset_locator import inset_axes


fig = figure(figsize = figsize(1))

ax1 = subplot(211)

# outer = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)#, height_ratios = [1,3])#, width_ratios = [1.6,0.7]) 
# gs = gridspec.GridSpec(2,3, wspace = 0.35, hspace = 0.35)#, wspace = 0.4, hspace = 0.4)
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[0])
# gs = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec = ax1)
colors = ['red', 'blue', 'green']
###################################################################################################
# A. AUTOCORRELOGRAM EXEMPLE
###################################################################################################
axA = subplot(4,3,1)
simpleaxis(axA)


def func(x, a, b, c):
	return a*np.exp(-b*x) + c

tmp = mappings[np.logical_and(mappings['hd'] == 0, mappings['nucleus'] == 'AM')]
idx = tmp[tmp.index.str.contains('Mouse12')].index.values

hd = 'Mouse12-120807_28'
nhd = 'Mouse12-120818_0'

autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG.h5")

# HD
tmp = autocorr['wak'][hd]
# tmp.loc[0] = 0
tmp = tmp.drop(0)
plot(tmp.index.values*1e-3, tmp.values, label = 'AD (HD)', color = 'darkgreen')
x = tmp.loc[5.01:].index.values*1e-3
plot(x, func(x, *lambdaa.loc[hd, 'wak'].values), '--', color = 'grey')
# NON HD
tmp = autocorr['wak'][nhd]
# tmp.loc[0] = 0
tmp = tmp.drop(0)
plot(tmp.index.values*1e-3, tmp.values, label = 'AM', color = 'darkblue')
print(nhd)
x = tmp.loc[5.01:].index.values*1e-3
plot(x, func(x, *lambdaa.loc[nhd, 'wak'].values), '--', color = 'grey', label = 'Exp fit')
legend(edgecolor = None, facecolor = None, frameon = False, bbox_to_anchor=(1.1, 1.1), bbox_transform=axA.transAxes)
xlabel("Time (s)")
ylabel("Autocorrelation")
locator_params(nbins = 4)

###################################################################################################
# B. LAMBDA AUTOCORRELOGRAM / NUCLEUS
###################################################################################################
axB = subplot(4,3,2)
simpleaxis(axB)
order = lambdaa_nucleus[('wak', 'mean')].sort_values().index
order = order.drop(['U', 'sm'])

labels = ['Wake', 'REM']

for i, ep in enumerate(['wak', 'rem']):
	m = lambdaa_nucleus.loc[order,(ep,'mean')].values.astype('float32')
	s = lambdaa_nucleus.loc[order,(ep,'sem')].values.astype('float32')
	plot(m, np.arange(len(order)), 'o-', color = colors[i], label = labels[i], markersize = 3, linewidth = 1)
	fill_betweenx(np.arange(len(order)), m+s, m-s, color = colors[i], alpha = 0.3)

legend(edgecolor = None, facecolor = None, frameon = False)
yticks(np.arange(len(order)), order.values)
ylabel("Nucleus")	
xlabel("Exp fit $\lambda$ (s)")
locator_params(axis = 'x', nbins = 4)

###################################################################################################
# C. BURSTINESS VS LAMBDA
###################################################################################################
axC = subplot(4,3,3)
simpleaxis(axC)

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
idx = lambdaa['rem']['b'].index
scatter(burst['sws'].loc[idx], lambdaa['wak']['b'].loc[idx], 4, color = colors[0], alpha = 0.5, edgecolors = 'none')
xlabel("Burstiness")
ylabel("Exp fit $\lambda$ (s)")

####################################################################################################
# D. MAP
####################################################################################################
axA = subplot(4,4,5)
noaxis(axA)
cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)
imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
line = subspace[subspace['session'] == to_show][['x', 'y']]
plot(line['x'], line['y'], '-', color = 'red', linewidth = 1)
scatter(subspace['x'], subspace['y'], s = 0.5, color = 'black', edgecolor = 'none')



store 			= pd.HDFStore("/mnt/DataGuillaume/corr_pop/"+datasets[to_show].split("/")[1]+".h5")
####################################################################################################
# E. WAKE
####################################################################################################
# axB = subplot(3,4,2)
ax = subplot2grid((4,4), (1,1), colspan = 3)
gs = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec = ax, wspace=0.8, width_ratios = [0.01, 1, 1, 1])
axB = subplot(gs[0,1])
a, b = tocut_corrmatrix['wake']
im = store['wak_corr'].iloc[a:b,a:b]
np.fill_diagonal(im.values, 1.0)

axp = imshow(im, interpolation = None, origin = 'lower')
ylabel("Theta Cycle")
xlabel("Theta Cycle")
title('WAKE', fontsize = 7)

#colorbar	
cax = inset_axes(axB, "40%", "4%",
                   bbox_to_anchor=(-0.7, -0.2, 1, 1),
                   bbox_transform=axB.transAxes, 
                   loc = 'lower left')
cb = colorbar(axp, cax = cax, orientation = 'horizontal', ticks = [0, 1])
cb.set_label('r' , labelpad = -0)
cb.ax.xaxis.set_tick_params(pad = 1)

####################################################################################################
# F. REM
####################################################################################################
# axC = subplot(3,4,3)
axC = subplot(gs[0,2])
a, b = tocut_corrmatrix['rem']
im = store['rem_corr'].iloc[a:b,a:b]
np.fill_diagonal(im.values, 1.0)
imshow(im, origin = 'lower', interpolation = None)
ylabel('Theta Cycle')
xlabel('Theta Cycle')
title("REM")

####################################################################################################
# G. SWS
####################################################################################################
axD = subplot(gs[0,3])
a, b = tocut_corrmatrix['sws']
im = store['rip_corr'].iloc[a:b,a:b]
np.fill_diagonal(im.values, 1.0)
imshow(im, origin = 'lower', interpolation = None)	
xlabel('SWR')
ylabel('SWR')
title('RIPPLES')

store.close()

####################################################################################################
# H. CROSS_CORR
####################################################################################################
axcc = subplot(4,1,3)
gs = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec = axcc, wspace=0.3, width_ratios = [1, 0.5, 0.5, 0.5])
# axE = subplot(3,3,4)
axE = subplot(gs[0,0])
simpleaxis(axE)


plot(xt, meanywak, '-', color = colors[0], label = 'WAKE')
plot(xt, meanyrem, '-', color = colors[1], label = 'REM')
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
legend(edgecolor = None, facecolor = None, frameon = False)
xlabel('Time between events (s) ')
ylabel("Correlation population")

locator_params(nbins = 4)

###################################################################################################
# I. HD VERSUS NON HD
###################################################################################################
meancorr = pd.read_hdf("../../figures/figures_articles/figure4/meancorr_hd_nohd.h5")
titles = ['WAKE', 'REM', 'RIPPLES']

for i, e in enumerate(['wak', 'rem', 'rip']):
	ax = subplot(gs[0,i+1])
	simpleaxis(ax)
	for c, l, n in zip(['hd', 'nohd'], ['-', '--'], ['HD', 'No HD']):
		plot(meancorr[e][c]['mean'], linestyle = l, color = colors[i], label = n)
	xlabel("Time between events (s)")
	title(titles[i])
	if i == 0:
		legend(edgecolor = None, facecolor=None, frameon = False, bbox_to_anchor=(0.65, 1.0), bbox_transform=ax.transAxes)
	axvline(0, linestyle = '--', color = 'grey')
	locator_params(nbins = 4)

###################################################################################################
# J. CORRPOP NUCLEUS
###################################################################################################
dfnucleus = pd.read_hdf("../../figures/figures_articles/figure4/meancorrpop_nucleus.h5")

for i, e in enumerate(['wak', 'rem', 'rip']):
	ax = subplot(4,4,13+i)
	simpleaxis(ax)
	tmp = dfnucleus.xs(e, 1, 1)
	for n in np.unique(tmp.columns.get_level_values(0)):
		plot(tmp[n]['mean'], label = n)
	if i == 0:
		legend(edgecolor = None, facecolor = None, frameon = False)

###################################################################################################
# H. TIME CONSTANT NUCLEUS
###################################################################################################
lambdaa = pd.read_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_POPCORR_NUCLEUS.h5", 'lambdaanucleus')

lambdaa = lambdaa.sort_values(('wak', 'b'))

tmp = lambdaa.xs('b', 1, 1)

subplot(4,4,16)
plot(tmp['wak'].values, np.arange(len(tmp.index)), label = 'Wake')
plot(tmp['rem'].values, np.arange(len(tmp.index)), label = 'REM')

yticks(np.arange(len(tmp.index)), tmp.index)




fig.subplots_adjust(wspace= 0.4, hspace = 0.6)


savefig("../../figures/figures_articles/figart_4.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_4.pdf &")

