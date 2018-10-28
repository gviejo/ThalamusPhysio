

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
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
lambdaa  = pd.read_hdf("/mnt/DataGuillaume/MergedData/LAMBDA_AUTOCORR.h5")
nucleus = np.unique(mappings['nucleus'])
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




###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*0.5           # height in inches
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
autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_LONG.h5")
axA = subplot(1,3,1)
simpleaxis(axA)


def func(x, a, b, c):
	return a*np.exp(-(1./b)*x) + c


labels = ['AD (HD)', 'AM', 'IAD']
colors = ['darkgreen', 'darkblue', 'darkred']


hd = 'Mouse12-120807_28'
iad = 'Mouse12-120817_52'
nhd = 'Mouse12-120819_3'
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
		plot(x, y, '--', color = 'grey', label = "Exp fit \n " r"$y=a e^{\frac{-t}{\tau}} + b$")
	else:
		plot(x, y, '--', color = 'grey')

# show()


legend(edgecolor = None, facecolor = None, frameon = False, bbox_to_anchor=(0.5, 1.1), bbox_transform=axA.transAxes)
xlabel("Time (s)")
ylabel("Autocorrelation (a.u)")
locator_params(nbins = 4)

###################################################################################################
# B. LAMBDA AUTOCORRELOGRAM / NUCLEUS
###################################################################################################
axB = subplot(1,3,2)
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
xlabel(r"Exp fit $\tau$ (s)")
locator_params(axis = 'x', nbins = 4)

###################################################################################################
# C. BURSTINESS VS LAMBDA
###################################################################################################
axC = subplot(1, 3,3)
simpleaxis(axC)

# firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
# fr_index = firing_rate.index.values[((firing_rate[['wake', 'rem']] > 1.0).sum(1) == 2).values]


burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
idx = lambdaa['rem']['b'].index

df = pd.concat([burst['sws'].loc[idx], lambdaa['wak']['b'].loc[idx]], axis = 1).rename(columns={'sws':'burst','b':'lambda'})

df = df[np.logical_and(df['burst']<25,df['lambda']<3)]


from scipy.stats import pearsonr

scatter(df['burst'].values, df['lambda'].values, 4, color = colors[0], alpha = 0.5, edgecolors = 'none')
xlabel("Burstiness")
ylabel(r"Exp fit $\tau$ (s)")

a, b = pearsonr(df['burst'].values, df['lambda'].values)

print(a, b)


fig.subplots_adjust(wspace= 0.4, hspace = 0.6)


savefig("../../figures/figures_articles/figart_4.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_4.pdf &")

