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
data = cPickle.load(open('../../figures/figures_articles/figure3/dict_fig3_article.pickle', 'rb'))
allzth 			= 	data['swr_modth'	]
eigen 			= 	data['eigen'		]		
times 			= 	data['times' 		]
allthetamodth 	= 	data['theta_modth'	]		
phi 			= 	data['phi' 			]		
zpca 			= 	data['zpca'			]		
phi2			= 	data['phi2' 		]	 # jpca phase					
jX				= 	data['rX'			]
jscore			= 	data['jscore'		]
force 			= 	data['force'		] # theta modulation
variance 		= 	data['variance'		] # ripple modulation


index = allzth[0].sort_values().index.values
index = index[::-1]
allzthsorted = allzth.loc[index]
phi = phi.loc[index]
phi2 = phi2.loc[index]
allthetamodth = allthetamodth.loc[index]
allthetamodth['phase'] += 2*np.pi
allthetamodth['phase'] %= 2*np.pi

data = pd.concat([phi2, allthetamodth['phase']], axis =1)
data.columns = ['swr', 'theta']

space = pd.read_hdf("../../figures/figures_articles/figure1/space.hdf5")

data['nucleus'] = space.loc[index,['nucleus']]

data = data.dropna()

data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
burstiness 				= pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta 					= pd.DataFrame(	index = theta_ses['rem'], 
									columns = ['phase', 'pvalue', 'kappa'],
									data = theta_mod['rem'])
rippower 				= pd.read_hdf("../../figures/figures_articles/figure2/power_ripples_2.h5")
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")

neurons = np.intersect1d(np.intersect1d(burstiness.index.values, theta.index.values), rippower.index.values)











sys.exit()






###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*2.0          # height in inches
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
colors = ['#444b6e', '#708b75', '#9ab87a']

fig = figure(figsize = figsize(1.0))

#########################################################################
# A HIST PHASE RIPPLES
#########################################################################
axA = subplot(3,2,1)
simpleaxis(axA)

nucleus = np.unique(data['nucleus'])
bins = np.linspace(0, 2*np.pi, 9)
hist_rip = pd.DataFrame(index = list(bins[0:-1]+(bins[1]-bins[0])/2)+[2*np.pi+(bins[1]-bins[0])/2], columns = nucleus, data = 0)
for i, n in enumerate(nucleus):
	idx = np.digitize(data[data['nucleus'] == n]['swr'], bins)-1
	for j in range(hist_rip.shape[0]):
		hist_rip.iloc[j,i] = np.sum(idx == j)	

hist_rip = hist_rip/hist_rip.max()
hist_rip.iloc[-1] = hist_rip.iloc[0]

# order_nucleus = hist_rip.idxmax().sort_values().index.values
order_nucleus = ['AD', 'AVd', 'AVv', 'AM']


for i, n in enumerate(order_nucleus):
	plot(hist_rip[n]+i, label = n)
yticks(np.arange(len(order_nucleus)), order_nucleus)

#########################################################################
# C ORBIT PHASE RIPPLES
#########################################################################
axC = subplot(3,2,3, projection = 'polar')
# simpleaxis(axC)

for i, n in enumerate(order_nucleus):
	plot(hist_rip[n], label = n)


#########################################################################
# B HIST PHASE THETA
#########################################################################
axB = subplot(3,2,2)
simpleaxis(axB)

nucleus = np.unique(data['nucleus'])
hist_theta = pd.DataFrame(index = list(bins[0:-1]+(bins[1]-bins[0])/2)+[2*np.pi+(bins[1]-bins[0])/2], columns = nucleus, data = 0)
for i, n in enumerate(nucleus):
	idx = np.digitize(data[data['nucleus'] == n]['theta'], bins)-1
	for j in range(hist_theta.shape[0]):
		hist_theta.iloc[j,i] = np.sum(idx == j)	

hist_theta = hist_theta/hist_theta.max()
hist_theta.iloc[-1] = hist_theta.iloc[0]

# order_nucleus = hist_theta.idxmax().sort_values().index.values
order_nucleus = ['AD', 'AVd', 'AVv', 'AM']

for i, n in enumerate(order_nucleus):
	plot(hist_theta[n]+i, label = n)
yticks(np.arange(len(order_nucleus)), order_nucleus)



#########################################################################
# D ORBIT PHASE THETA
#########################################################################
axD = subplot(3,2,4, projection = 'polar')
# simpleaxis(axD)

for i, n in enumerate(order_nucleus):
	plot(hist_theta[n], label = n)


savefig("../../figures/figures_articles/figart_5.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_5.pdf &")