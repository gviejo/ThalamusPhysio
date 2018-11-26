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

carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
carte38_mouse17_2 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)





mean_burst = pd.DataFrame(columns = ['12', '17','20', '32'])
count_nucl = pd.DataFrame(columns = ['12', '17','20', '32'])

for m in ['12', '17','20', '32']:
	subspace = pd.read_hdf("../../figures/figures_articles/figure1/subspace_Mouse"+m+".hdf5")	
	nucleus = np.unique(subspace['nucleus'])
	mean_burstiness = [burstiness.loc[subspace.index, 'sws'][subspace['nucleus'] == nu].mean() for nu in nucleus]
	mean_burst[m] = pd.Series(index = nucleus, data = mean_burstiness)	
	total = [np.sum(subspace['nucleus'] == n) for n in nucleus]
	count_nucl[m] = pd.Series(index = nucleus, data = total)
# nucleus = ['AD', 'LDvl', 'AVd', 'MD', 'AVv', 'IAD', 'CM', 'AM', 'VA', 'Re']
nucleus = list(count_nucl.dropna().index.values)




positions = {	'AD':[1.0,3.0],
				'LD':[1.5,3.0],
				'AVd':[2.0,2.5],
				'AVv':[2.2,2.0],
				'VA':[2.0,0.5],
				'CM':[-1.0,0.5],
				'PV':[-1.0,1.5],
				'MD':[-1.0,2.5],
				'sm':[0.0,3.0],
				'AM':[2.0,0.0],
				'IAD':[-1.0,2.0]}





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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
colors = ['#444b6e', '#708b75', '#9ab87a']

fig = figure(figsize = figsize(1.0))
gs = gridspec.GridSpec(2,1, wspace = 0.4, hspace = 0.5)
#########################################################################
# RECAPITULATIVE MAPS
#########################################################################
axbig = subplot(gs[0,0])
simpleaxis(axbig)

imshow(carte38_mouse17_2[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
xlim(cut_bound_map[0]-2.0, cut_bound_map[1]+1.0)
ylim(cut_bound_map[2], cut_bound_map[3]+1.0)
grid()


cax = inset_axes(axbig, "20%", "20%",
               bbox_to_anchor=(0, 2.5, 1, 1),
               bbox_transform=axbig.transData, 
               loc = 'lower left')





savefig("../../figures/figures_articles/figart_6.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_6.pdf &")
