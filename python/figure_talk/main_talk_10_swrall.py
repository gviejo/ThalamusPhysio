

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
from scipy.ndimage import gaussian_filter	

###############################################################################################################
# TO LOAD
###############################################################################################################
data = cPickle.load(open('../../figures/figures_articles_v2/figure3/dict_fig3_article.pickle', 'rb'))
allzth 			= 	data['swr_modth'	]
eigen 			= 	data['eigen'		]		
times 			= 	data['times' 		]
allthetamodth 	= 	data['theta_modth'	]		
phi 			= 	data['phi' 			]		
zpca 			= 	data['zpca'			]		
phi2			= 	data['phi2' 		]	 					
jX				= 	data['rX'			]
jscore			= 	data['jscore'		]
force 			= 	data['force'		] # theta modulation
variance 		= 	data['variance'		] # ripple modulation


# sort allzth 
index = allzth[0].sort_values().index.values
index = index[::-1]
allzthsorted = allzth.loc[index]
phi = phi.loc[index]
phi2 = phi2.loc[index]
allthetamodth = allthetamodth.loc[index]

theta2 = pd.read_hdf("/mnt/DataGuillaume/MergedData/THETA_THAL_mod_2.h5")
theta2 = theta2['rem']

# REPlACING WITH VERSION 2 OF THETA MOD HERE
allthetamodth = theta2.loc[allthetamodth.index]
allthetamodth.rename({'pval':'pvalue'}, inplace=True)

allthetamodth['phase'] += 2*np.pi
allthetamodth['phase'] %= 2*np.pi


spikes_theta_phase = cPickle.load(open('/mnt/DataGuillaume/MergedData/SPIKE_THETA_PHASE.pickle', 'rb'))


###############################################################################################################
# PLOT11
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean            # height in inches
	fig_size = [fig_width*1.2,fig_height]
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
from mpl_toolkits.axes_grid.inset_locator import inset_axes


fig = figure(figsize = figsize(1), tight_layout = True)
# outer = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)#, height_ratios = [1,3])#, width_ratios = [1.6,0.7]) 
gs = gridspec.GridSpec(1,2)

###############################################################################################################
# TO LOAD
###############################################################################################################
def softmax(x, b1 = 10.0, b2 = 0.5, lb = 0.2):
	x -= x.min()
	x /= x.max()
	return (1.0/(1.0+np.exp(-(x-b2)*b1)) + lb)/(1.0+lb)

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

nbins 					= 200
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
times2 					= swr_mod.index.values

# carte38_mouse17_2 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
# bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
# cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

carte_adrien = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_ALL-01.png')
carte_adrien2 = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_Contour-01.png')
bound_adrien = (-398/1254, 3319/1254, -(239/1254 - 20/1044), 3278/1254)


shifts = np.array([	[-0.34, 0.56],
					[0.12, 0.6],
					[-0.35, 0.75],
					[-0.3, 0.5]
				])

angles = np.array([15.0, 10.0, 15.0, 20.0])

nucleus = ['AD', 'AM', 'AVd', 'AVv', 'IAD', 'MD', 'PV', 'sm']

swr_nuc = pd.DataFrame(index = swr_mod.index, columns = pd.MultiIndex.from_product([['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32'],nucleus,['mean', 'sem']]))

neurons = np.intersect1d(swr_mod.columns.values, mappings.index.values)

for m in ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']:
	subspace = mappings.loc[neurons][mappings.loc[neurons].index.str.contains(m)]
	groups = subspace.groupby(['nucleus']).groups
	for n in nucleus:
		if len(groups[n])>3:				
			swr_nuc[(m,n)] = pd.concat([swr_mod[groups[n]].mean(1),swr_mod[groups[n]].sem(1)], axis = 1).rename(columns={0:'mean',1:'sem'})

swr_all = pd.DataFrame(index = swr_mod.index, columns = nucleus)
mappings = mappings.loc[neurons]
for n in nucleus:
	swr_all[n] = swr_mod[mappings[mappings['nucleus'] == n].index.values].mean(1)

xs = [-0.4,0.35]
ys = [-0.3,0.25]

lbs = ['A', 'B', 'C', 'D']

colors = ['blue', 'red', 'green', '#ff934f']
 
toplot = pd.DataFrame(index = ['Mouse12','Mouse17','Mouse20','Mouse32'], columns = pd.MultiIndex.from_product([range(3),['start','end']]))
for i,j,k in zip(range(3),[-80,120,250],[0,200,330]): 
	toplot.loc['Mouse17',(i,'start')] = j
	toplot.loc['Mouse17',(i,'end')] = k	


alljpc = dict()
pos = [1,0,2,3]

i = 0
m = 'Mouse17'

data = cPickle.load(open("../../data/maps/"+m+".pickle", 'rb'))
theta 	= data['movies']['theta']
swr 	= data['movies']['swr']
total 	= data['total']
x 		= data['x']
y 		= data['y']
headdir = data['headdir']
jpc 	= data['jpc']
jpc 	= pd.DataFrame(index = times2, data = jpc)

toplot = pd.DataFrame(index = ['Mouse12','Mouse17','Mouse20','Mouse32'], columns = pd.MultiIndex.from_product([range(3),['start','end']]))
for i,j,k in zip(range(3),[-80,120,250],[0,200,330]): 
	toplot.loc['Mouse17',(i,'start')] = j
	toplot.loc['Mouse17',(i,'end')] = k	

#####################################################################
# E MAPS MOUSE 17
#####################################################################
gsm = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = gs[:,0])
bound = cPickle.load(open("../../figures/figures_articles/figure1/rotated_images_"+m+".pickle", 'rb'))['bound']
newswr = []
for j in range(3):
	tmp = swr[:,:,np.where(times2 == toplot.loc[m,(j,'start')])[0][0]:np.where(times2 == toplot.loc[m,(j,'end')])[0][0]].mean(-1)
	xnew, ynew, frame = interpolate(tmp.copy(), x, y, 0.01)
	frame = gaussian_filter(frame, (10, 10))
	newswr.append(frame)
newswr = np.array(newswr)
newswr = newswr - newswr.min()
newswr = newswr / newswr.max()	
newswr = softmax(newswr, 10, 0.5, 0.0)
for j in range(3):
	subplot(gsm[j,0])
	# if j == 0:
	# 	text(-0.1, 1.05, "e", transform = gca().transAxes, fontsize = 10, fontweight='bold')
	if j == 1: 
		text(-1.0, 1.0, "SWR modulation \n (Mouse 1)", transform = gca().transAxes, fontsize = 10)
		# title("SWR modulation (Mouse 1)", pad = -0.05)
	noaxis(gca())
	image = newswr[j]
	h, w = image.shape
	rotated_image = np.zeros((h*3, w*3))*np.nan
	rotated_image[h:h*2,w:w*2] = image.copy() + 1.0	
	rotated_image = rotateImage(rotated_image, -angles[pos[i]])
	rotated_image[rotated_image == 0.0] = np.nan
	rotated_image -= 1.0
	tocrop = np.where(~np.isnan(rotated_image))
	rotated_image = rotated_image[tocrop[0].min()-1:tocrop[0].max()+1,tocrop[1].min()-1:tocrop[1].max()+1]			
	imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bilinear', aspect = 'equal')
	im = imshow(rotated_image, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'bwr')	
	xlim(np.minimum(bound_adrien[0],bound[0]),np.maximum(bound_adrien[1],bound[1]))
	ylim(np.minimum(bound_adrien[2],bound[2]),np.maximum(bound_adrien[3],bound[3]))
	xlabel(str(toplot.loc[m,(j,'start')])+r"ms $\rightarrow$ "+str(toplot.loc[m,(j,'end')])+"ms")

	#colorbar	
	cax = inset_axes(gca(), "4%", "20%",
	                   bbox_to_anchor=(0.75, 0.0, 1, 1),
	                   bbox_transform=gca().transAxes, 
	                   loc = 'lower left')
	cb = colorbar(im, cax = cax, orientation = 'vertical', ticks = [0.25, 0.75])
	# cb.set_label('Burstiness', labelpad = -4)
	# cb.ax.xaxis.set_tick_params(pad = 1)
	# cax.set_title("Cluster 2", fontsize = 6, pad = 2.5)








savefig(r"../../../Dropbox (Peyrache Lab)/Talks/fig_talk_20.png", dpi = 300, facecolor = 'white')
