

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

def softmax(x, b1 = 10.0, b2 = 0.5, lb = 0.2):
	x -= x.min()
	x /= x.max()
	return (1.0/(1.0+np.exp(-(x-b2)*b1)) + lb)/(1.0+lb)

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



nbins 					= 200
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
times2 					= swr_mod.index.values

carte38_mouse17_2 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

shifts = np.array([	[-0.34, 0.56],
					[0.12, 0.6],
					[-0.35, 0.75],
					[-0.3, 0.5]
				])

angles = np.array([15.0, 10.0, 15.0, 20.0])

nucleus = ['AD', 'AM', 'AVd', 'AVv', 'IAD', 'MD', 'PV', 'sm']

swr_nuc = pd.DataFrame(index = swr_mod.index, columns = pd.MultiIndex.from_product([['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32'],nucleus,['mean', 'sem']]), data = 0)
for m in ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']:
	subspace = pd.read_hdf("../../figures/figures_articles/figure1/subspace_"+m+".hdf5")
	groups = subspace.groupby(['nucleus']).groups
	for n in nucleus:				
		swr_nuc[(m,n)] = pd.concat([swr_mod[groups[n]].mean(1),swr_mod[groups[n]].sem(1)], axis = 1).rename(columns={0:'mean',1:'sem'})

		


###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*2.0            # height in inches
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
gs = gridspec.GridSpec(3,1, wspace = 0.4, hspace = 0.5)
# gs.update(hspace = 0.2)
xs = [-0.4,0.35]
ys = [-0.3,0.25]

lbs = ['A', 'B', 'C']

colors = ['blue', 'red', 'green', '#ff934f']
 
toplot = pd.DataFrame(index = ['Mouse12','Mouse17','Mouse20','Mouse32'], columns = pd.MultiIndex.from_product([range(3),['start','end']]))
for i,j,k in zip(range(3),[-80,40,230],[0,120,300]):
	toplot.loc['Mouse12',(i,'start')] = j
	toplot.loc['Mouse12',(i,'end')] = k
for i,j,k in zip(range(3),[-80,10,120],[0,90,200]): 
	toplot.loc['Mouse17',(i,'start')] = j
	toplot.loc['Mouse17',(i,'end')] = k	
for i,j,k in zip(range(3),[-150,-25,125],[-50,50,225]): 
	toplot.loc['Mouse20',(i,'start')] = j
	toplot.loc['Mouse20',(i,'end')] = k
for i,j,k in zip(range(3),[-200,-125,50],[-150,-50,150]): 
	toplot.loc['Mouse32',(i,'start')] = j
	toplot.loc['Mouse32',(i,'end')] = k


alljpc = dict()
pos = [0,2,3]

for i, m in enumerate(['Mouse12', 'Mouse20', 'Mouse32']):
	gsm = gridspec.GridSpecFromSubplotSpec(2,4, subplot_spec = gs[i,:], height_ratios=[1,2])
	########################################################################
	# 1. Orbit
	########################################################################
	subplot(gsm[0,0])
	# simpleaxis(gca())
	noaxis(gca())
	
	data = cPickle.load(open("../../data/maps/"+m+".pickle", 'rb'))
	theta 	= data['movies']['theta']
	swr 	= data['movies']['swr']
	total 	= data['total']
	x 		= data['x']
	y 		= data['y']
	headdir = data['headdir']
	jpc 	= data['jpc']
	jpc 	= pd.DataFrame(index = times2, data = jpc)

	alljpc[m] = jpc
	plot(jpc.iloc[0,0], jpc.iloc[0,1], 'o', markersize = 3, color = 'grey')
	plot(jpc[0], jpc[1], linewidth = 0.8, color = 'grey')
	arrow(jpc.iloc[-3,0],jpc.iloc[-3,1],jpc.iloc[-2,0]-jpc.iloc[-3,0],jpc.iloc[-2,1]-jpc.iloc[-3,1], color = 'grey', head_width = 0.06)
	for j in range(3): 
		plot(jpc.loc[toplot.loc[m,(j,'start')]:toplot.loc[m,(j,'end')],0], jpc.loc[toplot.loc[m,(j,'start')]:toplot.loc[m,(j,'end')],1], color = 'blue', alpha = 0.3)		


	gca().spines['left'].set_bounds(ys[0]+0.1,ys[0]+0.2)
	gca().spines['bottom'].set_bounds(ys[0]+0.1,ys[0]+0.2)
	xticks([], [])
	yticks([], [])	
	# gca().xaxis.set_label_coords(0.25, -0.02)
	# gca().yaxis.set_label_coords(-0.02, 0.15)
	# ylabel('jPC2')
	# xlabel('jPC1')
	xlim(xs[0], xs[1])
	ylim(ys[0], ys[1])
	text(-0.15, 1.17, lbs[i], transform = gca().transAxes, fontsize = 9)



	########################################################################
	# 2. SWR NUCLEUS
	########################################################################
	subplot(gsm[1,0])
	simpleaxis(gca())
	for n in nucleus:
		plot(swr_nuc[m][n]['mean'], '-', label = n)
		# fill_between(times2, swr_nuc[m][n]['mean'].values - swr_nuc[m][n]['sem'].values, swr_nuc[m][n]['mean'].values + swr_nuc[m][n]['sem'], facecolor = 'grey', edgecolor = 'grey', alpha = 0.7)	
	for j in range(3): 
		# axvline(toplot[m][j])
		axvspan(toplot.loc[m,(j,'start')], toplot.loc[m,(j,'end')], alpha = 0.5)
	ylabel("SWR")
	xlabel("Times (ms)")
	if i == 0:
		legend(frameon=False,loc = 'lower left', bbox_to_anchor=(1.1,-0.4),handlelength=1,ncol = len(nucleus)//2)

	########################################################################
	# 3. MAPS 
	########################################################################		
	bound = cPickle.load(open("../../figures/figures_articles/figure1/rotated_images_"+m+".pickle", 'rb'))['bound']
	newswr = []
	# for t in range(len(times2)):	
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
		subplot(gsm[:,j+1])
		if j == 0: title("Mouse "+str(i+1))
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
		imshow(rotated_image, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'jet')
		imshow(carte38_mouse17_2[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
		xlim(np.minimum(cut_bound_map[0],bound[0]),np.maximum(cut_bound_map[1],bound[1]))
		ylim(np.minimum(cut_bound_map[2],bound[2]),np.maximum(cut_bound_map[3],bound[3]))
		xlabel(str(toplot.loc[m,(j,'start')])+"ms -> "+str(toplot.loc[m,(j,'end')])+"ms")


savefig("../../figures/figures_articles/figart_supp_3.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_supp_3.pdf &")

