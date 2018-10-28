

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
import neuroseries as nts


###############################################################################################################
# TO LOAD
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
spind_mod, spind_ses 	= loadSpindMod('/mnt/DataGuillaume/MergedData/SPINDLE_mod.pickle', datasets, return_index=True)
spike_spindle_phase 	= cPickle.load(open('/mnt/DataGuillaume/MergedData/SPIKE_SPINDLE_PHASE.pickle', 'rb'))		
spike_theta_phase 		= cPickle.load(open('/mnt/DataGuillaume/MergedData/SPIKE_THETA_PHASE.pickle', 'rb'))		

nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

theta 				= pd.DataFrame(	index = theta_ses['rem'], 
									columns = ['phase', 'pvalue', 'kappa'],
									data = theta_mod['rem'])

# filtering swr_mod
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
# Cut swr_mod from -500 to 500
swr = swr.loc[-500:500]
# CHECK FOR NAN
tmp1 			= swr.columns[swr.isnull().any()].values
tmp2 			= theta.index[theta.isnull().any(1)].values
# CHECK P-VALUE 
tmp3	 		= theta.index[(theta['pvalue'] > 1).values].values
tmp 			= np.unique(np.concatenate([tmp1,tmp2,tmp3]))
# copy and delete 
if len(tmp):
	swr_modth 	= swr.drop(tmp, axis = 1)
	theta_modth = theta.drop(tmp, axis = 0)

swr_modth_copy 	= swr_modth.copy()
neuron_index = swr_modth.columns
times = swr_modth.loc[-500:500].index.values




m = 'Mouse17'
data 	= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
theta 	= data['movies']['theta']
swr 	= data['movies']['swr']
total 	= data['total']
x 		= data['x']
y 		= data['y']
headdir = data['headdir']
jpc 	= data['jpc']

interval_to_cut = {	'Mouse12':[88,120],
					'Mouse17':[84,123]}
					# 'Mouse20':[92,131],
					# 'Mouse32':[80,125]}

exemples = {'ldvl':['Mouse12-120807_7', 'Mouse12-120807_8', 'Mouse12-120807_9',
       'Mouse12-120807_10', 'Mouse12-120807_11', 'Mouse12-120807_12',
       'Mouse12-120807_13'],			
			're':['Mouse12-120819_3', 'Mouse12-120819_5'],
			'av':['Mouse12-120814_20', 'Mouse12-120814_22', 'Mouse12-120814_23',
       'Mouse12-120814_24']
			}

depths = [0.07, 0.49, 1.61]
shanks = [1.2, 1.0 ,0.8]

def interpolate(z, x, y, inter, bbox = None):	
	xnew = np.arange(x.min(), x.max()+inter, inter)
	ynew = np.arange(y.min(), y.max()+inter, inter)
	if bbox == None:
		f = scipy.interpolate.RectBivariateSpline(y, x, z)
	else:
		f = scipy.interpolate.RectBivariateSpline(y, x, z, bbox = bbox)
	znew = f(ynew, xnew)
	return (xnew, ynew, znew)

def filter_(z, n):
	from scipy.ndimage import gaussian_filter	
	return gaussian_filter(z, n)

def softmax(x, b1 = 10.0, b2 = 0.5, lb = 0.2):
	x -= x.min()
	x /= x.max()
	return (1.0/(1.0+np.exp(-(x-b2)*b1)) + lb)/(1.0+lb)

def get_rgb(mapH, mapS, mapV, bound):
	from matplotlib.colors import hsv_to_rgb	
	"""
		1. convert mapH to x between -1 and 1
		2. get y value between 0 and 1 -> mapV
		3. rescale mapH between 0 and 0.6
		4. normalize mapS

	"""		
	# x = mapH.copy() * 2.0
	# x = x - 1.0
	# y = 1.0 - 0.4*x**6.0
	# mapV = y.copy()
	H 	= (1-mapH)*bound
	S 	= mapS	
	V 	= mapV
	HSV = np.dstack((H,S,V))	
	RGB = hsv_to_rgb(HSV)	
	return RGB

nbins 					= 200
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
times2 					= times
space 					= 0.01

from scipy.ndimage import gaussian_filter	
# swr 					= gaussian_filter(swr, (1,0.2,0.2))
swr_copy = swr.copy()

times = times[interval_to_cut[m][0]:interval_to_cut[m][1]]
swr = swr[:,:,interval_to_cut[m][0]:interval_to_cut[m][1]]

##############################################################################################################
# TOTAL NEURON
##############################################################################################################
total = total / total.max()
xnew, ynew, xytotal = interpolate(total.copy(), x, y, space)
filtotal = gaussian_filter(xytotal, (10, 10))
newtotal = softmax(filtotal, 15.0, 0.25)


# newtotal[newtotal > 0.9] = np.NaN

##############################################################################################################
# HEAD DIRECTION
##############################################################################################################
xnew, ynew, newheaddir = interpolate(headdir.copy(), x, y, space)
newheaddir[newheaddir < np.percentile(newheaddir, 95)] = 0.0
##############################################################################################################
# THALAMUS LINES
##############################################################################################################
thl_lines = scipy.ndimage.imread("../figures/thalamus_lines_4.png").sum(2)
xlines, ylines, thl_lines = interpolate(thl_lines, 	np.linspace(x.min(), x.max(), thl_lines.shape[1]),
 													np.linspace(y.min(), y.max(), thl_lines.shape[0]), space*0.1)

thl_lines[thl_lines < 200] = np.NaN
thl_lines[thl_lines > 200] = 1.0
# thl_lines[thl_lines < 230] = np.NaN
# thl_lines[thl_lines > 230] = 1.0

##############################################################################################################
# SWR
##############################################################################################################
newswr = []
for t in range(len(times)):	
	xnew, ynew, frame = interpolate(swr[:,:,t].copy(), x, y, space)
	frame = gaussian_filter(frame, (1, 1))
	newswr.append(frame)
newswr = np.array(newswr)

newswr = newswr - newswr.min()
newswr = newswr / newswr.max()
newswr = softmax(newswr, 10, 0.5, 0.0)

##############################################################################################################
# THETA
##############################################################################################################
phase = np.linspace(0, 2*np.pi, theta.shape[-1])
newtheta = []
for i in range(len(phase)):
	xnew, ynew, frame = interpolate(theta[:,:,i].copy(), x, y, space)
	newtheta.append(frame)
newtheta = np.array(newtheta)


###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.2            # height in inches
	fig_size = [fig_width*0.9,fig_height]
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
import matplotlib.patches as patches


mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 9,               # LaTeX default is 10pt font.
	"font.size": 8,
	"legend.fontsize": 8,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 5,
	"ytick.labelsize": 5,
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
n = 4
# to_plot = [0, 11, 16]
if m == 'Mouse12':
	to_plot = [0, 11, 22]
elif m == 'Mouse17':
	to_plot = [4, 18, 26]
##############################################################
# ORBIT
##############################################################
gs1 = gridspec.GridSpec(2,3)
gs1.update(hspace = 0.4, bottom = 0.01, top = 0.95, right = 0.98, left = 0.04)
ax = subplot(gs1[0, 0])
# axis('off')
start, stop = (10, -65)
simpleaxis(ax)
plot(jpc[start,0], jpc[start,1], 'o', markersize = 3, color = '#5c7d6f')
plot(jpc[start:stop,0], jpc[start:stop,1], linewidth = 0.8, color = '#5c7d6f')
arrow(jpc[stop-2,0],jpc[stop-2,1],jpc[stop-1,0]-jpc[stop-2,0],jpc[stop-1,1]-jpc[stop-2,1], color = '#5c7d6f', head_width = 0.06)
ax.spines['left'].set_bounds(np.min(jpc[:,1]), np.min(jpc[:,1]+0.1))
ax.spines['bottom'].set_bounds(np.min(jpc[:,0]), np.min(jpc[:,0]+0.1))
xticks([], [])
yticks([], [])
ax.xaxis.set_label_coords(0.25, -0.02)
ax.yaxis.set_label_coords(-0.02, 0.15)
ylabel('jPC2')
xlabel('jPC1')
xlim(-0.4,0.4)
ylim(-0.4,0.4)
specialposition = {	to_plot[0]:[-0.15, 0.05],
					to_plot[1]:[-0.10, -0.05],
					to_plot[2]:[0.0, 0.05]}
					#to_plot[3]:[0.01, 0.01]}
for i in to_plot:
	idx = np.where(times[i] == times2)[0][0]
	plot(jpc[idx,0], jpc[idx,1], 'o', markersize = 4, color = 'green')	
	if i == 11:
		text(jpc[idx,0]+specialposition[i][0], jpc[idx,1]+specialposition[i][1], "0 ms")
	else :
		text(jpc[idx,0]+specialposition[i][0], jpc[idx,1]+specialposition[i][1], str(int(times[i]))+" ms")
title("SWR projection \n (one mouse)", y = 0.91)


##############################################################
# MAP
##############################################################

# for i,j in zip(range(4), ((0,2), (1,0), (1,1), (1,2))):		
for i,j in zip(range(3), ((1,0), (1,1), (1,2))):		
		ax = subplot(gs1[j[0], j[1]])
		frame = newswr[to_plot[i]]		
		rgbframe = get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65)		
		# rgbframe = get_rgb(frame, )	
		imshow(rgbframe, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
		# imshow(newtotal,  extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'gist_gray', alpha = 0.64)		
		if i == 1: 
			title("T = 0 ms")
		else:
			title("T = "+str(int(times[to_plot[i]]))+" ms")
		contour(newheaddir, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'winter')
		# imshow(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]))	

		# xticks([], [])
		# yticks([], [])

##############################################################
# SWR
##############################################################
if m == 'Mouse12':
	gs00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs1[0,2])

	titles = ['LVDL', 'AV', 'Re']
	axswr = {}
	for i,z in zip(range(3),['ldvl', 'av', 're']):
		ax1 = subplot(gs00[i,0])
		axswr[z] = ax1
		simpleaxis(ax1)
		if i in [0,1]:
			ax1.set_xticks([])
			ax1.spines['bottom'].set_visible(False)
		# bounds = [-200, 200]
		mean = swr_modth[exemples[z]].mean(1)
		sem = swr_modth[exemples[z]].sem(1)
		times = mean.index.values
		# plot(swr_modth[exemples[z]], linestyle = '--', color = 'red', linewidth = 0.9)
		plot(times, mean, color = 'black', linewidth = 2)
		fill_between(times, mean-sem, mean+sem, alpha = 0.4, color = 'grey')
		title(titles[i], loc = 'right')
		ylim(-4,4)
		if i == 2:
			xlabel("Time from SPWR (ms)", fontsize = 8)

		# axvline(-60, color = 'grey', linewidth = 0.6)
		# axvline(-5, color = 'grey', linewidth = 0.6)
		# axvline(20, color = 'grey', linewidth = 0.6)
		# if i == 1:
		# 	ax1.set_xticks([-60, -5, 20])
		# fill_between(mean.index.values, mean - sem, mean + sem, alpha = 0.5)
		# ylim(-1.5, 1.5)





##############################################################
# THALAMUS
##############################################################
ax = subplot(gs1[0, 1])
ax.imshow(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]))	
ax.contour(newheaddir, origin = 'upper', aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'winter')
# ax.set_xticks(x)
# ax.set_xticklabels(np.arange(1,9))
ax.set_xlabel("Shanks")
ax.set_ylabel("Depth per session")
ax.set_yticks(y)
ax.set_title("Thalamus Map")
ax.text(0.82, 0.21, '$\mathbf{AD}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(0.67, 0.8, 	'$\mathbf{IAD}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold', rotation = 70)
ax.text(0.8, 1.05, 	'$\mathbf{AM}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(1.1, 0.4, 	'$\mathbf{AV}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(1.24, 0.07, '$\mathbf{LDVL}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(0.55, 0.21, 	'$\mathbf{sm}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(0.45, 0.49,	'$\mathbf{MD}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(1.22, 1.13,	'$\mathbf{VA}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')	
ax.text(0.28, 0.65,	'$\mathbf{PVA}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')	
ax.text(0.7, 1.53,	'$\mathbf{Re}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')	
ax.text(0.5, 0.77,	'$\mathbf{PT}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')	

scatter(shanks[0], depths[0], 7, color = 'red', zorder = 2)
scatter(shanks[1], depths[1], 7, color = 'red')
scatter(shanks[2], depths[2], 7, color = 'red')

##############################################################
# ARROWS
##############################################################
if m == 'Mouse12':
	ax1tr = ax.transData
	axad = axswr['ldvl'].transData
	axam = axswr['re'].transData
	axav = axswr['av'].transData
	figtr = fig.transFigure.inverted()
	ptB = figtr.transform(ax1tr.transform((shanks[0],depths[0])))
	ptE = figtr.transform(axad.transform((-700,0)))
	style="simple,head_width=2,head_length=3"
	kw = dict(arrowstyle=style, color="k")
	arrow = matplotlib.patches.FancyArrowPatch(
	    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
	    fc = "None", connectionstyle="arc3,rad=-0.1", alpha = 0.5,
	    mutation_scale = 3., **kw)
	fig.patches.append(arrow)

	ptB = figtr.transform(ax1tr.transform((shanks[2],depths[2])))
	ptE = figtr.transform(axam.transform((-700,0)))
	style="<->,head_width=2,head_length=3"
	arrow = matplotlib.patches.FancyArrowPatch(
	    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
	    fc = "None", connectionstyle="arc3,rad=-0.1", alpha = 0.5,
	    mutation_scale = 3., **kw)
	fig.patches.append(arrow)

	ptB = figtr.transform(ax1tr.transform((shanks[1],depths[1])))
	ptE = figtr.transform(axav.transform((-700,0)))
	style="<->,head_width=2,head_length=3"
	arrow = matplotlib.patches.FancyArrowPatch(
	    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
	    fc = "None", connectionstyle="arc3,rad=0.0", alpha = 0.5,
	    mutation_scale = 3., **kw)
	fig.patches.append(arrow)








cbaxes = fig.add_axes([0.34, 0.41, 0.01, 0.06])
cmap = cm.jet
norm = matplotlib.colors.Normalize(swr.min(), swr.max())
cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
cbaxes.axes.set_xlabel('SWR \n mod')

cbaxes = fig.add_axes([0.34, 0.25, 0.01, 0.06])
cmap = cm.gist_gray
norm = matplotlib.colors.Normalize(0, 1)
cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
cbaxes.axes.set_xlabel('Neurons \n density')

# cbaxes = fig.add_axes([0.34, 0.1, 0.01, 0.06])
# cmap = cm.winter
# norm = matplotlib.colors.Normalize(0, 1)
# cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
# cbaxes.axes.set_xlabel('HD \n neurons')



savefig("../figures/figures_articles/figart_4"+m+".pdf", dpi = 900, facecolor = 'white')
os.system("evince ../figures/figures_articles/figart_4"+m+".pdf &")



sys.exit()



newswr = []
for t in range(len(times)):	
	xnew, ynew, frame = interpolate(swr_copy[:,:,t].copy(), x, y, space)
	frame = gaussian_filter(frame, (10, 10))
	newswr.append(frame)
newswr = np.array(newswr)

newswr = gaussian_filter(newswr, (0,0.2,0.2))
newswr = newswr - newswr.min()
newswr = newswr / newswr.max()




from matplotlib import animation, rc
from IPython.display import HTML, Image

rc('animation', html='html5')
fig, axes = plt.subplots(1,1)
start = 70


frame = newswr[t]		
rgbframe = get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65)		
images = [axes.imshow(rgbframe, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))]
axes.imshow(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]))


def init():
	images[0].set_data(rgbframe)	
	return images
		
def animate(t):
	frame = newswr[t]
	rgbframe = get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65)
	images[0].set_data(rgbframe)	
	images[0].axes.set_title("time = "+str(times[t]))
	return images
	
anim = animation.FuncAnimation(fig, animate, init_func=init,
						   frames=range(start,132), interval=10, blit=False, repeat_delay = 1000)

anim.save('../figures/swr_mod_'+m+'.gif', writer='imagemagick', fps=15)
# show()
# sys.exit()