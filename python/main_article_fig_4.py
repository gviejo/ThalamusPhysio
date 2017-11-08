

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
m = 'Mouse12'
data 	= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
theta 	= data['movies']['theta']
swr 	= data['movies']['swr']
total 	= data['total']
x 		= data['x']
y 		= data['y']
headdir = data['headdir']
jpc 	= data['jpc']

interval_to_cut = {	'Mouse12':[88,120]}
					# 'Mouse17':[84,123],
					# 'Mouse20':[92,131],
					# 'Mouse32':[80,125]}


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

def softmax(x, b1 = 10.0, b2 = 0.5):
	x -= x.min()
	x /= x.max()
	return 1.0/(1.0+np.exp(-(x-b2)*b1))

def get_rgb(mapH, mapV, mapS, bound):
	from matplotlib.colors import hsv_to_rgb	
	"""
		1. convert mapH to x between -1 and 1
		2. get y value between 0 and 1 -> mapV
		3. rescale mapH between 0 and 0.6
		4. normalize mapS

	"""	
	mapH -= mapH.min()
	mapH /= mapH.max()
	mapS -= mapS.min()
	mapS /= mapS.max()
	x = mapH.copy() * 2.0
	x = x - 1.0
	y = 1.0 - 0.4*x**6.0
	mapV = y
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
swr 					= gaussian_filter(swr, (1,1,1))

# sys.exit()
times = times[interval_to_cut[m][0]:interval_to_cut[m][1]]
swr = swr[:,:,interval_to_cut[m][0]:interval_to_cut[m][1]]

##############################################################################################################
# TOTAL NEURON
##############################################################################################################
total = total / total.max()
xnew, ynew, newtotal = interpolate(total.copy(), x, y, space)
newtotal = softmax(newtotal, 10.0, 0.3)
# newtotal[newtotal > 0.9] = np.NaN

##############################################################################################################
# HEAD DIRECTION
##############################################################################################################
xnew, ynew, newheaddir = interpolate(headdir.copy(), x, y, space)
newheaddir[newheaddir < np.percentile(newheaddir, 95)] = 0.0
##############################################################################################################
# THALAMUS LINES
##############################################################################################################
thl_lines = scipy.ndimage.imread("../figures/thalamus_lines_2.png").sum(2)
xlines, ylines, thl_lines = interpolate(thl_lines, 	np.linspace(x.min(), x.max(), thl_lines.shape[1]),
 													np.linspace(y.min(), y.max(), thl_lines.shape[0]), space*0.1)

thl_lines[thl_lines > 300] = np.NaN
thl_lines[thl_lines > 0] = 1.0


##############################################################################################################
# SWR
##############################################################################################################
newswr = []
for t in range(len(times)):	
	xnew, ynew, frame = interpolate(swr[:,:,t].copy(), x, y, space)
	newswr.append(frame)
newswr = np.array(newswr)

newswr = newswr - newswr.min()
newswr = newswr / newswr.max()
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
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
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


fig = figure(figsize = figsize(1))
n = 4
to_plot = [0, 11, 16, 22]
##############################################################
# SWR 
##############################################################
gs1 = gridspec.GridSpec(2,3)
gs1.update(hspace = 0.3, bottom = 0.01, top = 0.95, right = 1, left = 0.04)
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
					to_plot[2]:[0.05, -0.05],
					to_plot[3]:[0.01, 0.01]}
for i in to_plot:
	idx = np.where(times[i] == times2)[0][0]
	plot(jpc[idx,0], jpc[idx,1], 'o', markersize = 4, color = 'red')	
	text(jpc[idx,0]+specialposition[i][0], jpc[idx,1]+specialposition[i][1], str(int(times[i]))+" ms")
title("SWR projection \n (one mouse)", y = 0.91)


for i,j in zip(range(4), ((0,2), (1,0), (1,1), (1,2))):		
		ax = subplot(gs1[j[0], j[1]])
		frame = newswr[to_plot[i]]
		# rgbframe = get_rgb(frame, )	
		imshow(frame, vmin = 0.0, vmax = 1.0, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'jet')
		imshow(newtotal,  extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'gist_gray', alpha = 0.65)		
		title("T = "+str(int(times[to_plot[i]]))+" ms")
		contour(newheaddir, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'winter')
		imshow(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]))	
		# xticks(xnew[np.arange(0, frame.shape[1],20)])
		# yticks(ynew[np.arange(0, frame.shape[0],20)])
		xticks([], [])
		yticks([], [])

ax = subplot(gs1[0, 1])
# ax.imshow(thl_lines, vmin = 0.0, vmax = 1.0, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))

ax.imshow(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]))	
ax.contour(newheaddir, origin = 'upper', aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'winter')
# ax.set_xticks(x)
ax.set_xticklabels(np.arange(1,9))
ax.set_xlabel("Shanks")
ax.set_ylabel("Depth per sessions")
ax.set_yticks(y)
ax.set_title("Thalamus Map")
ax.text(0.82, 0.21, '$\mathbf{AD}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(0.67, 0.8, 	'$\mathbf{IAD}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(0.8, 1.3, 	'$\mathbf{AM}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(1.1, 0.4, 	'$\mathbf{AV}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(1.24, 0.07, '$\mathbf{LDVL}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(0.5, 0.21, 	'$\mathbf{sm}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(0.3, 0.63, 	'$\mathbf{MD}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')
ax.text(1.22, 1.13,	'$\mathbf{VA}$'	, horizontalalignment = 'center', verticalalignment = 'center', fontweight='bold')	


cbaxes = fig.add_axes([0.68, 0.65, 0.01, 0.06])
cmap = cm.jet
norm = matplotlib.colors.Normalize(swr.min(), swr.max())
cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
cbaxes.axes.set_xlabel('SWR \n mod')

cbaxes = fig.add_axes([0.68, 0.5, 0.01, 0.06])
cmap = cm.gist_gray
norm = matplotlib.colors.Normalize(0, 1)
cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
cbaxes.axes.set_xlabel('Neurons \n density')

cbaxes = fig.add_axes([0.68, 0.35, 0.01, 0.06])
cmap = cm.winter
norm = matplotlib.colors.Normalize(0, 1)
cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
cbaxes.axes.set_xlabel('HD \n neurons')


savefig("../figures/figures_articles/figart_4.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../figures/figures_articles/figart_4.pdf &")















sys.exit()

from matplotlib import animation, rc
from IPython.display import HTML, Image

rc('animation', html='html5')
fig, axes = plt.subplots(1,1)
# images = [axes.imshow(get_rgb(filmov['swr'][0].copy(), np.ones_like(total), total, 0.65), vmin = 0.0, vmax = 1.0, aspect = 'equal', origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))]
images = [axes.imshow(newswr[0], vmin = newswr.min(), vmax = newswr.max(), aspect = 'equal', origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))]
# images = [axes.imshow(filmov['swr'][0], aspect = 'equal', origin = 'upper', cmap = 'jet', vmin = 0.0, vmax = 1.0, extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))]
# axes.contour(head, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'winter')
# axes.contour(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]), colors = 'white')	
def init():
	images[0].set_data(newswr[0])
	# images[0].set_data(filmov['swr'][0])
	return images
		
def animate(t):
	images[0].set_data(newswr[t])
	# images[0].set_data(filmov['swr'][t])		
	return images
	
anim = animation.FuncAnimation(fig, animate, init_func=init,
						   frames=range(len(times)), interval=100, blit=False, repeat_delay = 1000)

anim.save('../figures/swr_mod_'+m+'.gif', writer='imagemagick', fps=60)
show()
sys.exit()