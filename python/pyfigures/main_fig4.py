

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
data = cPickle.load(open('../../data/to_plot.pickle', 'rb'))
allzth 			= 	data['swr_modth'		]		
eigen 			= 	data['eigen'		]		
times 			= 	data['times' 		]		
allthetamodth 	= 	data['theta_modth']		
phi 			= 	data['phi' 			]		
zpca 			= 	data['zpca'			]		
phi2			= 	data['phi2' 		]	 	
jX				= 	data['rX'			]
jscore			= 	data['jscore'		]
force 			= 	data['force'		] # theta modulation
variance 		= 	data['variance'		] # ripple modulation
scoretheta		=	data['scoretheta'	]	
scorerip		=	data['scorerip'		]	
phitheta		=	data['phitheta'		]	
phirip			=	data['phirip'		]	
indextheta		=	data['indextheta'	]	
indexrip		=	data['indexrip'		]	
jpca_theta		=	data['jpca_theta'	] 	
jpca_rip		=	data['jpca_rip'		] 	
 	


index = np.argsort(allthetamodth[:,2])
tmp = allthetamodth[index,0]
phasetheta1 = [	tmp[indextheta[0]],
				tmp[indextheta[1]]
				]
variance = np.var(allzth, 1)
index = np.argsort(variance)
tmp = allthetamodth[index,0]
phasetheta2 = [	tmp[indexrip[0]],
				tmp[indexrip[1]]
				]				

###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687 *0.8                        # Get this from LaTeX using \the\textwidth
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


fig = figure(figsize = figsize(1))
# outer = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)#, height_ratios = [1,3])#, width_ratios = [1.6,0.7]) 
gs = gridspec.GridSpec(2,2, wspace = 0.4, hspace = 0.4)

# ax = subplot(gs[0, 0])
# simpleaxis(ax)
# plot(jpca_theta[0,0], jpca_theta[0,1], 'o', markersize = 3, color = '#5c7d6f')
# plot(jpca_theta[:,0], jpca_theta[:,1], linewidth = 0.8, color = '#5c7d6f')
# arrow(jpca_theta[-10,0],jpca_theta[-10,1],jpca_theta[-1,0]-jpca_theta[-10,0],jpca_theta[-1,1]-jpca_theta[-10,1], color = '#5c7d6f', head_width = 0.01)
# ax.spines['left'].set_bounds(np.min(jpca_theta[:,1]), np.min(jpca_theta[:,1]+0.1))
# ax.spines['bottom'].set_bounds(np.min(jpca_theta[:,0]), np.min(jpca_theta[:,0]+0.1))
# xticks([], [])
# yticks([], [])
# ax.xaxis.set_label_coords(0.15, -0.02)
# ax.yaxis.set_label_coords(-0.02, 0.15)
# ylabel('jPC2')
# xlabel('jPC1')


# ax = subplot(gs[0,1])
# axis('off')
# axhline(0, xmin = 0.25, xmax = 0.75, color = '#386150', linewidth = 0.7)
# axvline(0, ymin = 0.25, ymax = 0.75, color = '#58b09c', linewidth = 0.7)
# text(19.0, -2.0, 'jPC1', fontsize = 5)
# text(0.7, 20.0, 'jPC2', fontsize = 5)
# xlim(-50, 50)
# ylim(-45, 45)
# x, y = (np.cos(phi), np.sin(phi2))
# r = 40+2*np.random.rand(len(x))
# # scatter(x*r, y*r, s = 3, c = phi, cmap = cm.get_cmap('hsv'), zorder = 2)
# scatter(x*r, y*r, s = 2, c = allzth[:,100], cmap = cm.get_cmap('viridis'), zorder = 2)
# scatter(jscore[:,0], jscore[:,1], s = 2, c = allzth[:,100], cmap = cm.get_cmap('viridis'), zorder = 2)
# text(0, 32, '$\mathbf{SWR\ jPCA\ phase}$',horizontalalignment='center')
# lower_left = np.argmin(jscore[:,0])
# arrow(jscore[lower_left,0],jscore[lower_left,1], x[lower_left]*r[lower_left]-jscore[lower_left,0]+3, y[lower_left]*r[lower_left]-jscore[lower_left,1]+3,
# 		head_width = 0.8,
# 		linewidth = 0.5
# 		)
# text(-35.,-7, 'arctan2', rotation = 13.0)
# cbaxes = fig.add_axes([0.25, 0.45, 0.01, 0.04])
# cmap = cm.viridis
# norm = matplotlib.colors.Normalize(allzth[:,100].min(), allzth[:,100].max())
# cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
# cbaxes.axes.set_xlabel('SWR \n modulation')
r1, p1 = corr_circular_(phasetheta2[0], phirip[0])
print(r1, p1)
r2, p2 = corr_circular_(phasetheta2[1], phirip[1])
print(r2, p2)


ax = subplot(gs[0,0])
simpleaxis(ax)
theta_mod_toplot = phasetheta2[0]#,dist_cp>0.02]
phi_toplot = phirip[0]
x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
scatter(x, y, s = 0.8, c = 'black', cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.5)
xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad)', labelpad = 1.2)
ylabel('SWR PCA phase (rad)')
title('Low SWR modulation')

ax = subplot(gs[0,1])
H, xedges, yedges = np.histogram2d(y, x, 50)
H = gaussFilt(H, (3,3))
H = H - H.min()
H = H / H.max()
levels = np.linspace(H.min(), H.max(), 50)
axp = ax.contourf(H, V = levels, cmap = 'binary')
xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad)')
ylabel('SWR PCA phase (rad)')
tik = np.array([0, np.pi, 2*np.pi, 3*np.pi])
xtik = [np.argmin(np.abs(i-xedges)) for i in tik]
ytik = [np.argmin(np.abs(i-yedges)) for i in tik]
xticks(xtik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
yticks(ytik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
title("Density", fontsize = 5)
# cbaxes = fig.add_axes([0.63, 0.11, 0.01, 0.04])
# cb = colorbar(axp, cax = cbaxes, ticks = [0, 1])
# cbaxes.yaxis.set_ticks_position('left')

ax = subplot(gs[1,0])
simpleaxis(ax)
theta_mod_toplot = phasetheta2[1]#,dist_cp>0.02]
phi_toplot = phirip[1]
x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
scatter(x, y, s = 0.8, c = 'black', cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.5)
xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad)', labelpad = 1.2)
ylabel('SWR PCA phase (rad)')
title('High SWR modulation')

ax = subplot(gs[1,1])
H, xedges, yedges = np.histogram2d(y, x, 50)
H = gaussFilt(H, (3,3))
H = H - H.min()
H = H / H.max()
levels = np.linspace(H.min(), H.max(), 50)
axp = ax.contourf(H, V = levels, cmap = 'binary')
xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad)')
ylabel('SWR PCA phase (rad)')
tik = np.array([0, np.pi, 2*np.pi, 3*np.pi])
xtik = [np.argmin(np.abs(i-xedges)) for i in tik]
ytik = [np.argmin(np.abs(i-yedges)) for i in tik]
xticks(xtik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
yticks(ytik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
title("Density", fontsize = 5)
# cbaxes = fig.add_axes([0.63, 0.11, 0.01, 0.04])
# cb = colorbar(axp, cax = cbaxes, ticks = [0, 1])
# cbaxes.yaxis.set_ticks_position('left')

# CORRELATION COEFFICIENT


savefig("../../figures/fig4.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig4.pdf &")

