

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


fig = figure(figsize = figsize(1), tight_layout=True)
# outer = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)#, height_ratios = [1,3])#, width_ratios = [1.6,0.7]) 
gs = gridspec.GridSpec(2,3, height_ratios = [0.3,0.2], width_ratios = [0.4, 0.35, 0.9])

########################################################################
# A. SHarp waves ripples modulation
########################################################################
subplot(gs[0,0])
im = imshow(allzthsorted, aspect = 'auto', cmap = 'viridis')
xticks(np.arange(20,200,40), (times[np.arange(20,200,40)]).astype('int'))
yticks([0,700], ['1', '700'])
# cb = colorbar()
# cb.set_label("z", labelpad = -10, y = 1.08, rotation = 0)
ylabel("Thalamic\nneurons", labelpad = -9.0)
xlabel("Time from SWRs (ms)", labelpad = -0.05)
title("SWR modulation", fontsize = 8)
# text(-0.3, 1.03, "a", transform = gca().transAxes, fontsize = 10, fontweight='bold')

cax = inset_axes(gca(), "5%", "100%",
                   bbox_to_anchor=(1, -0.06, 1, 1),
                   bbox_transform=gca().transAxes, 
                   loc = 'lower left')
cax.set_title("z", fontsize = 7, pad = 2.5)
cb = colorbar(im, cax = cax, orientation = 'vertical', ticks = [-2, 0, 2])


########################################################################
# B. JPCA
########################################################################
ax = subplot(gs[1, 0])
simpleaxis(ax)	
plot(times, jX[:,0], color = 'black', label = 'jPC 1')
plot(times, jX[:,1], color = 'grey', label = 'jPC 2')
legend(frameon=False,loc = 'lower left', bbox_to_anchor=(0.9,0.6),handlelength=1)
ylabel('jPC', labelpad = 0.1)
xlabel('Time from SWRs (ms)', labelpad = -0.05)
xticks([-400,-200,0,200,400])
# title('jPCA', fontsize = 8, y = 1)
# text(0.15, 0.86, "jPCA", transform = gca().transAxes, fontsize = 10)
# text(-0.3, 1.05, "b", transform = gca().transAxes, fontsize = 10, fontweight='bold')

########################################################################
# C circle
########################################################################
ax = subplot(gs[0:2,1:])
ax.set_aspect("equal")
# text(-0.5, 1.00, "c", transform = gca().transAxes, fontsize = 10, fontweight='bold')
axis('off')
axhline(0, xmin = 0.25, xmax = 0.75, color = 'black', linewidth = 1)
axvline(0, ymin = 0.25, ymax = 0.75, color = 'grey', linewidth = 1)
xlim(-20, 20)
# ylim(-14, 16)
ylim(-18,22)
phase_circle = np.arange(0, 2*np.pi, 0.0001)
# x, y = (np.cos(phi2.values.flatten()), np.sin(phi2.values.flatten()))
x, y = (np.cos(phase_circle),np.sin(phase_circle))
r = 14
plot(x*r, y*r, '-', color = 'black', linewidth = 0.5)
r = r+1
text(-r, 0,'$\pi$', horizontalalignment='center', verticalalignment='center', 	 	fontsize = 7)
text(r, 0,'0', horizontalalignment='center', verticalalignment='center', 		 	fontsize = 7)
text(0, r,'$\pi/2$', horizontalalignment='center', verticalalignment='center',  	fontsize = 7)
text(0, -r,'$3\pi/2$', horizontalalignment='center', verticalalignment='center',  	fontsize = 7)
text(r-7, -2.5, 'jPC1', fontsize = 8)
text(0.7, r-6, 'jPC2', fontsize = 8)

#text(0.25,0.95,"Theta phase", fontsize =8,transform=ax.transAxes, color = 'red')

color_points = allthetamodth['phase'].copy()
color_points -= color_points.min()
color_points /= color_points.max()
# scatter(jscore.values[:,0], jscore.values[:,1], s = 3, c = color_points.values, cmap = cm.get_cmap('hsv'), zorder = 2, alpha = 1, linewidth = 0.0)
# scatter(jscore.values[:,0], jscore.values[:,1], s = 3, c = 'black', zorder = 2, alpha = 0.7, linewidth = 0.0)
scatter(jscore.values[:,0], jscore.values[:,1], s = 5, c = allzth.values[:,100], cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.7, linewidth = 0.0)

bb = ax.get_position().bounds
aiw = 0.2
ail = 0.2
position_axes = [
				[bb[0]+bb[2]*0.85,bb[1]+bb[3]*0.8],
				[bb[0]+bb[2]*-0.05,bb[1]+bb[3]*0.8],
				[bb[0]+bb[2]*-0.05,bb[1]+bb[3]*-0.1],
				[bb[0]+bb[2]*0.85,bb[1]+bb[3]*-0.1]]
r -= 1
best_neurons = []
lbs = ['a', 'b', 'c', 'd']
for i,j in zip(np.arange(0, 2*np.pi, np.pi/2),np.arange(4)):	
	quarter = phi2[np.logical_and(phi2 > i, phi2 < i+(np.pi/2)).values]
	tmp = jscore.loc[quarter.index.values]	
	if j == 2:		
		best_n = np.abs(allthetamodth.loc[tmp.index.values,'phase'] - (i+np.pi/8)).sort_values().index.values[9]
	elif j == 0:
		best_n = np.abs(allthetamodth.loc[tmp.index.values,'phase'] - (i+np.pi/8)).sort_values().index.values[4]
	elif j == 3:
		best_n = np.abs(allthetamodth.loc[tmp.index.values,'phase'] - (i+np.pi/8)).sort_values().index.values[1]
	else:
		best_n = np.abs(allthetamodth.loc[tmp.index.values,'phase'] - (i+np.pi/8)).astype('float').idxmin()
	best_neurons.append(best_n)
	# ai = axes([position_axes[j][0],position_axes[j][1], aiw, ail], projection = 'polar')
	# ai.get_xaxis().tick_bottom()
	# ai.get_yaxis().tick_left()
	# #ai.hist(spikes_theta_phase['rem'][best_n], 30, color = 'red', normed = True)
	# xticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	# yticks([])	
	# grid(linestyle = '--')
	# # xlabel(lbs[j])
	# # if j == 1:
	# # 	# ai.set_title("Theta phase", fontsize = 8, color = 'red')
	# # 	ai.text(1,1,"Theta phase", fontsize =8, color = 'red')		
	# ai.yaxis.grid(False)
	# ai.tick_params(axis='x', pad = -5)
	# ai.set_ylim(0,0.5)
	#ai.arrow(x = allthetamodth.loc[best_n,'phase'], y = 0, dx = 0, dy = ai.get_ylim()[1]*0.6, edgecolor = 'black', facecolor = 'green', lw = 1.0, head_width = 0.1, head_length = 0.02,zorder = 5)
	

	x = np.cos(quarter.loc[best_n,0])*r
	y = np.sin(quarter.loc[best_n,0])*r
	xx, yy = (jscore.loc[best_n,0],jscore.loc[best_n,1])
	ax.scatter(x, y, s = 20, c = 'red', cmap = cm.get_cmap('viridis'), alpha = 1)
	ax.scatter(xx, yy, s = 6, c = 'red', cmap = cm.get_cmap('viridis'), zorder = 2)
	
	ax.arrow(xx, yy, x - xx, y - yy,
		head_width = 0.8,
		linewidth = 0.6,
		length_includes_head = True,		
		color = 'grey'
		)
	

text(0.77,0.72, "a", fontsize =9,transform=ax.transAxes)
text(0.2,0.71,"b", fontsize =9,transform=ax.transAxes)
text(0.2,0.16,"c", fontsize =9,transform=ax.transAxes)
text(0.73,0.12,"d", fontsize =9,transform=ax.transAxes)



# text(0, 0, '$\mathbf{SWR\ jPCA\ phase}$',horizontalalignment='center')
text(0.04, 0.35, 'SWR\njPCA phase', fontsize = 8, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# lower_left = np.argmin(jscore.values[:,0])
# text(-35.,-7, 'arctan2', rotation = 13.0)
# cbaxes = fig.add_axes([0.25, 0.45, 0.01, 0.04])
# cmap = cm.viridis
# norm = matplotlib.colors.Normalize(allzth.values[:,100].min(), allzth.values[:,100].max())
# cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
# cbaxes.axes.set_xlabel('SWR \n modulation')






savefig(r"../../../Dropbox (Peyrache Lab)/Talks/fig_talk_17.png", dpi = 300, facecolor = 'white')
