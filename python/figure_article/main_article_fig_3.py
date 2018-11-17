

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

sys.exit()
###############################################################################################################
# PLOT11
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.4            # height in inches
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
# outer = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)#, height_ratios = [1,3])#, width_ratios = [1.6,0.7]) 
gs = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)

########################################################################
# A. SHarp waves ripples modulation
########################################################################

subplot(gs[0,0])
imshow(allzthsorted, aspect = 'auto', cmap = 'viridis')
xticks(np.arange(20,200,40), (times[np.arange(20,200,40)]).astype('int'))
yticks([0,700], ['0', '700'])
cb = colorbar()
cb.set_label("z", labelpad = -13, y = 1.08, rotation = 0)
ylabel("Thalamic neurons", labelpad = -5.0, y = 0.6)
xlabel("Time from SWR (ms)")
title("Sharp-waves Ripples \n modulation", fontsize = 7)
text(-0.15, 1.17, "A", transform = gca().transAxes, fontsize = 9)

########################################################################
# B. JPCA
########################################################################

ax = subplot(gs[0, 1])
simpleaxis(ax)	
plot(times, jX[:,0], color = '#386150')
plot(times, jX[:,1], color = '#58b09c')
ylabel('jPC')
xlabel('Time from SWR (ms)')
title('jPCA', fontsize = 7, y = 1)
text(-0.1, 1.14, "B", transform = gca().transAxes, fontsize = 9)

########################################################################
# C. ORBIT
########################################################################
ax = subplot(gs[0, 2])
# axis('off')
simpleaxis(ax)
plot(jX[0,0], jX[0,1], 'o', markersize = 3, color = '#5c7d6f')
plot(jX[:,0], jX[:,1], linewidth = 0.8, color = '#5c7d6f')
arrow(jX[-10,0],jX[-10,1],jX[-1,0]-jX[-10,0],jX[-1,1]-jX[-10,1], color = '#5c7d6f', head_width = 0.01)
plot(jX[np.where(times==-250),0], jX[np.where(times==-250),1], 'o', color = '#5c7d6f', markersize = 2)
plot(jX[np.where(times== 250),0], jX[np.where(times== 250),1], 'o', color = '#5c7d6f', markersize = 2)
plot(jX[np.where(times==   0),0], jX[np.where(times==   0),1], 'o', color = '#5c7d6f', markersize = 2)
annotate("-250", xy = (jX[np.where(times==-250),0], jX[np.where(times==-250),1]), xytext = (jX[np.where(times==-250),0]-0.06, jX[np.where(times==-250),1]), fontsize = 4)
annotate( "250", xy = (jX[np.where(times== 250),0], jX[np.where(times== 250),1]), xytext = (jX[np.where(times== 250),0]+0.03, jX[np.where(times== 250),1]), fontsize = 4)
annotate(   "0", xy = (jX[np.where(times==   0),0], jX[np.where(times==   0),1]), xytext = (jX[np.where(times==   0),0]-0.01, jX[np.where(times==   0),1]+0.01), fontsize = 4)
ax.spines['left'].set_bounds(np.min(jX[:,1]), np.min(jX[:,1]+0.1))
ax.spines['bottom'].set_bounds(np.min(jX[:,0]), np.min(jX[:,0]+0.1))
xticks([], [])
yticks([], [])
ax.xaxis.set_label_coords(0.15, -0.02)
ax.yaxis.set_label_coords(-0.02, 0.15)
ylabel('jPC2')
xlabel('jPC1')
text(-0.1, 1.14, "C", transform = gca().transAxes, fontsize = 9)

########################################################################
# D circle
########################################################################
ax = subplot(gs[1:,0:2])
text(-0.05, 1.05, "D", transform = gca().transAxes, fontsize = 9)
axis('off')
axhline(0, xmin = 0.25, xmax = 0.75, color = '#386150', linewidth = 0.7)
axvline(0, ymin = 0.25, ymax = 0.75, color = '#58b09c', linewidth = 0.7)
xlim(-20, 20)
ylim(-20, 20)
phase_circle = np.arange(0, 2*np.pi, 0.0001)
# x, y = (np.cos(phi2.values.flatten()), np.sin(phi2.values.flatten()))
x, y = (np.cos(phase_circle),np.sin(phase_circle))
r = 14
plot(x*r, y*r, '-', color = 'black', linewidth = 0.5)
r = r+1
text(-r, 0,'$\pi$', horizontalalignment='center', verticalalignment='center', 	 	fontsize = 6)
text(r, 0,'0', horizontalalignment='center', verticalalignment='center', 		 	fontsize = 6)
text(0, r,'$\pi/2$', horizontalalignment='center', verticalalignment='center',  	fontsize = 6)
text(0, -r,'$3\pi/2$', horizontalalignment='center', verticalalignment='center',  	fontsize = 6)
text(r-5, -2.0, 'jPC1', fontsize = 6)
text(0.7, r-6, 'jPC2', fontsize = 6)


color_points = allthetamodth['phase'].copy()
color_points -= color_points.min()
color_points /= color_points.max()
# scatter(jscore.values[:,0], jscore.values[:,1], s = 3, c = color_points.values, cmap = cm.get_cmap('hsv'), zorder = 2, alpha = 1, linewidth = 0.0)
# scatter(jscore.values[:,0], jscore.values[:,1], s = 3, c = 'black', zorder = 2, alpha = 0.7, linewidth = 0.0)
scatter(jscore.values[:,0], jscore.values[:,1], s = 3, c = allzth.values[:,100], cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.7, linewidth = 0.0)

bb = ax.get_position().bounds
aiw = 0.1
ail = 0.1
position_axes = [
				[bb[0]+bb[2]*0.75,bb[1]+bb[3]*0.75],
				[bb[0]+bb[2]*0.05,bb[1]+bb[3]*0.75],
				[bb[0]+bb[2]*0.05,bb[1]+bb[3]*0.05],
				[bb[0]+bb[2]*0.75,bb[1]+bb[3]*0.05]]
r -= 1
best_neurons = []
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
		best_n = np.argmin(np.abs(allthetamodth.loc[tmp.index.values,'phase'] - (i+np.pi/8)))	
	best_neurons.append(best_n)
	ai = axes([position_axes[j][0],position_axes[j][1], aiw, ail], projection = 'polar')
	ai.get_xaxis().tick_bottom()
	ai.get_yaxis().tick_left()
	ai.hist(spikes_theta_phase['rem'][best_n], 30, color = 'red', normed = True)
	xticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	yticks([])	
	grid(linestyle = '--')
	if j == 1:
		ai.set_title("Theta phase", fontsize = 8)
	ai.yaxis.grid(False)
	ai.tick_params(axis='x', pad = -5)
	# ai.set_ylim(0,0.5)
	ai.arrow(x = allthetamodth.loc[best_n,'phase'], y = 0, dx = 0, dy = ai.get_ylim()[1]*0.6,
			edgecolor = 'black', facecolor = 'green', lw = 1.0, head_width = 0.1, head_length = 0.02,zorder = 5)
	

	x = np.cos(quarter.loc[best_n,0])*r
	y = np.sin(quarter.loc[best_n,0])*r
	xx, yy = (jscore.loc[best_n,0],jscore.loc[best_n,1])
	ax.scatter(x, y, s = 6, c = 'red', cmap = cm.get_cmap('viridis'), zorder = 2)
	ax.scatter(xx, yy, s = 6, c = 'red', cmap = cm.get_cmap('viridis'), zorder = 2)
	
	ax.arrow(xx, yy, x - xx, y - yy,
		head_width = 0.8,
		linewidth = 0.4,
		length_includes_head = True,		
		color = 'grey'
		)
	



# text(0, 0, '$\mathbf{SWR\ jPCA\ phase}$',horizontalalignment='center')
text(0.5, 0.8, 'SWR jPCA phase', fontsize = 8, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# lower_left = np.argmin(jscore.values[:,0])
# text(-35.,-7, 'arctan2', rotation = 13.0)
# cbaxes = fig.add_axes([0.25, 0.45, 0.01, 0.04])
# cmap = cm.viridis
# norm = matplotlib.colors.Normalize(allzth.values[:,100].min(), allzth.values[:,100].max())
# cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
# cbaxes.axes.set_xlabel('SWR \n modulation')





#########################################################################
# E PHASE PHASE SCATTER
#########################################################################
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[4])
ax = subplot(gs[1,2])
simpleaxis(ax)
# dist_cp = np.sqrt(np.sum(np.power(eigen[0] - eigen[1], 2))
theta_mod_toplot = allthetamodth.values[:,0].astype('float32')#,dist_cp>0.02]
phi_toplot = phi2.values.flatten()

r, p = corr_circular_(theta_mod_toplot, phi_toplot)
print(r, p)

x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
scatter(x, y, s = 0.8, c = np.tile(allzth.values[:,100],4), cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.5)
# scatter(x, y, s = 0.8, c = np.tile(color_points,4), cmap = cm.get_cmap('hsv'), zorder = 2, alpha = )

scatter(allthetamodth.loc[best_neurons, 'phase'].values, phi2.loc[best_neurons].values.flatten(), color = 'red', s = 5, zorder = 5)


xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad)', labelpad = 1.2)
ylabel('SWR jPCA phase (rad)')
title(r'$r = 0.18, p = 2.3 \times 10^{-7}$',fontsize = 6)
text(-0.1, 1.1, "E", transform = gca().transAxes, fontsize = 9)

########################################################################
# F PHASE PHASE DENSITY
########################################################################
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[5])
ax = subplot(gs[2,2])
text(-0.1, 1.1, "F", transform = gca().transAxes, fontsize = 9)
H, xedges, yedges = np.histogram2d(y, x, 50)
H = gaussFilt(H, (3,3))
H = H - H.min()
H = H / H.max()
print(np.sum(np.isnan(H)))
# imshow(H, origin = 'lower', interpolation = 'nearest', aspect = 'auto')
levels = np.linspace(H.min(), H.max(), 50)
axp = ax.contourf(H, V = levels, cmap = 'binary')
xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad)')
ylabel('SWR jPCA phase (rad)')
tik = np.array([0, np.pi, 2*np.pi, 3*np.pi])
xtik = [np.argmin(np.abs(i-xedges)) for i in tik]
ytik = [np.argmin(np.abs(i-yedges)) for i in tik]
xticks(xtik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
yticks(ytik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
title("Density", fontsize = 8)
cbaxes = fig.add_axes([0.63, 0.11, 0.01, 0.04])
cb = colorbar(axp, cax = cbaxes, ticks = [0, 1])
cbaxes.yaxis.set_ticks_position('left')





savefig("../../figures/figures_articles/figart_3.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_3.pdf &")

