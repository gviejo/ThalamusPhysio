

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
gs = gridspec.GridSpec(5,3, wspace = 0.3, hspace = 0.6, left = 0.08, right = 0.98, top = 0.97, bottom = 0.05, height_ratios = [0.25,0.2,0.3,-0.12,0.6])

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
title("SWR modulation", fontsize = 8, y = 0.95)
text(-0.3, 1.03, "A", transform = gca().transAxes, fontsize = 10)

cax = inset_axes(gca(), "4%", "100%",
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
legend(frameon=False,loc = 'lower left', bbox_to_anchor=(0.9,0.4),handlelength=1)
ylabel('jPC', labelpad = 0.1)
xlabel('Time from SWRs (ms)', labelpad = -0.05)
xticks([-400,-200,0,200,400])
# title('jPCA', fontsize = 8, y = 1)
# text(0.15, 0.86, "jPCA", transform = gca().transAxes, fontsize = 10)
text(-0.3, 1.05, "B", transform = gca().transAxes, fontsize = 10)

# ########################################################################
# # C. ORBIT
# ########################################################################
# ax = subplot(gs[0, 2])
# # axis('off')
# simpleaxis(ax)
# plot(jX[0,0], jX[0,1], 'o', markersize = 3, color = '#5c7d6f')
# plot(jX[:,0], jX[:,1], linewidth = 0.8, color = '#5c7d6f')
# arrow(jX[-10,0],jX[-10,1],jX[-1,0]-jX[-10,0],jX[-1,1]-jX[-10,1], color = '#5c7d6f', head_width = 0.01)
# # plot(jX[np.where(times==-250),0], jX[np.where(times==-250),1], 'o', color = '#5c7d6f', markersize = 2)
# # plot(jX[np.where(times== 250),0], jX[np.where(times== 250),1], 'o', color = '#5c7d6f', markersize = 2)
# # plot(jX[np.where(times==   0),0], jX[np.where(times==   0),1], 'o', color = '#5c7d6f', markersize = 2)
# annotate("-50 ms", xy = (jX[np.where(times==-50),0], jX[np.where(times==-50),1]), xytext = (jX[np.where(times==-50),0]-0.124, jX[np.where(times==-50),1]-0.015), fontsize = 6)
# annotate( "50 ms", 	xy = (jX[np.where(times== 50),0], jX[np.where(times== 50),1]), 		xytext = (jX[np.where(times== 50),0]+0.01, jX[np.where(times==  50),1]), fontsize = 6)
# annotate(   "0 ms", 	xy = (jX[np.where(times==   0),0], jX[np.where(times==   0),1]), 	xytext = (jX[np.where(times==  0),0]-0.04, jX[np.where(times==   0),1]+0.02), fontsize = 6)
# ax.spines['left'].set_bounds(np.min(jX[:,1]), np.min(jX[:,1]+0.1))
# ax.spines['bottom'].set_bounds(np.min(jX[:,0]), np.min(jX[:,0]+0.1))
# xticks([], [])
# yticks([], [])
# ax.xaxis.set_label_coords(0.15, -0.02)
# ax.yaxis.set_label_coords(-0.02, 0.15)
# ylabel('jPC2')
# xlabel('jPC1')
# text(-0.1, 1.14, "C", transform = gca().transAxes, fontsize = 10)

# jpca = pd.DataFrame(index = times, data = jX)

# offs = 0.1

# # arrow(jpca.loc[50,0], jpca.loc[50,1], jpca.loc[55,0]-jpca.loc[50,0], jpca.loc[55,1]-jpca.loc[50,1], head_width=.020, fc = '#5c7d6f', shape='full', lw=0, length_includes_head=True)
# # arrow(jpca.loc[-5,0], jpca.loc[-5,1], jpca.loc[0,0]-jpca.loc[-5,0], jpca.loc[0,1]-jpca.loc[-5,1], head_width=.020, fc = '#5c7d6f', shape='full', lw=0, length_includes_head=True)
# # arrow(jpca.loc[-45,0], jpca.loc[-45,1], jpca.loc[-40,0]-jpca.loc[-45,0], jpca.loc[-40,1]-jpca.loc[-45,1], head_width=.020, fc = '#5c7d6f', shape='full', lw=0, length_includes_head=True)
# # arrow(jpca.loc[-115,0], jpca.loc[-115,1], jpca.loc[-110,0]-jpca.loc[-115,0], jpca.loc[-110,1]-jpca.loc[-115,1], head_width=.020, fc = '#5c7d6f', shape='full', lw=0, length_includes_head=True)

# for t in np.arange(-200,250,50):	
# 	arrow(jpca.loc[t-5,0], jpca.loc[t-5,1], jpca.loc[t,0]-jpca.loc[t-5,0], jpca.loc[t,1]-jpca.loc[t-5,1], head_width=.020, fc = '#5c7d6f', shape='full', lw=0, length_includes_head=True)	

########################################################################
# C circle
########################################################################
ax = subplot(gs[0:2,1:3])
ax.set_aspect("equal")
text(-0.5, 1.00, "C", transform = gca().transAxes, fontsize = 10)
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

text(0.16,0.95,"Theta phase", fontsize =8,transform=ax.transAxes, color = 'red')

color_points = allthetamodth['phase'].copy()
color_points -= color_points.min()
color_points /= color_points.max()
# scatter(jscore.values[:,0], jscore.values[:,1], s = 3, c = color_points.values, cmap = cm.get_cmap('hsv'), zorder = 2, alpha = 1, linewidth = 0.0)
# scatter(jscore.values[:,0], jscore.values[:,1], s = 3, c = 'black', zorder = 2, alpha = 0.7, linewidth = 0.0)
scatter(jscore.values[:,0], jscore.values[:,1], s = 5, c = allzth.values[:,100], cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.7, linewidth = 0.0)

bb = ax.get_position().bounds
aiw = 0.1
ail = 0.1
position_axes = [
				[bb[0]+bb[2]*0.85,bb[1]+bb[3]*0.70],
				[bb[0]+bb[2]*-0.2,bb[1]+bb[3]*0.70],
				[bb[0]+bb[2]*-0.2,bb[1]+bb[3]*-0.1],
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
	ai = axes([position_axes[j][0],position_axes[j][1], aiw, ail], projection = 'polar')
	ai.get_xaxis().tick_bottom()
	ai.get_yaxis().tick_left()
	ai.hist(spikes_theta_phase['rem'][best_n], 30, color = 'red', normed = True)
	xticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	yticks([])	
	grid(linestyle = '--')
	# xlabel(lbs[j])
	# if j == 1:
	# 	# ai.set_title("Theta phase", fontsize = 8, color = 'red')
	# 	ai.text(1,1,"Theta phase", fontsize =8, color = 'red')		
	ai.yaxis.grid(False)
	ai.tick_params(axis='x', pad = -5)
	# ai.set_ylim(0,0.5)
	ai.arrow(x = allthetamodth.loc[best_n,'phase'], y = 0, dx = 0, dy = ai.get_ylim()[1]*0.6,
			edgecolor = 'black', facecolor = 'green', lw = 1.0, head_width = 0.1, head_length = 0.02,zorder = 5)
	

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
text(0.14,0.71,"b", fontsize =9,transform=ax.transAxes)
text(0.15,0.16,"c", fontsize =9,transform=ax.transAxes)
text(0.73,0.1,"d", fontsize =9,transform=ax.transAxes)



# text(0, 0, '$\mathbf{SWR\ jPCA\ phase}$',horizontalalignment='center')
text(-0.2, 0.45, 'SWR jPCA phase', fontsize = 8, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# lower_left = np.argmin(jscore.values[:,0])
# text(-35.,-7, 'arctan2', rotation = 13.0)
# cbaxes = fig.add_axes([0.25, 0.45, 0.01, 0.04])
# cmap = cm.viridis
# norm = matplotlib.colors.Normalize(allzth.values[:,100].min(), allzth.values[:,100].max())
# cb = matplotlib.colorbar.ColorbarBase(cbaxes, cmap = cmap, norm = norm)
# cbaxes.axes.set_xlabel('SWR \n modulation')





#########################################################################
# D PHASE PHASE SCATTER
#########################################################################
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[4])
ax = subplot(gs[2,0])
simpleaxis(ax)
ax.set_aspect("equal")
# dist_cp = np.sqrt(np.sum(np.power(eigen[0] - eigen[1], 2))
theta_mod_toplot = allthetamodth.values[:,0].astype('float32')#,dist_cp>0.02]
phi_toplot = phi2.values.flatten()

r, p = corr_circular_(theta_mod_toplot, phi_toplot)
print(r, p)

x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
# scatter(x, y, s = 2, c = np.tile(allzth.values[:,100],4), cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.5)
# # scatter(x, y, s = 0.8, c = np.tile(color_points,4), cmap = cm.get_cmap('hsv'), zorder = 2, alpha = )
# xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
# yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
# xlabel('Theta phase (rad.)', labelpad = 1.2)
# ylabel('SWR jPCA phase (rad.)')
gca().text(0.15, 0.9, r'$r = 0.18$', transform = gca().transAxes, fontsize = 8, color = 'white')
gca().text(0.15, 0.78, r'$ p = 2.3 \times 10^{-7}$', transform = gca().transAxes, fontsize = 8, color = 'white')
gca().text(-0.9, 1.05, "D", transform = gca().transAxes, fontsize = 10)


# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[5])
# ax = subplot(gs[2,2])
# text(-0.1, 1.1, "F", transform = gca().transAxes, fontsize = 10)
H, xedges, yedges = np.histogram2d(y, x, 50)
H = gaussFilt(H, (3,3))
H = H - H.min()
H = H / H.max()
print(np.sum(np.isnan(H)))
# imshow(H, origin = 'lower', interpolation = 'nearest', aspect = 'auto')
# levels = np.linspace(H.min(), H.max(), 50)
axp = ax.contourf(H, cmap = 'Greys', extent = (xedges[0], xedges[-2], yedges[0], yedges[-2]))
# xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
# yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad.)', labelpad = 0.1)
ylabel('SWR jPCA \nphase (rad.)', labelpad = 4)
tik = np.array([0, np.pi, 2*np.pi, 3*np.pi])
# xtik = [np.argmin(np.abs(i-xedges)) for i in tik]
# ytik = [np.argmin(np.abs(i-yedges)) for i in tik]
xticks(tik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
yticks(tik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
title("Density", fontsize = 8, y = 0.94)

scatter(allthetamodth.loc[best_neurons, 'phase'].values, phi2.loc[best_neurons].values.flatten(), color = 'red', s = 6, zorder = 5)
for i in range(4):
	xy = (allthetamodth.loc[best_neurons, 'phase'].values[i], phi2.loc[best_neurons].values.flatten()[i])
	annotate(lbs[i], xy, (xy[0]+0.1, xy[1]+0.2), color = 'white')


# cbaxes = fig.add_axes([0.4, 0.4, 0.04, 0.01])
# cb = colorbar(axp, cax = cbaxes, orientation = 'horizontal', ticks = [0, 1])
# # cbaxes.yaxis.set_ticks_position('left')

#colorbar	
cax = inset_axes(gca(), "4%", "20%",
                   bbox_to_anchor=(1.0, 0.0, 1, 1),
                   bbox_transform=gca().transAxes, 
                   loc = 'lower left')
cb = colorbar(axp, cax = cax, orientation = 'vertical', ticks = [0.25, 0.75])


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
gsm = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec = gs[2,1:])
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
	subplot(gsm[0,j])
	if j == 0:
		text(-0.1, 1.05, "E", transform = gca().transAxes, fontsize = 10)
	if j == 1: 
		title("SWR modulation (Mouse 1)", pad = -0.05)
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




#####################################################################
# F SWR NUCLEUS
#####################################################################
gsm = gridspec.GridSpecFromSubplotSpec(len(nucleus)+1,2, subplot_spec = gs[4,:], height_ratios = [1]+[0.1]*len(nucleus))

idm = swr_all.idxmax()
idm = idm.sort_values()
order = idm.index.values

subplot(gsm[0,0])
simpleaxis(gca())
for n in nucleus:
	plot(swr_all[n], label = n)
axvline(0, linestyle = '--', linewidth = 1, alpha = 0.5, color = 'black')
xticks([], [])
text(-0.17, 1.1, "F", transform = gca().transAxes, fontsize = 10)
ylabel("SWRs mod.", labelpad = 2, y = 0.6)
xlim(-500,500)
legend(frameon=False,loc = 'lower left', bbox_to_anchor=(0.6,0.38),handlelength=1,ncol = 2)

######################################################################
for i, n in enumerate(order):
	subplot(gsm[i+1,0])
	tmp = swr_nuc.xs(n,1,1).xs('mean',1,1).T.values.astype(np.float32)
	imshow(tmp, aspect = 'auto', cmap = 'bwr')
	if i == len(nucleus)-1:	
		xticks([0,100,200],[-500,0,500])		
		xlabel("Time lag (ms)", labelpad = 0.9)	
	else:
		xticks([], [])
	# if i == 0:
	# 	# yticks([0,3], [1,4])
	# 	ylabel(n, rotation = 0, labelpad = 8, y = 0.2)
	# else:
	yticks([0,3], ['',''])
	ylabel(n, rotation = 0, labelpad = 11, y = 0.2)
	# if i == len(order)-1:
	if i == 0:
		annotate("Mouse", 		(1.08, 2.4), (1.04,  2.7 ), xycoords = 'axes fraction', fontsize = 7)
		annotate("1", 		(1, 0.75+0.125), (1.08,  1.6 ), xycoords = 'axes fraction', fontsize = 7, arrowprops = {'arrowstyle':'-'})
		annotate("2", 		(1, 0.50+0.125), (1.08,  0.8), xycoords = 'axes fraction', 	fontsize = 7, arrowprops = {'arrowstyle':'-'})
		annotate("3", 		(1, 0.25+0.125), (1.08,  0.0), xycoords = 'axes fraction', 	fontsize = 7, arrowprops = {'arrowstyle':'-'})
		annotate("4", 		(1, 0.0 +0.125), (1.08, -0.8 ), xycoords = 'axes fraction', fontsize = 7, arrowprops = {'arrowstyle':'-'})


#####################################################################
# G ORBIT
#####################################################################
subplot(gsm[:,1])
simpleaxis(gca())
gca().set_aspect("equal")
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


plot(jX[0,0], jX[0,1], 'o', markersize = 3, color = 'black')
plot(jX[:,0], jX[:,1], linewidth = 1, color = 'black')
arrow(jX[-10,0],jX[-10,1],jX[-1,0]-jX[-10,0],jX[-1,1]-jX[-10,1], color = 'black', head_width = 0.01)
gca().spines['left'].set_bounds(np.min(jX[:,1]), np.min(jX[:,1]+0.1))
gca().spines['bottom'].set_bounds(np.min(jX[:,0]), np.min(jX[:,0]+0.1))
# gca().spines['left'].set_visible(False)
# gca().spines['bottom'].set_visible(False)
xticks([], [])
yticks([], [])
gca().xaxis.set_label_coords(0.15, -0.02)
gca().yaxis.set_label_coords(-0.02, 0.15)
ylabel('jPC2')
xlabel('jPC1')
text(-0.1, 1.02, "G", transform = gca().transAxes, fontsize = 10)

jpca = pd.DataFrame(index = times2, data = jX)


fontsize = np.array([swr_all.loc[idm[n],n]*100.0 for n in order])
fontsize -= np.min(fontsize)
fontsize /= np.max(fontsize)
fontsize = 1/(1+np.exp(-8*(fontsize-0.5)))
fontsize = fontsize*6 + 7.0
fontsize = pd.Series(index = order, data = fontsize)

for n in order :	
	if n == 'PV':
		text(jpca.loc[idm[n],0]-0.04,jpca.loc[idm[n],1]+0.03,n, 
			ha = 'center', 
			va = 'center', 
			bbox = dict(boxstyle='square,pad=0.1',fc='white',ec='none'),
			fontsize = fontsize[n])
	else:
		text(jpca.loc[idm[n],0],jpca.loc[idm[n],1],n, 
			ha = 'center', 
			va = 'center', 
			bbox = dict(boxstyle='square,pad=0.1',fc='white',ec='none'),
			fontsize = fontsize[n])

tx = [-100,0]
# plot(jpca.loc[tx,0], jpca.loc[tx,1], '.', color = 'black')

annotate('0 ms', (jpca.loc[0,0],jpca.loc[0,1]), (jpca.loc[0,0]+0.009,jpca.loc[0,1]-0.02))
# annotate('-100 ms', (jpca.loc[-100,0],jpca.loc[-100,1]), (jpca.loc[-100,0]-0.0,jpca.loc[-100,1]-0.03))
annotate('50 ms', (jpca.loc[50,0],jpca.loc[50,1]), (jpca.loc[50,0]-0.0,jpca.loc[50,1]+0.01))
offs = 0.1
# arrow(jpca.loc[50,0], jpca.loc[50,1], jpca.loc[55,0]-jpca.loc[50,0], jpca.loc[55,1]-jpca.loc[50,1], head_width=.015, fc = 'black', shape='full', lw=0, length_includes_head=True)
# arrow(jpca.loc[0,0], jpca.loc[0,1], jpca.loc[5,0]-jpca.loc[0,0], jpca.loc[5,1]-jpca.loc[0,1], head_width=.015, fc = 'black', shape='full', lw=0, length_includes_head=True)
# arrow(jpca.loc[-45,0], jpca.loc[-45,1], jpca.loc[-40,0]-jpca.loc[-45,0], jpca.loc[-40,1]-jpca.loc[-45,1], head_width=.015, fc = 'black', shape='full', lw=0, length_includes_head=True)
# arrow(jpca.loc[-115,0], jpca.loc[-115,1], jpca.loc[-110,0]-jpca.loc[-115,0], jpca.loc[-110,1]-jpca.loc[-115,1], head_width=.015, fc = 'black', shape='full', lw=0, length_includes_head=True)
for t in np.arange(-200,250,50):	
	arrow(jpca.loc[t-5,0], jpca.loc[t-5,1], jpca.loc[t,0]-jpca.loc[t-5,0], jpca.loc[t,1]-jpca.loc[t-5,1], head_width=.018, fc = 'black', shape='full', lw=0, length_includes_head=True)	






savefig("../../figures/figures_articles_v2/figart_4.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v2/figart_4.pdf &")

