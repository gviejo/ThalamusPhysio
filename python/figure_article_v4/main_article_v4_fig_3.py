

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


fig = figure(figsize = figsize(1.0))

outergs = gridspec.GridSpec(3,2, figure = fig, height_ratios = [0.9,1.0,1.1], hspace = 0.45)

#############################################################################
# A. LFP SPIKES
#############################################################################
gsEx = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outergs[0,0], hspace = 0, wspace = 0, height_ratios = [0.4,1])

session 		= 'Mouse12/Mouse12-120810'
neurons 		= [session.split("/")[1]+"_"+str(u) for u in [38,37,40]]
path_snippet 	= "../../figures/figures_articles_v2/figure2/"
store 			= pd.HDFStore(path_snippet+'snippet_'+session.split("/")[1]+'.h5')
phase_spike 	= pd.HDFStore(path_snippet+'spikes_'+session.split("/")[1]+'.h5')
spike_in_swr 	= pd.HDFStore(path_snippet+'spikes_in_swr_'+session.split("/")[1]+'.h5')		
modulations 	= pd.HDFStore(path_snippet+'modulation_theta2_swr_Mouse17.h5')
phase_spike_theta = {
'Mouse12-120810_37':store['phase_spike_theta_Mouse12-120810_37'],
'Mouse12-120810_38':store['phase_spike_theta_Mouse12-120810_38'],
'Mouse12-120810_40':store['phase_spike_theta_Mouse12-120810_40']	
}
store.close()
store_ex = pd.HDFStore('../../figures/figures_articles_v2/figure3/lfp_exemple.h5', 'r')
lfp_sws_ex  = store_ex['lfp_sws_ex']
lfp_rem_ex  = store_ex['lfp_rem_ex'] 	
spikes_sws_ex  = store_ex['spikes_sws_ex']
spikes_rem_ex  = store_ex['spikes_rem_ex']
hd_info_neuron = store_ex['hd_info_neuron']
rip_ep = store_ex['rip_ep']
rip_tsd = store_ex['rip_tsd']
hd_phi_neuron = store_ex['hd_phi_neuron']
store_ex.close()

colors = ['#231f20', '#707174', '#abacad']
titles = ['REM']
# 1 lfps
i = 0
lfp = lfp_rem_ex
# ax = Subplot(fig, )
ax = fig.add_subplot(gsEx[0,i])
noaxis(ax)
plot(lfp, color = 'black', linewidth = 0.4)
title(titles[i], pad = -2)
text(-0.1, 1.1, 'a', transform=ax.transAxes, fontsize = 10, fontweight='bold')
ylabel("CA1")

# 2 spikes
i = 0
spikes = spikes_rem_ex
# ax = Subplot(fig, )
ax = fig.add_subplot(gsEx[1,i])
noaxis(ax)
# no hd
id = 0
for n in np.where(hd_info_neuron == 0)[0]:
	plot(spikes[n].dropna().replace(n, id), '|', markersize = 2, mew = 0.8, color = 'black')
	id += 1
# hd 
hd_order = np.where(hd_info_neuron==1)[0][np.argsort(hd_phi_neuron.values[hd_info_neuron==1])]
for n in hd_order:
	plot(spikes[n].dropna().replace(n, id), '|', markersize = 2, mew = 0.8, color = 'red')
	id += 1

if i == 0:
	ylabel("Thalamus")
start = spikes.index.min()
plot([start, start+5e5], [-2, -2], color = 'black', linewidth = 2)
text(0.05, -0.1,'500 ms', transform=ax.transAxes, fontsize = 8)

ylim(-2, id)


#############################################################################
# B. THETA PHASE MODULATION
#############################################################################
gsC = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec = outergs[0,1], wspace = 0.6)

for i, n in enumerate(neurons):	
	ax = fig.add_subplot(gsC[0,i])
	simpleaxis(ax)
	if neurons.index(n) == 1:
		text(0.5, 1.17,'Theta phase modulation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 8)
	if neurons.index(n) > 0:
		ax.spines['left'].set_visible(False)	
		yticks([], [])
	if neurons.index(n) == 0:
		ylabel("Theta cycles")
		ax.text(-0.82, 1.1, "b", transform = ax.transAxes, fontsize = 10, fontweight='bold')
	ax.spines['bottom'].set_visible(False)
	sp = phase_spike[n]
	plot(sp.iloc[0:500], '|', markersize = 2, color = colors[neurons.index(n)], mew = 0.5)	
	xticks([], [])
	axp = fig.add_subplot(gsC[1,i])	
	simpleaxis(axp)
	tmp = phase_spike_theta[n].values
	tmp += 2*np.pi
	tmp %= 2*np.pi
	axp.hist(tmp,20, color = colors[neurons.index(n)], density = True, histtype='stepfilled')
	xticks([0, 2*np.pi], ['0', '$2\pi$'])
	if i ==1 : xlabel("phase (rad)", labelpad = -0.05)

#############################################################################
# C. THETA MODULATION VS RIPPLES POWER
#############################################################################
outer = gridspec.GridSpecFromSubplotSpec(1,5, subplot_spec = outergs[1,:], wspace = 0.2, width_ratios= [0.5, 0.3, 0.06, 0.5, 0.01])

rippower 				= pd.read_hdf("../../figures/figures_articles_v2/figure2/power_ripples_2.h5")
theta2 = pd.read_hdf("/mnt/DataGuillaume/MergedData/THETA_THAL_mod_2.h5")
theta = theta2['rem']
theta = theta[theta['kappa']<2.0]
neurons = np.intersect1d(theta.index.values, rippower.index.values)
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
hd_neurons = mappings.loc[neurons][mappings.loc[neurons, 'hd'] == 1].index.values
nohd_neurons = mappings.loc[neurons][mappings.loc[neurons, 'hd'] == 0].index.values

gs = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = outer[0,0], width_ratios = [0.8, 0.2], height_ratios = [0.2, 0.8], wspace= 0.05, hspace = 0.05)

ax1 = fig.add_subplot(gs[1,1])
y = rippower.loc[nohd_neurons].values.astype('float')
weights = np.ones_like(y)/float(len(y))
hist(y, 20, weights = weights, color = 'grey', orientation = 'horizontal', histtype='stepfilled')
x = rippower.loc[hd_neurons].values.astype('float')
weights = np.ones_like(x)/float(len(x))
hist(x, 20, weights = weights, color = 'red', orientation = 'horizontal', histtype='stepfilled')
yticks([], [])
simpleaxis(ax1)

ax2 = fig.add_subplot(gs[1,0])
scatter(theta.loc[nohd_neurons, 'kappa'], rippower.loc[nohd_neurons], c = 'grey', s = 6)
scatter(theta.loc[hd_neurons, 'kappa'], rippower.loc[hd_neurons], c = 'red', s= 6)
ax2.set_xlabel("Theta modulation ($\\kappa$)")
ax2.set_ylabel("SWR energy (|z|)")

ax3 = fig.add_subplot(gs[0,0])
y = theta.loc[nohd_neurons, 'kappa'].values.astype('float')
weights = np.ones_like(y)/float(len(y))
hist(y, 20, weights = weights, color = 'grey', histtype='stepfilled')
x = theta.loc[hd_neurons, 'kappa'].values.astype('float')
weights = np.ones_like(x)/float(len(x))
hist(x, 20, weights = weights, color = 'red', histtype='stepfilled')
xticks([], [])
ax3.text(-0.15, 1.3, "c", transform = ax3.transAxes, fontsize = 10, fontweight='bold')
simpleaxis(ax3)

#############################################################################
# D. MAPS theta 
#############################################################################
axE = fig.add_subplot(outer[0,1])


carte_adrien2 = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_Contour-01.png')
bound_adrien = (-398/1254, 3319/1254, -(239/1254 - 20/1044), 3278/1254)

noaxis(axE)
tmp = modulations['theta']
bound = (tmp.columns[0], tmp.columns[-1], tmp.index[-1], tmp.index[0])
im = imshow(tmp, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'Greys', vmin = 0, vmax = 1)
imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bilinear', aspect = 'equal')
# title("Theta spatial modulation", fontsize = 7, y = 1.3)
axE.text(0.06, 1.05, "Theta spatial\n modulation", transform = axE.transAxes, fontsize = 8, multialignment='center')
#colorbar	
cax = inset_axes(axE, "40%", "8%",
                   bbox_to_anchor=(0.2, -0.25, 1, 1),
                   bbox_transform=axE.transAxes, 
                   loc = 'lower left')
cb = colorbar(im, cax = cax, orientation = 'horizontal', ticks = [0,1])
# cb.set_label('Density (p < 0.05)' , labelpad = -0)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("Density (p < 0.01)", fontsize = 7, pad = 2.5)
axE.text(-0.15, 1.22, "d", transform = axE.transAxes, fontsize = 10, fontweight='bold')


########################################################################
# E. SHarp waves ripples modulation
#######################################################################
gs = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outer[0,3], height_ratios = [0.6,0.4], hspace = 0.5)
subplot(gs[0,0])
im = imshow(allzthsorted, aspect = 'auto', cmap = 'viridis')
xticks(np.arange(20,200,40), (times[np.arange(20,200,40)]).astype('int'))
yticks([0,700], ['1', '700'])
# cb = colorbar()
# cb.set_label("z", labelpad = -10, y = 1.08, rotation = 0)
ylabel("Thalamic\nneurons", labelpad = -5.0)
xlabel("Time from SWRs (ms)", labelpad = -0.05)
title("SWR modulation", fontsize = 8, y = 0.95)
text(-0.3, 1.06, "e", transform = gca().transAxes, fontsize = 10, fontweight='bold')

cax = inset_axes(gca(), "4%", "100%",
                   bbox_to_anchor=(1, -0.06, 1, 1),
                   bbox_transform=gca().transAxes, 
                   loc = 'lower left')
cax.set_title("z", fontsize = 7, pad = 2.5)
cb = colorbar(im, cax = cax, orientation = 'vertical', ticks = [-2, 0, 2])


########################################################################
# F. JPCA
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
text(-0.3, 1.05, "f", transform = gca().transAxes, fontsize = 10, fontweight='bold')


########################################################################
# g circle
########################################################################
gs = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[2,:], width_ratios = [0.6,0.4], wspace = 0.5)
ax = subplot(gs[0,0])
ax.set_aspect("equal")
text(-0.6, 1.12, "g", transform = gca().transAxes, fontsize = 10, fontweight='bold')
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
				[bb[0]+bb[2]*0.9,bb[1]+bb[3]*0.48],
				[bb[0]+bb[2]*-0.85,bb[1]+bb[3]*0.48],
				[bb[0]+bb[2]*-0.85,bb[1]+bb[3]*-0.43],
				[bb[0]+bb[2]*0.9,bb[1]+bb[3]*-0.43]]
r -= 1
best_neurons = []
lbs = ['a', 'b', 'c', 'd']
cmv = get_cmap('viridis')
clr = cmv(0.6)

text(0,1,"Theta phase", fontsize =8,transform=ax.transAxes, color = clr)

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
	ai.hist(spikes_theta_phase['rem'][best_n], 30, color = clr, normed = True, histtype='stepfilled')
	xticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	yticks([])	
	grid(linestyle = '--')
	
	ai.yaxis.grid(False)
	ai.tick_params(axis='x', pad = -5)
	# ai.set_ylim(0,0.5)
	ai.arrow(x = allthetamodth.loc[best_n,'phase'], y = 0, dx = 0, dy = ai.get_ylim()[1]*0.6,
			edgecolor = 'black', facecolor = 'green', lw = 1.0, head_width = 0.1, head_length = 0.02,zorder = 5)
	

	x = np.cos(quarter.loc[best_n,0])*r
	y = np.sin(quarter.loc[best_n,0])*r
	xx, yy = (jscore.loc[best_n,0],jscore.loc[best_n,1])
	
	
	ax.scatter([x], [y], s = 20, c = clr, cmap = cm.get_cmap('viridis'), alpha = 1)
	ax.scatter([xx], [yy], s = 6, c = clr, cmap = cm.get_cmap('viridis'), zorder = 2)
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
text(-0.25, 0.45, 'SWR jPCA phase', fontsize = 8, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


#########################################################################
# H PHASE PHASE SCATTER
#########################################################################
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[4])
ax = subplot(gs[0,1])
simpleaxis(ax)
ax.set_aspect("equal")
theta_mod_toplot = allthetamodth.values[:,0].astype('float32')#,dist_cp>0.02]
phi_toplot = phi2.values.flatten()

r, p = corr_circular_(theta_mod_toplot, phi_toplot)
print(r, p)

x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
gca().text(-0.6, 0.96, r'$r = 0.18$', transform = gca().transAxes, fontsize = 8)#, color = 'white')
gca().text(-0.6, 0.82, r'$ p = 2.3 \times 10^{-7}$', transform = gca().transAxes, fontsize = 8)#, color = 'white')
gca().text(-0.7, 1.08, "h", transform = gca().transAxes, fontsize = 10, fontweight='bold')
H, xedges, yedges = np.histogram2d(y, x, 50)
H = gaussFilt(H, (3,3))
H = H - H.min()
H = H / H.max()
print(np.sum(np.isnan(H)))
axp = ax.contourf(H, cmap = 'Greys', extent = (xedges[0], xedges[-2], yedges[0], yedges[-2]))
xlabel('Theta phase (rad.)', labelpad = 0.1)
ylabel('SWR jPCA \nphase (rad.)', labelpad = 4)
tik = np.array([0, np.pi, 2*np.pi, 3*np.pi])
xticks(tik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
yticks(tik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
title("Density", fontsize = 9)

clr = cmv(0.6)
scatter(allthetamodth.loc[best_neurons, 'phase'].values, phi2.loc[best_neurons].values.flatten(), color = clr, s = 12, zorder = 5)
for i in range(4):
	xy = (allthetamodth.loc[best_neurons, 'phase'].values[i], phi2.loc[best_neurons].values.flatten()[i])
	annotate(lbs[i], xy, (xy[0]+0.1, xy[1]+0.2), color = 'white')


#colorbar	
cax = inset_axes(gca(), "4%", "20%",
                   bbox_to_anchor=(-0.4, -0.1, 1, 1),
                   bbox_transform=gca().transAxes, 
                   loc = 'lower left')
cb = colorbar(axp, cax = cax, orientation = 'vertical', ticks = [0.25, 0.75])




fig.subplots_adjust(top = 0.97, bottom = 0.05, right = 0.97, left = 0.06)

savefig("../../figures/figures_articles_v4/figart_3.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v4/figart_3.pdf &")

