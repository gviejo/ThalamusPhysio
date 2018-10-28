
###############################################################################################################
# TO PLOT THE EXEMPLES OF THETA MODULATION
# AND SWR MODULATION
#
###############################################################################################################


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
allzth 			= 	data['swr_modth'	]
eigen 			= 	data['eigen'		]		
times 			= 	data['times' 		]		
allthetamodth 	= 	data['theta_modth'	]		
phi 			= 	data['phi' 			]		
zpca 			= 	data['zpca'			]		
phi2			= 	data['phi2' 		]	 					
sys.exit()
# sort allzth 
index = np.argsort(allzth[:,np.where(times == 0)[0][0]])
index = index[::-1]
allzthsorted = allzth[index,:]
zpca = zpca[index,:]
phi = phi[index]
allthetamodth = allthetamodth[index,:]

###############################################################################################################
# PLOT
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
gs = gridspec.GridSpec(6,3, wspace = 0.4, hspace = 1.0)
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[0])

subplot(gs[0:2,0])
imshow(allzthsorted, aspect = 'auto', cmap = 'viridis')
xticks(np.arange(20,200,40), (times[np.arange(20,200,40)]).astype('int'))
yticks([0,1000], ['0', '1000'])
cb = colorbar()
cb.set_label("z", labelpad = -13, y = 1.08, rotation = 0)
ylabel("Thalamic neurons", labelpad = -5.0)
xlabel("Time from SWR (ms)")
title("Sharp-waves Ripples \n modulation", fontsize = 5)

# gs = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = outer[1], hspace = 0.6)
ax = subplot(gs[0, 1])
simpleaxis(ax)	
cmap = plt.get_cmap('viridis')
cNorm = matplotlib.colors.Normalize(vmin = allzthsorted[:,100].min(), vmax = allzthsorted[:,100].max())
scalarMap = matplotlib.cm.ScalarMappable(norm = cNorm, cmap = cmap)
for i in range(0,10):
	colorVal = scalarMap.to_rgba(allzthsorted[i,100])
	plot(times, allzthsorted[i], color = colorVal, linewidth = 1.0, alpha = 0.8)
ylabel('z')
title('Positive modulation', fontsize = 5, y = 1)
ax = subplot(gs[1, 1])	
simpleaxis(ax)
for i in range(len(allzthsorted)-10,len(allzthsorted)):
	colorVal = scalarMap.to_rgba(allzthsorted[i,100])
	plot(times, allzthsorted[i], color = colorVal, linewidth = 1.0, alpha = 0.8)	
ylabel('z')
xlabel('Time from SWR (ms)')
title('Negative modulation', fontsize = 5, y = 1)


# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[2])
ax = subplot(gs[0:2, 2])
simpleaxis(ax)
plot(times, eigen[0], label = 'PC 1', color = 'black')
plot(times, eigen[1], label = 'PC 2', color = 'red')
ylabel('PC score')
legend(fancybox = False, edgecolor = 'None')
title('Principal Component \n analysis', fontsize = 5)
xlabel('Time from SWR')


# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[3:5])
ax = subplot(gs[2:,0:2])
# ax = subplot2grid(gs[0], colspan = 2, rowspan = 2)
# simpleaxis(ax)
axis('off')
axhline(0, xmin = 0.25, xmax = 0.75, color = 'black', linewidth = 0.7)
axvline(0, ymin = 0.25, ymax = 0.75, color = 'red', linewidth = 0.7)
text(14.0, -1.7, 'PC1', fontsize = 5)
text(0.7, 15.0, 'PC2', fontsize = 5)
xlim(-35, 35)
ylim(-35, 35)
# tmp = ax.get_position().bounds
# ai = axes([tmp[0]+tmp[2]*0.7,tmp[1]+tmp[3]*0.8, 0.06, 0.07])
# ai.get_xaxis().tick_bottom()
# ai.get_yaxis().tick_left()
# ai.plot(np.random.rand(100), 'o')
x, y = (np.cos(phi), np.sin(phi))
r = 29+np.random.rand(len(x))

# scatter(x*r, y*r, s = 3, c = phi, cmap = cm.get_cmap('hsv'), zorder = 2)
scatter(x*r, y*r, s = 2, c = allzthsorted[:,100], cmap = cm.get_cmap('viridis'), zorder = 2)
x2, y2 = (np.cos(allthetamodth[:,0]), np.sin(allthetamodth[:,0]))
r2 = 35
# scatter(x2*r2, y2*r2, s = 3, c = 'grey')
# for n in range(0,len(zpca),2):
# 	plot([zpca[n][0], x[n]*r], [zpca[n][1], y[n]*r], color = 'grey', linewidth = 0.2, alpha = 0.2, zorder = 1)
scatter(zpca[:,0], zpca[:,1], s = 2, c = allzthsorted[:,100], cmap = cm.get_cmap('viridis'), zorder = 2)
# text(0, 37, 'Theta phase'  ,horizontalalignment='center')
text(0, 32, '$\mathbf{SWR\ PCA\ phase}$',horizontalalignment='center')
# lower_left = 1001
lower_left = 1010
arrow(zpca[lower_left,0],zpca[lower_left,1], x[lower_left]*r[lower_left]-zpca[lower_left,0]+3, y[lower_left]*r[lower_left]-zpca[lower_left,1]+3,
		head_width = 0.8,
		linewidth = 0.5
		)
text(-25.,5, 'arctan2', rotation = -15.0)

# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[4])
ax = subplot(gs[2:4,2])
simpleaxis(ax)
# dist_cp = np.sqrt(np.sum(np.power(eigen[0] - eigen[1], 2))
theta_mod_toplot = allthetamodth[:,0]#,dist_cp>0.02]
phi_toplot = phi #[dist_cp>0.02]
x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
scatter(x, y, s = 0.8, c = np.tile(allzthsorted[:,100],4), cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.5)
xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad)', labelpad = 1.2)
ylabel('SWR PCA phase (rad)')


# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[5])
ax = subplot(gs[4:6,2])
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
ylabel('SWR PCA phase (rad)')
tik = np.array([0, np.pi, 2*np.pi, 3*np.pi])
xtik = [np.argmin(np.abs(i-xedges)) for i in tik]
ytik = [np.argmin(np.abs(i-yedges)) for i in tik]
xticks(xtik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
yticks(ytik, ('0', '$\pi$', '$2\pi$', '$3\pi$'))
title("Density", fontsize = 5)
cbaxes = fig.add_axes([0.63, 0.11, 0.01, 0.04])
cb = colorbar(axp, cax = cbaxes, ticks = [0, 1])
cbaxes.yaxis.set_ticks_position('left')


savefig("../../figures/fig1.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/fig1.pdf &")

