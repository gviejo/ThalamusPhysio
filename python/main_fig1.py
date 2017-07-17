

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

###############################################################################################################
# TO LOAD
###############################################################################################################
data = cPickle.load(open('../data/to_plot.pickle', 'rb'))
allzth 			= 	data['allzth'		]		
eigen 			= 	data['eigen'		]		
times 			= 	data['times' 		]		
allthetamodth 	= 	data['allthetamodth']		
phi 			= 	data['phi' 			]		
zpca 			= 	data['zpca'			]		
phi2			= 	data['phi2' 		]	 					

# sort allzth 
index = np.argsort(allzth[:,np.where(times == 0)[0][0]])
allzthsorted = allzth[index,:]

###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*0.85            # height in inches
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
	"legend.fontsize": 5,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 5,
	"ytick.labelsize": 5,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.5,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}    
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid.inset_locator import inset_axes


figure(figsize = figsize(1))

subplot(2,3,1)
imshow(allzthsorted, aspect = 'auto')

subplot(2,3,2)
plot(times, allzthsorted.transpose())

subplot(2,3,3)
plot(times, eigen[0])
plot(times, eigen[1])

subplot(2,3,4)
idxColor = np.digitize(allthetamodth[:,0], np.linspace(0, 2*np.pi+0.0001, 61))
idxColor = idxColor-np.min(idxColor)
idxColor = idxColor/float(np.max(idxColor))
sizes = allthetamodth[:,2] - np.min(allthetamodth[:,2])
sizes = allthetamodth[:,2]/float(np.max(allthetamodth[:,2])) 
colors = cm.rainbow(idxColor)
scatter(zpca[:,0], zpca[:,1], s = sizes*150.+10., c = colors)

subplot(2,3,5)
# dist_cp = np.sqrt(np.sum(np.power(eigen[0] - eigen[1], 2))
theta_mod_toplot = allthetamodth[:,0]#,dist_cp>0.02]
phi_toplot = phi #[dist_cp>0.02]
x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
plot(x, y, 'o', markersize = 1)
xlabel('Theta phase (rad)')
ylabel('SWR PCA phase (rad)')

subplot(2,3,6)
H, xedges, yedges = np.histogram2d(y, x, 50)
H = gaussFilt(H, (3,3))
imshow(H, origin = 'lower', interpolation = 'nearest', aspect = 'auto')

savefig("../figures/fig1.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../figures/fig1.pdf &")

