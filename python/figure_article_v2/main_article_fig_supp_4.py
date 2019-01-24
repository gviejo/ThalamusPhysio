

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
	fig_height = fig_width*golden_mean*1.6            # height in inches
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


fig = figure(figsize = figsize(1))
gs = gridspec.GridSpec(4,3, wspace = 0.4, hspace = 0.7, height_ratios = [0.7, 1, 1, 1])


#########################################################################
# A PHASE PHASE SCATTER
#########################################################################
subplot(gs[0,0])
simpleaxis(gca())
# gca().set_aspect('equal')
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


# dist_cp = np.sqrt(np.sum(np.power(eigen[0] - eigen[1], 2))
theta_mod_toplot = allthetamodth.values[:,0].astype('float32')#,dist_cp>0.02]
phi_toplot = phi2.values.flatten()

r, p = corr_circular_(theta_mod_toplot, phi_toplot)
print(r, p)

x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
y = np.concatenate([phi_toplot, phi_toplot + 2*np.pi, phi_toplot, phi_toplot + 2*np.pi])
scatter(x, y, s = 1, alpha = 0.5)# c = np.tile(allzth.values[:,100],4), cmap = cm.get_cmap('viridis'), zorder = 2, alpha = 0.5)
# scatter(x, y, s = 0.8, c = np.tile(color_points,4), cmap = cm.get_cmap('hsv'), zorder = 2, alpha = )
xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi], ('0', '$\pi$', '$2\pi$', '$3\pi$', '$4\pi$'))
xlabel('Theta phase (rad.)', labelpad = 1.2)
ylabel('SWR jPCA phase (rad.)')
# gca().text(0.2, 0.9, r'$r = 0.18, p = 2.3 \times 10^{-7}$', transform = gca().transAxes, fontsize = 8, color = 'white')
gca().text(-0.0, 1.17, "A", transform = gca().transAxes, fontsize = 10)

######################################################################################
# B ORBIT MOUSE 17
######################################################################################
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
xs = [-0.4,0.35]
ys = [-0.3,0.25]
alljpc = dict()
pos = [0,2,3]
m = 'Mouse17'
########################################################################
# 1. Orbit
########################################################################
subplot(gs[0,1])
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

gradients = ['#37333f', '#536c7f', '#8fcdbd']

alljpc[m] = jpc
plot(jpc.iloc[0,0], jpc.iloc[0,1], 'o', markersize = 3, color = 'grey')
plot(jpc[0], jpc[1], linewidth = 0.8, color = 'grey')
arrow(jpc.iloc[-3,0],jpc.iloc[-3,1],jpc.iloc[-2,0]-jpc.iloc[-3,0],jpc.iloc[-2,1]-jpc.iloc[-3,1], color = 'grey', head_width = 0.06)
offsets = [-0.08,0.01,0.01]
for j in range(3): 
	plot(jpc.loc[toplot.loc[m,(j,'start')]:toplot.loc[m,(j,'end')],0], jpc.loc[toplot.loc[m,(j,'start')]:toplot.loc[m,(j,'end')],1], color = gradients[j], linewidth =2, alpha = 1)		
	tmp = toplot.loc[m,(j,'start')] + (toplot.loc[m,(j,'end')]-toplot.loc[m,(j,'start')])/2
	tmp = jpc.index.values[np.argmin(np.abs(tmp-jpc.index.values))]
	# text(jpc.loc[tmp,0], jpc.loc[tmp,1]+offsets[j], str(j), horizontalalignment = 'center', fontsize = 7)

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
text(-0.15, 1.17, 'B', transform = gca().transAxes, fontsize = 10)
title("Mouse 1")


########################################################################
# 2. SWR NUCLEUS
########################################################################
subplot(gs[0,2])
simpleaxis(gca())
for n in nucleus:
	plot(swr_nuc[m][n]['mean'], '-', label = n)
	# fill_between(times2, swr_nuc[m][n]['mean'].values - swr_nuc[m][n]['sem'].values, swr_nuc[m][n]['mean'].values + swr_nuc[m][n]['sem'], facecolor = 'grey', edgecolor = 'grey', alpha = 0.7)	
for j in range(3): 
	# axvline(toplot[m][j])
	axvspan(toplot.loc[m,(j,'start')], toplot.loc[m,(j,'end')], alpha = 0.5, color = gradients[j])
	# text(toplot.loc[m,(j,'start')] + (toplot.loc[m,(j,'end')]-toplot.loc[m,(j,'start')])/2, gca().get_ylim()[1], "T"+str(j), horizontalalignment = 'center')
ylabel("SWR")
xlabel("Time lag (ms)")
legend(frameon=False,loc = 'lower left', bbox_to_anchor=(-1.5,-0.5),handlelength=1,ncol = 4)

######################################################################################
# MAPS
######################################################################################
# gs.update(hspace = 0.2)


lbs = ['C', 'D', 'E']


 



alljpc = dict()
pos = [0,2,3]

for i, m in enumerate(['Mouse12', 'Mouse20', 'Mouse32']):
	gsm = gridspec.GridSpecFromSubplotSpec(2,4, subplot_spec = gs[i+1,:], height_ratios=[1,2])
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

	gradients = ['#37333f', '#536c7f', '#8fcdbd']

	alljpc[m] = jpc
	plot(jpc.iloc[0,0], jpc.iloc[0,1], 'o', markersize = 3, color = 'grey')
	plot(jpc[0], jpc[1], linewidth = 0.8, color = 'grey')
	arrow(jpc.iloc[-3,0],jpc.iloc[-3,1],jpc.iloc[-2,0]-jpc.iloc[-3,0],jpc.iloc[-2,1]-jpc.iloc[-3,1], color = 'grey', head_width = 0.06)
	offsets = [-0.08,0.01,0.01]
	for j in range(3): 
		plot(jpc.loc[toplot.loc[m,(j,'start')]:toplot.loc[m,(j,'end')],0], jpc.loc[toplot.loc[m,(j,'start')]:toplot.loc[m,(j,'end')],1], color = gradients[j], linewidth =2, alpha = 1)		
		tmp = toplot.loc[m,(j,'start')] + (toplot.loc[m,(j,'end')]-toplot.loc[m,(j,'start')])/2
		tmp = jpc.index.values[np.argmin(np.abs(tmp-jpc.index.values))]
		# text(jpc.loc[tmp,0], jpc.loc[tmp,1]+offsets[j], str(j), horizontalalignment = 'center', fontsize = 7)

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
	text(-0.15, 1.17, lbs[i], transform = gca().transAxes, fontsize = 10)
	title("Mouse "+str(i+2))


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
		axvspan(toplot.loc[m,(j,'start')], toplot.loc[m,(j,'end')], alpha = 0.5, color = gradients[j])
		# text(toplot.loc[m,(j,'start')] + (toplot.loc[m,(j,'end')]-toplot.loc[m,(j,'start')])/2, gca().get_ylim()[1], "T"+str(j), horizontalalignment = 'center')
	ylabel("SWR modulation")
	xlabel("Time lag (ms)")
	# if i == 0:
	# 	legend(frameon=False,loc = 'lower left', bbox_to_anchor=(1.1,-0.4),handlelength=1,ncol = len(nucleus)//2)

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
		# if j == 0: title("Mouse "+str(i+2))
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
		title(str(toplot.loc[m,(j,'start')])+r"ms $\rightarrow$ "+str(toplot.loc[m,(j,'end')])+"ms")

		#colorbar	
		cax = inset_axes(gca(), "4%", "20%",
		                   bbox_to_anchor=(0.8, 0.0, 1, 1),
		                   bbox_transform=gca().transAxes, 
		                   loc = 'lower left')
		if np.nanmax(rotated_image)<0.75:
			cb = colorbar(im, cax = cax, orientation = 'vertical', ticks = [0.15, 0.50])
		else:	
			cb = colorbar(im, cax = cax, orientation = 'vertical', ticks = [0.25, 0.75])
	

subplots_adjust(top = 0.95, bottom = 0.09, left = 0.1, right = 0.95, hspace = 0.4)

savefig("../../figures/figures_articles_v2/figart_supp_4.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v2/figart_supp_4.pdf &")

