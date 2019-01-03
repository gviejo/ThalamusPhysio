

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


###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.3           # height in inches
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


fig = figure(figsize = figsize(1), tight_layout=True)
gs = gridspec.GridSpec(2,1, wspace = 0.4, hspace = 0.2, left = 0.08, right = 0.96, top = 0.96, bottom = 0.08)
# gs.update(hspace = 0.2)
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
gsm = gridspec.GridSpecFromSubplotSpec(2,4, subplot_spec = gs[i,:], height_ratios=[1,2])
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
arrow(jpc.iloc[-3,0],jpc.iloc[-3,1],jpc.iloc[-2,0]-jpc.iloc[-3,0],jpc.iloc[-2,1]-jpc.iloc[-3,1], color = 'grey', head_width = 0.02)
offsets = [-0.08,0.01,0.01]
for j in range(3): 
	plot(jpc.loc[toplot.loc[m,(j,'start')]:toplot.loc[m,(j,'end')],0], jpc.loc[toplot.loc[m,(j,'start')]:toplot.loc[m,(j,'end')],1], linewidth = 2.0, color = gradients[j], alpha = 1.0)		
	tmp = toplot.loc[m,(j,'start')] + (toplot.loc[m,(j,'end')]-toplot.loc[m,(j,'start')])/2
	# text(jpc.loc[tmp,0], jpc.loc[tmp,1]+offsets[j], str(j), horizontalalignment = 'center')

gca().spines['left'].set_bounds(ys[0]+0.1,ys[0]+0.2)
gca().spines['bottom'].set_bounds(ys[0]+0.1,ys[0]+0.2)
xticks([], [])
yticks([], [])	
# xlim(xs[0], xs[1])
# ylim(ys[0], ys[1])
text(-0.3, 1.01, lbs[0], transform = gca().transAxes, fontsize = 10)



########################################################################
# 2. SWR NUCLEUS
########################################################################
subplot(gsm[1,0])
simpleaxis(gca())
for j in range(3): 
	# axvline(toplot[m][j])
	axvspan(toplot.loc[m,(j,'start')], toplot.loc[m,(j,'end')], alpha = 0.5, color = gradients[j])
	
for n in nucleus:
	plot(swr_nuc[m][n]['mean'], '-', label = n)
	# fill_between(times2, swr_nuc[m][n]['mean'].values - swr_nuc[m][n]['sem'].values, swr_nuc[m][n]['mean'].values + swr_nuc[m][n]['sem'], facecolor = 'grey', edgecolor = 'grey', alpha = 0.7)	
ylabel("SWRs modulation", labelpad = -0.1)
xlabel("Time lag (ms)")
legend(frameon=False,loc = 'lower left', bbox_to_anchor=(1.1,-0.2),handlelength=1,ncol = 4)
for j in range(3): 
	text(toplot.loc[m,(j,'start')] + (toplot.loc[m,(j,'end')]-toplot.loc[m,(j,'start')])/2, gca().get_ylim()[1], "T"+str(j), horizontalalignment = 'center')
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
	# if j == 0: title("Mouse "+str(i+1))	
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
	title("T"+str(j)+': '+str(toplot.loc[m,(j,'start')])+"ms -> "+str(toplot.loc[m,(j,'end')])+"ms")

	#colorbar	
	cax = inset_axes(gca(), "4%", "20%",
	                   bbox_to_anchor=(0.8, 0.0, 1, 1),
	                   bbox_transform=gca().transAxes, 
	                   loc = 'lower left')
	cb = colorbar(im, cax = cax, orientation = 'vertical', ticks = [0.25, 0.75])
	# cb.set_label('Burstiness', labelpad = -4)
	# cb.ax.xaxis.set_tick_params(pad = 1)
	# cax.set_title("Cluster 2", fontsize = 6, pad = 2.5)



########################################################################
# B. ALL MOUSE
########################################################################		
gsm = gridspec.GridSpecFromSubplotSpec(len(nucleus)+1,2, subplot_spec = gs[1,:], height_ratios = [1]+[0.1]*len(nucleus))

idm = swr_all.idxmax()
idm = idm.sort_values()
order = idm.index.values

subplot(gsm[0,0])
simpleaxis(gca())
for n in nucleus:
	plot(swr_all[n])
xticks([], [])
text(-0.15, 1.1, lbs[1], transform = gca().transAxes, fontsize = 10)
ylabel("SWRs modulation", labelpad = -0.1)
xlim(-500,500)

######################################################################
for i, n in enumerate(order):
	subplot(gsm[i+1,0])
	tmp = swr_nuc.xs(n,1,1).xs('mean',1,1).T.values.astype(np.float32)
	imshow(tmp, aspect = 'auto', cmap = 'bwr')
	if i == len(nucleus)-1:	
		xticks([0,100,200],[-500,0,500])		
		xlabel("Time lag (ms)")	
	else:
		xticks([], [])
	# if i == 0:
	# 	# yticks([0,3], [1,4])
	# 	ylabel(n, rotation = 0, labelpad = 8, y = 0.2)
	# else:
	yticks([0,3], ['',''])
	ylabel(n, rotation = 0, labelpad = 11, y = 0.2)
	if i == len(order)-1:
		annotate("Mouse", 		(1.08, 2.4), (1.04,  2.7 ), xycoords = 'axes fraction', fontsize = 7)
		annotate("1", 		(1, 0.75+0.125), (1.08,  1.6 ), xycoords = 'axes fraction', fontsize = 7, arrowprops = {'arrowstyle':'-'})
		annotate("2", 		(1, 0.50+0.125), (1.08,  0.8), xycoords = 'axes fraction', 	fontsize = 7, arrowprops = {'arrowstyle':'-'})
		annotate("3", 		(1, 0.25+0.125), (1.08,  0.0), xycoords = 'axes fraction', 	fontsize = 7, arrowprops = {'arrowstyle':'-'})
		annotate("4", 		(1, 0.0 +0.125), (1.08, -0.8 ), xycoords = 'axes fraction', fontsize = 7, arrowprops = {'arrowstyle':'-'})

	
		
####################################################################
# C. ORBIT ALL MOUSE
####################################################################
subplot(gsm[:,1])
simpleaxis(gca())

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
# gca().spines['left'].set_bounds(np.min(jX[:,1]), np.min(jX[:,1]+0.1))
# gca().spines['bottom'].set_bounds(np.min(jX[:,0]), np.min(jX[:,0]+0.1))
gca().spines['left'].set_visible(False)
gca().spines['bottom'].set_visible(False)
xticks([], [])
yticks([], [])
# gca().xaxis.set_label_coords(0.15, -0.02)
# gca().yaxis.set_label_coords(-0.02, 0.15)
# ylabel('jPC2')
# xlabel('jPC1')
text(-0.1, 1.02, "C", transform = gca().transAxes, fontsize = 10)

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


# threshold = swr_all.mean() + (swr_all.max() - swr_all.mean())/2
# phase = swr_all - threshold
# phase[phase < 0.0] = np.nan
# # putting nucleus name
# ct = 0
# for n in order:
# 	# excitatory phase
# 	text(jpca.loc[idm[n],0],jpca.loc[idm[n],1],n, ha = 'center', va = 'center')
# 	tx = phase[n].dropna().index
# 	if np.max(np.diff(tx)) > 5.:
# 		tx1 = tx[0:np.argmax(np.diff(tx))+1]
# 		tx2 = tx[np.argmax(np.diff(tx))+1:]
# 		plot(jpca.loc[tx1, 0]+np.sign(jpca.loc[tx1, 0])*ct, jpca.loc[tx1, 1]+np.sign(jpca.loc[tx1, 1])*ct, linewidth = 2)
# 		plot(jpca.loc[tx2, 0]+np.sign(jpca.loc[tx2, 0])*ct, jpca.loc[tx2, 1]+np.sign(jpca.loc[tx2, 1])*ct, linewidth = 2)
# 	else:
# 		plot(jpca.loc[tx, 0]+np.sign(jpca.loc[tx, 0])*ct, jpca.loc[tx, 1]+np.sign(jpca.loc[tx, 1])*ct, linewidth = 2)
# 	ct+=0.0018

subplots_adjust(bottom = 0.01, top = 0.97, right = 0.98, left = 0.03)

savefig("../../figures/figures_articles/figart_5.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_5.pdf &")

