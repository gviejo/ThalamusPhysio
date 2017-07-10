

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

allz = []
allthetamod = []

for session in datasets:
	###############################################################################################################
	# GENERAL INFO
	###############################################################################################################
	generalinfo = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/GeneralInfo.mat')

	###############################################################################################################
	# SHANK INFO
	###############################################################################################################
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]
		else :
			shankStructure[k[0]] = []

	spikedata = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/SpikeData.mat')
	shank = spikedata['shank']

	shankIndex = np.where(shank == shankStructure['thalamus'])[0]

	###############################################################################################################
	# THETA 
	###############################################################################################################
	thetaInfo = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/ThetaInfo.mat')
	thMod = thetaInfo['thetaModREM'][shankIndex,:] # all neurons vs 
	allthetamod.append(thMod[shankIndex,:])
	###############################################################################################################
	# RIPPLE INFO
	###############################################################################################################
	RipInfo = scipy.io.loadmat(data_directory+'/'+session+'/Analysis/RipInfo.mat')
	H0All = RipInfo['H0All']
	HmAll = RipInfo['HmAll']
	HsdAll = RipInfo['HsdAll']

	z = (H0All - HmAll) / HsdAll
	z = gaussFilt(z[:,shankIndex],(2,1))

	allz.append(z.transpose())
	###############################################################################################################
	# CHANNEL STRUCTURE
	###############################################################################################################
	channelStructure = {}
	for k,i in zip(generalinfo['channelStructure'][0][0][0][0],range(len(generalinfo['channelStructure'][0][0][0][0]))):
		if len(generalinfo['channelStructure'][0][0][1][0][i]):
			channelStructure[k[0]] = generalinfo['channelStructure'][0][0][1][0][i][0][0]
		else:
			channelStructure[k[0]] = []



allz = np.concatenate(allz, axis = 0)
allthetamod = np.concatenate(allthetamod, axis = 0)

# CHECK FOR NAN
tmp1 = np.unique(np.where(np.isnan(allz))[0])
tmp2 = np.unique(np.where(np.isnan(allthetamod[:,0]))[0])
tmp3 = np.unique(np.where(allthetamod[:,1]>0.05))
tmp = np.unique(np.concatenate([tmp1,tmp2,tmp3]))

if len(tmp):
	allzth = np.delete(allz, tmp, axis = 0)
	allthetamodth = np.delete(allthetamod, tmp, axis = 0)
# allz = allz[np.argsort(np.max(allz, axis = 1))]
# allzth = allzth.transpose()
# allthetamodth = allthetamodth.transpose()

###############################################################################################################
# PCA
###############################################################################################################
pca = PCA(n_components = 10)
zpca = pca.fit_transform(allzth)
pc = zpca[:,0:2]
eigen = pca.components_

phi = np.mod(np.arctan2(zpca[:,0], zpca[:,1]), 2*np.pi)

times = RipInfo['Tr'].flatten()

###############################################################################################################
# jPCA
###############################################################################################################
from scipy.sparse import lil_matrix

X = pca.components_.transpose()
dX = np.hstack([np.vstack(derivative(times, X[:,i])) for i in range(X.shape[1])])
#build the H mapping for a given n
# work by lines but H is made for column based
n = X.shape[1]
M = np.zeros((n,n), dtype = np.int)
M[np.triu_indices(n,1)] = np.arange(1,int(n*(n-1)/2)+1)
M = M - M.transpose()
m = np.vstack(M.reshape(n*n))
k = np.vstack(M[np.triu_indices(n,1)]).astype('int')
H = lil_matrix( (len(m), len(k)), dtype = np.float16)
H = np.zeros( (len(m), len(k) ))
# first column 
for i in k.flatten():
	# positive
	H[np.where(m == i)[0][0],i-1] = 1.0
	# negative
	H[np.where(m == -i)[0][0],i-1] = -1.0

# X tilde
Xtilde = np.zeros( (X.shape[0]*X.shape[1], X.shape[1]*X.shape[1]) )
for i, j in zip( (np.arange(0,n**2,n) ), np.arange(0, n*X.shape[0], X.shape[0]) ):
	Xtilde[j:j+X.shape[0],i:i+X.shape[1]] = X
# put dx in columns
dXv = np.vstack(dX.transpose().reshape(X.shape[0]*X.shape[1]))
# multiply Xtilde by H
XtH = np.dot(Xtilde, H)
# solve XtH k = dXv
k, residuals, rank, s = np.linalg.lstsq(XtH, dXv)
# multiply by H to get m then M
m = np.dot(H, k)
Mskew = m.reshape(n,n).transpose()
# Contruct the two vectors for projection
evalues, evectors = np.linalg.eig(Mskew)
index = np.argsort(evalues).reshape(5,2)[:,1]
evectors = evectors.transpose().reshape(5,2,10)
u = np.vstack([
				np.real(evectors[index[-1]][0] + evectors[index[-1]][1]),
				np.imag(evectors[index[-1]][0] - evectors[index[-1]][1])
			]).transpose()
# PRoject X
rX = np.dot(X, u)





###############################################################################################################
# PLOT
###############################################################################################################

import matplotlib.cm as cm
figure()
subplot(2,3,1)
imshow(allzth, aspect = 'auto')

subplot(2,3,2)
plot(times, allzth.transpose())

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
show()



