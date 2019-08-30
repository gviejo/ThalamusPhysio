import numpy as np
from multiprocessing import Pool, cpu_count
from scipy import stats, linalg
import pandas as pd
import neuroseries as nts
from scipy.fftpack import *
from numba import jit

def jPCA(data, times):
	#PCA
	from sklearn.decomposition import PCA
	n 		= 6
	pca 	= PCA(n_components = n)
	zpca 	= pca.fit_transform(data)
	pc 		= zpca[:,0:2]
	eigen 	= pca.components_
	phi 	= np.mod(np.arctan2(zpca[:,1], zpca[:,0]), 2*np.pi)
	#jPCA
	X 		= pca.components_.transpose()
	dX 		= np.hstack([np.vstack(derivative(times, X[:,i])) for i in range(X.shape[1])])
	#build the H mapping for a given n
	# work by lines but H is made for column based
	n 		= X.shape[1]
	H 		= buildHMap(n)
	# X tilde
	Xtilde 	= np.zeros( (X.shape[0]*X.shape[1], X.shape[1]*X.shape[1]) )
	for i, j in zip( (np.arange(0,n**2,n) ), np.arange(0, n*X.shape[0], X.shape[0]) ):
		Xtilde[j:j+X.shape[0],i:i+X.shape[1]] = X
	# put dx in columns
	dXv 	= np.vstack(dX.transpose().reshape(X.shape[0]*X.shape[1]))
	# multiply Xtilde by H
	XtH 	= np.dot(Xtilde, H)
	# solve XtH k = dXv
	k, residuals, rank, s = np.linalg.lstsq(XtH, dXv)
	# multiply by H to get m then M
	m 		= np.dot(H, k)
	Mskew = m.reshape(n,n).transpose()
	# Contruct the two vectors for projection with MSKEW
	evalues, evectors = np.linalg.eig(Mskew)
	# index = np.argsort(evalues).reshape(5,2)[:,1]
	index 	= np.argsort(np.array([np.linalg.norm(i) for i in evalues]).reshape(int(n/2),2)[:,0])
	evectors = evectors.transpose().reshape(int(n/2),2,n)
	u 		= np.vstack([
					np.real(evectors[index[-1]][0] + evectors[index[-1]][1]),
					np.imag(evectors[index[-1]][0] - evectors[index[-1]][1])
				]).transpose()
	# PRoject X
	rX 		= np.dot(X, u)
	rX 		= rX*-1.0
	score 	= np.dot(data, rX)
	phi2 	= np.mod(np.arctan2(score[:,1], score[:,0]), 2*np.pi)
	# Construct the two vectors for projection with MSYM
	Msym 	= np.copy(Mskew)
	Msym[np.triu_indices(n)] *= -1.0
	evalues2, evectors2 = np.linalg.eig(Msym)
	v 		= evectors2[:,0:2]
	rY 		= np.dot(X, v)
	score2 	= np.dot(data, rY)
	phi3 	= np.mod(np.arctan2(score2[:,1], score2[:,0]), 2*np.pi)
	dynamical_system = {	'x'		:	X,
							'dx'	:	dX,
							'Mskew'	:	Mskew,
							'Msym'	:	Msym,
														}
	return (rX, phi2, dynamical_system)

def gaussFilt(X, wdim = (1,)):
	'''
		Gaussian Filtering in 1 or 2d.		
		Made to fit matlab
	'''
	from scipy.signal import gaussian

	if len(wdim) == 1:
		from scipy.ndimage.filters import convolve1d
		l1 = len(X)
		N1 = wdim[0]*10
		S1 = (N1-1)/float(2*5)
		gw = gaussian(N1, S1)
		gw = gw/gw.sum()
		#convolution
		if len(X.shape) == 2:
			filtered_X = convolve1d(X, gw, axis = 1)
		elif len(X.shape) == 1:
			filtered_X = convolve1d(X, gw)
		return filtered_X	
	elif len(wdim) == 2:
		from scipy.signal import convolve2d
		def conv2(x, y, mode='same'):
			return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)			

		l1, l2 = X.shape
		N1, N2 = wdim		
		# create bordered matrix
		Xf = np.flipud(X)
		bordered_X = np.vstack([
				np.hstack([
					np.fliplr(Xf),Xf,np.fliplr(Xf)
				]),
				np.hstack([
					np.fliplr(X),X,np.fliplr(X)
				]),
				np.hstack([
					np.fliplr(Xf),Xf,np.fliplr(Xf)
				]),
			])
		# gaussian windows
		N1 = N1*10
		N2 = N2*10
		S1 = (N1-1)/float(2*5)
		S2 = (N2-1)/float(2*5)
		gw = np.vstack(gaussian(N1,S1))*gaussian(N2,S2)
		gw = gw/gw.sum()
		# convolution
		filtered_X = conv2(bordered_X, gw, mode ='same')
		return filtered_X[l1:l1+l1,l2:l2+l2]
	else :
		print("Error, dimensions larger than 2")
		return

def derivative(x, f):
	''' 
		Compute the derivative of a time serie
		Used for jPCA
	'''
	from scipy.stats import linregress
	fish = np.zeros(len(f))
	slopes_ = []
	tmpf = np.hstack((f[0],f,f[-1])) # not circular
	binsize = x[1]-x[0]	
	tmpx = np.hstack((np.array([x[0]-binsize]),x,np.array([x[-1]+binsize])))	
	# plot(tmpx, tmpf, 'o')
	# plot(x, f, '+')
	for i in range(len(f)):
		slope, intercept, r_value, p_value, std_err = linregress(tmpx[i:i+3], tmpf[i:i+3])
		slopes_.append(slope)	
		# plot(tmpx[i:i+3], tmpx[i:i+3]*slope+intercept, '-')
	return np.array(slopes_)/binsize

def buildHMap(n, ):
	'''
		build the H mapping for a given n
		used for the jPCA
	'''
	from scipy.sparse import lil_matrix
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
	return H

def crossValidation(data, times, n_cv = 10, dims = (6,2)):
	''' 
		Perform a randomized cross-validation 
		of PCA -> jPCA
		dims = (dimension reduction of pca, dimension reduction of jpca)
	'''
	from sklearn.model_selection import KFold
	from sklearn.decomposition import PCA
	cv_kf = KFold(n_splits = n_cv, shuffle = True, random_state=42)
	skf = cv_kf.split(data)
	scorecv = np.zeros( (data.shape[0],2) )
	phicv = np.zeros(data.shape[0])
	n = dims[0]
	for idx_r, idx_t in skf:
		Xtrain = data[idx_r, :]
		Xtest  = data[idx_t, :]		
		pca = PCA(n_components = n)
		zpca = pca.fit_transform(Xtrain)
		X = pca.components_.transpose()
		dX = np.hstack([np.vstack(derivative(times, X[:,i])) for i in range(X.shape[1])])
		#build the H mapping for a given n
		# work by lines but H is made for column based
		n = X.shape[1]
		H = buildHMap(n)
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
		# Contruct the two vectors for projection with MSKEW
		evalues, evectors = np.linalg.eig(Mskew)
		# index = np.argsort(evalues).reshape(5,2)[:,1]
		index = np.argsort(np.array([np.linalg.norm(i) for i in evalues]).reshape(int(n/2),2)[:,0])
		evectors = evectors.transpose().reshape(int(n/2),2,n)
		u = np.vstack([
						np.real(evectors[index[-1]][0] + evectors[index[-1]][1]),
						np.imag(evectors[index[-1]][0] - evectors[index[-1]][1])
					]).transpose()
		# PRoject X
		rX = np.dot(X, u)
		score = np.dot(Xtest, rX)
		phi = np.mod(np.arctan2(score[:,0], score[:,1]), 2*np.pi)
		scorecv[idx_t,:] = score
		phicv[idx_t] = phi

	return np.array(scorecv), np.array(phicv)

def quartiles(data, times, n_fold = 4, dims = (6,2)):
	''' 		
		PCA -> jPCA
		dims = (dimension reduction of pca, dimension reduction of jpca)
	'''	
	from sklearn.decomposition import PCA
	indexdata = np.linspace(0,len(data),n_fold+1).astype('int')	
	scorecv = []
	phicv = []
	n = dims[0]
	jpca = []
	for i in range(n_fold):
		Xtrain = data[indexdata[i]:indexdata[i+1], :]		
		pca = PCA(n_components = n)
		zpca = pca.fit_transform(Xtrain)
		X = pca.components_.transpose()
		dX = np.hstack([np.vstack(derivative(times, X[:,i])) for i in range(X.shape[1])])
		#build the H mapping for a given n
		# work by lines but H is made for column based
		n = X.shape[1]
		H = buildHMap(n)
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
		# Contruct the two vectors for projection with MSKEW
		evalues, evectors = np.linalg.eig(Mskew)
		# index = np.argsort(evalues).reshape(5,2)[:,1]
		index = np.argsort(np.array([np.linalg.norm(i) for i in evalues]).reshape(int(n/2),2)[:,0])
		evectors = evectors.transpose().reshape(int(n/2),2,n)
		u = np.vstack([
						np.real(evectors[index[-1]][0] + evectors[index[-1]][1]),
						np.imag(evectors[index[-1]][0] - evectors[index[-1]][1])
					]).transpose()
		# PRoject X
		rX = np.dot(X, u)
		score = np.dot(Xtrain, rX)
		phi = np.mod(np.arctan2(score[:,1], score[:,0]), 2*np.pi)
		scorecv.append(score)
		phicv.append(phi)
		jpca.append(rX)

	index = np.array([np.arange(indexdata[i],indexdata[i+1]) for i in range(len(indexdata)-1)])
	return np.array(scorecv), np.array(phicv), index, np.array(jpca)

def downsample(tsd, up, down):
	import scipy.signal
	import neuroseries as nts
	dtsd = scipy.signal.resample_poly(tsd.values, up, down)
	dt = tsd.as_units('s').index.values[np.arange(0, tsd.shape[0], down)]
	if len(tsd.shape) == 1:		
		return nts.Tsd(dt, dtsd, time_units = 's')
	elif len(tsd.shape) == 2:
		return nts.TsdFrame(dt, dtsd, time_units = 's', columns = list(tsd.columns))

def getPhase(lfp, fmin, fmax, nbins, fsamp, power = False):
	""" Continuous Wavelets Transform
		return phase of lfp in a Tsd array
	"""
	import neuroseries as nts
	from Wavelets import MyMorlet as Morlet
	if isinstance(lfp, nts.time_series.TsdFrame):
		allphase 		= nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
		allpwr 			= nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
		for i in lfp.keys():
			allphase[i], allpwr[i] = getPhase(lfp[i], fmin, fmax, nbins, fsamp, power = True)
		if power:
			return allphase, allpwr
		else:
			return allphase			

	elif isinstance(lfp, nts.time_series.Tsd):
		cw 				= Morlet(lfp.values, fmin, fmax, nbins, fsamp)
		cwt 			= cw.getdata()
		cwt 			= np.flip(cwt, axis = 0)
		wave 			= np.abs(cwt)**2.0
		phases 			= np.arctan2(np.imag(cwt), np.real(cwt)).transpose()	
		cwt 			= None
		index 			= np.argmax(wave, 0)
		# memory problem here, need to loop
		phase 			= np.zeros(len(index))	
		for i in range(len(index)) : phase[i] = phases[i,index[i]]
		phases 			= None
		if power: 
			pwrs 		= cw.getpower()		
			pwr 		= np.zeros(len(index))		
			for i in range(len(index)):
				pwr[i] = pwrs[index[i],i]	
			return nts.Tsd(lfp.index.values, phase), nts.Tsd(lfp.index.values, pwr)
		else:
			return nts.Tsd(lfp.index.values, phase)

def getPeaksandTroughs(lfp, min_points):
	"""	 
		At 250Hz (1250/5), 2 troughs cannont be closer than 20 (min_points) points (if theta reaches 12Hz);		
	"""
	import neuroseries as nts
	import scipy.signal
	if isinstance(lfp, nts.time_series.Tsd):
		troughs 		= nts.Tsd(lfp.as_series().iloc[scipy.signal.argrelmin(lfp.values, order =min_points)[0]], time_units = 'us')
		peaks 			= nts.Tsd(lfp.as_series().iloc[scipy.signal.argrelmax(lfp.values, order =min_points)[0]], time_units = 'us')
		tmp 			= nts.Tsd(troughs.realign(peaks, align = 'next').as_series().drop_duplicates('first')) # eliminate double peaks
		peaks			= peaks[tmp.index]
		tmp 			= nts.Tsd(peaks.realign(troughs, align = 'prev').as_series().drop_duplicates('first')) # eliminate double troughs
		troughs 		= troughs[tmp.index]
		return (peaks, troughs)
	elif isinstance(lfp, nts.time_series.TsdFrame):
		peaks 			= nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
		troughs			= nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
		for i in lfp.keys():
			peaks[i], troughs[i] = getPeaksandTroughs(lfp[i], min_points)
		return (peaks, troughs)

def getCircularMean(theta, high = np.pi, low = -np.pi):
	""" 
	see CircularMean.m in TSToolbox_Utils/Stats/CircularMean.m
	or Fisher N.I. Analysis of Circular Data p. 30-35
	"""
	from scipy.stats import circmean
	n 			= float(len(theta))
	alpha 		= 0.05	
	S 			= np.sum(np.sin(theta))
	C 			= np.sum(np.cos(theta))
	mu 			= circmean(theta, high, low)
	Rmean		= np.sqrt(S**2.0 + C**2.0) / n
	rho2 		= np.sum(np.cos(2*(theta - mu)))/n
	delta		= (1 - rho2) / (2* Rmean**2)
	sigma		= np.sqrt(-2.0*np.log(Rmean))
	Z 			= n * Rmean**2.0
	pval 		= np.exp(-Z)*(1+(2*Z-Z**2.0)/(4*n) - (24*Z - 132*Z**2.0 + 76*Z**3.0 - 9*Z**4.0)/(288.0*n**2.0))
	if Rmean < 0.5:
		Kappa 	= 2.0*Rmean + Rmean**3.0 + 5*(Rmean**5.0)/6.0
	elif Rmean >= 0.53 and Rmean < 0.85:
		Kappa 	= -0.4 + 1.39*Rmean + 0.43/(1-Rmean)
	else:
		Kappa 	= 1/(Rmean**3.0 - 4*Rmean**2.0 + 3*Rmean)
	if n < 15:
		if Kappa < 2:
			Kappa = np.max([Kappa-2/(n*Kappa),0])
		else:
			Kappa = (((n-1)**3) * Kappa)/(n**3 + n)
	return (mu, Kappa, pval)

def butter_bandpass(lowcut, highcut, fs, order=5):
	from scipy.signal import butter
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	from scipy.signal import lfilter
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def getFiringRate(tsd_spike, bins):
	"""bins shoud be in us
	"""
	import neuroseries as nts
	frate 		= nts.Tsd(bins, np.zeros(len(bins)))
	bins_size 	= (bins[1] - bins[0])*1.e-6 # convert to s for Hz
	if type(tsd_spike) is dict:
		for n in tsd_spike.keys():
			index = np.digitize(tsd_spike[n].index.values, bins)
			for i in index:
				frate[bins[i]] += 1.0
		frate = nts.Tsd(bins+(bins[1]-bins[0])/2, frate.values/len(tsd_spike)/bins_size)
		return frate
	else:
		index 	= np.digitize(tsd_spike.index.values, bins)
		for i in index:
			frate[bins[i]] += 1.0
		frate = nts.Tsd(bins+(bins[1]-bins[0])/2, frate.values/bins_size)
		return frate	

def computePhaseModulation(phase, spikes, ep, get_phase = False):
	n_neuron 		= len(spikes)
	spikes_evt		= {n:spikes[n].restrict(ep) for n in spikes.keys()}
	spikes_phase	= {n:phase.realign(spikes_evt[n], align = 'closest') for n in spikes_evt.keys()}
	evt_mod 		= np.ones((n_neuron,3))*np.nan	
	for n in range(len(spikes_phase.keys())):
		neuron = list(spikes_phase.keys())[n]
		ph = spikes_phase[neuron]
		mu, kappa, pval = getCircularMean(ph.values)
		evt_mod[n] = np.array([mu, pval, kappa])
	if get_phase:
		return evt_mod, spikes_phase
	else:
		return evt_mod
	
def getPhaseCoherence(phase):
	r = np.sqrt(np.sum([np.power(np.sum(np.cos(phase)),2), np.power(np.sum(np.sin(phase)),2)]))/len(phase)	
	return r

def computeBurstiness(spikes, epoch, start = 2.0, stop = 8.0):
	s = spikes.restrict(epoch)
	dt = np.diff(s.as_units('ms').index.values)
	Nobs = float(np.logical_and(dt>start, dt<stop).sum())
	r = len(s)/epoch.tot_length('s')
	Nthe = len(dt)*(np.exp(-r*(start/1000)) - np.exp(-r*(stop/1000)))
	return Nobs/(Nthe+1.0)

#########################################################
# INTERPOLATION
#########################################################
def interpolate(z, x, y, inter, bbox = None):	
	import scipy.interpolate
	xnew = np.arange(x.min(), x.max()+inter, inter)
	ynew = np.arange(y.min(), y.max()+inter, inter)
	if bbox == None:
		f = scipy.interpolate.RectBivariateSpline(y, x, z)
	else:
		f = scipy.interpolate.RectBivariateSpline(y, x, z, bbox = bbox)
	znew = f(ynew, xnew)
	return (xnew, ynew, znew)

def filter_(z, n):
	from scipy.ndimage import gaussian_filter	
	return gaussian_filter(z, n)

def softmax(x, b1 = 20.0, b2 = 0.5):
	# x -= x.min()
	# x /= x.max()
	# return 0.75*(1.0/(1.0+np.exp(-(x-b2)*b1)))+0.25
	return 1.0/(1.0+np.exp(-(x-b2)*b1))

def get_rgb(mapH, mapS, mapV, bound):
	from matplotlib.colors import hsv_to_rgb	
	"""
		1. convert mapH to x between -1 and 1
		2. get y value between 0 and 1 -> mapV
		3. rescale mapH between 0 and 0.6
		4. normalize mapS
	"""		
	# x = mapH.copy() * 2.0
	# x = x - 1.0
	# y = 1.0 - 0.4*x**6.0
	# mapV = y.copy()
	H 	= (1-mapH)*bound
	S 	= mapS	
	V 	= mapV
	HSV = np.dstack((H,S,V))	
	RGB = hsv_to_rgb(HSV)	
	return RGB

def rotateImage(image, angle):
	import cv2
	row,col = image.shape
	center=tuple(np.array([row,col])/2)
	rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
	new_image = cv2.warpAffine(image, rot_mat, (col,row))
	return new_image

def getRotationMatrix(shape, angle):
	import cv2
	row,col = shape
	center=tuple(np.array([row,col])/2)
	rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
	return rot_mat

def getXYshapeofRotatedMatrix(x, y, angle):	
	return (x * np.cos(angle * np.pi /180.) + y * np.sin(angle * np.pi / 180.),
			y * np.cos(angle * np.pi /180.) + x * np.sin(angle * np.pi / 180.)			)


#########################################################
# CORRELATION
#########################################################
def crossCorr(t1, t2, binsize, nbins):
	''' 
		Fast crossCorr 
	'''
	nt1 = len(t1)
	nt2 = len(t2)
	if np.floor(nbins/2)*2 == nbins:
		nbins = nbins+1

	m = -binsize*((nbins+1)/2)
	B = np.zeros(nbins)
	for j in range(nbins):
		B[j] = m+j*binsize

	w = ((nbins/2) * binsize)
	C = np.zeros(nbins)
	i2 = 1

	for i1 in range(nt1):
		lbound = t1[i1] - w
		while i2 < nt2 and t2[i2] < lbound:
			i2 = i2+1
		while i2 > 1 and t2[i2-1] > lbound:
			i2 = i2-1

		rbound = lbound
		l = i2
		for j in range(nbins):
			k = 0
			rbound = rbound+binsize
			while l < nt2 and t2[l] < rbound:
				l = l+1
				k = k+1

			C[j] += k

	# for j in range(nbins):
	# C[j] = C[j] / (nt1 * binsize)
	C = C/(nt1 * binsize/1000)

	return C

def crossCorr2(t1, t2, binsize, nbins):
	'''
		Slow crossCorr
	'''
	window = np.arange(-binsize*(nbins/2),binsize*(nbins/2)+2*binsize,binsize) - (binsize/2.)
	allcount = np.zeros(nbins+1)
	for e in t1:
		mwind = window + e
		# need to add a zero bin and an infinite bin in mwind
		mwind = np.array([-1.0] + list(mwind) + [np.max([t1.max(),t2.max()])+binsize])	
		index = np.digitize(t2, mwind)
		# index larger than 2 and lower than mwind.shape[0]-1
		# count each occurences 
		count = np.array([np.sum(index == i) for i in range(2,mwind.shape[0]-1)])
		allcount += np.array(count)
	allcount = allcount/(float(len(t1))*binsize / 1000)
	return allcount

def xcrossCorr_slow(t1, t2, binsize, nbins, nbiter, jitter, confInt):		
	times 			= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	H0 				= crossCorr(t1, t2, binsize, nbins)	
	H1 				= np.zeros((nbiter,nbins+1))
	t2j	 			= t2 + 2*jitter*(np.random.rand(nbiter, len(t2)) - 0.5)
	t2j 			= np.sort(t2j, 1)
	for i in range(nbiter):			
		H1[i] 		= crossCorr(t1, t2j[i], binsize, nbins)
	Hm 				= H1.mean(0)
	tmp 			= np.sort(H1, 0)
	HeI 			= tmp[int((1-confInt)/2*nbiter),:]
	HeS 			= tmp[int((confInt + (1-confInt)/2)*nbiter)]
	Hstd 			= np.std(tmp, 0)

	return (H0, Hm, HeI, HeS, Hstd, times)

def xcrossCorr_fast(t1, t2, binsize, nbins, nbiter, jitter, confInt):		
	times 			= np.arange(0, binsize*(nbins*2+1), binsize) - (nbins*2*binsize)/2
	# need to do a cross-corr of double size to convolve after and avoid boundary effect
	H0 				= crossCorr(t1, t2, binsize, nbins*2)	
	window_size 	= 2*jitter//binsize
	window 			= np.ones(int(window_size))*(1/window_size)
	Hm 				= np.convolve(H0, window, 'same')
	Hstd			= np.sqrt(np.var(Hm))	
	HeI 			= np.NaN
	HeS 			= np.NaN	
	return (H0, Hm, HeI, HeS, Hstd, times)	

def corr_circular_(alpha1, alpha2):
	axis = None
	from pycircstat import center
	from scipy.stats import norm
	alpha1_bar, alpha2_bar = center(alpha1, alpha2, axis=axis)
	num = np.sum(np.sin(alpha1_bar) * np.sin(alpha2_bar), axis=axis)
	den = np.sqrt(np.sum(np.sin(alpha1_bar) ** 2, axis=axis) * np.sum(np.sin(alpha2_bar) ** 2, axis=axis))
	rho = num/den
	# pvalue
	l20 = np.mean(np.sin(alpha1 - alpha1_bar)**2)
	l02 = np.mean(np.sin(alpha2 - alpha2_bar)**2)
	l22 = np.mean((np.sin(alpha1-alpha1_bar)**2) * (np.sin(alpha2 - alpha2_bar)**2))
	ts = np.sqrt((float(len(alpha1)) * l20 * l02)/l22) * rho
	pval = 2.0 * (1.0 - norm.cdf(np.abs(ts)))

	return rho, pval

def autocorr(tsd):
	import pandas as pd
	# bin_size 	= 0.5 # ms 
	# nb_bins 	= 1000
	bin_size 	= 1
	nb_bins 	= 10000	
	# confInt 	= 0.95
	# nb_iter 	= 1000
	# jitter  	= 150 # ms					
	# H0, Hm, HeI, HeS, Hstd, times = xcrossCorr_fast(tsd, tsd, bin_size, nb_bins, nb_iter, jitter, confInt)
	C = crossCorr(tsd, tsd, bin_size, nb_bins)
	times 		= np.arange(0, bin_size*(nb_bins+1), bin_size) - (nb_bins*bin_size)/2
	return pd.Series(index = times, data = C)
	# return (H0 - Hm)/Hstd

#########################################################
# WRAPPERS
#########################################################
def loadWaveforms(data_directory, datasets):
	# load waveforms
	import scipy.io	
	to_return = pd.DataFrame()
	for i in range(len(datasets)):
		path = data_directory+datasets[i]
		try:
			data = scipy.io.loadmat(path+'/Analysis/SpikeWaveF.mat')
			meanWaveF = data['meanWaveF'][0]
			maxIx = data['maxIx'][0]
			generalinfo 	= scipy.io.loadmat(path+'/Analysis/GeneralInfo.mat')
			shankStructure 	= loadShankStructure(generalinfo)
			spikes,shank = loadSpikeData(path+'/Analysis/SpikeData.mat', shankStructure['thalamus'])	
			index_neurons = [path.split("/")[-1]+"_"+str(n) for n in spikes.keys()]
			for i, n in zip(list(spikes.keys()), index_neurons):	
				to_return[n] = meanWaveF[i][maxIx[i]-1]			
		except:
			print(datasets[i])
			pass

	return to_return

def loadMappingNucleus(path):
	import pandas as pd
	tmp = open(path, 'r').readlines()
	tmp = [x.strip() for x in tmp]
	data = {}
	table = []
	name = tmp[0]	
	for l in tmp[1:]:		
		if 'Mouse' in l:
			table = np.array(table)			
			data[name] = pd.DataFrame(table)
			table = []
			name = l			
		else:
			table.append(l.split(" "))
	data[name] = pd.DataFrame(np.array(table))
	return data

def loadShankStructure(generalinfo):
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]-1
		else :
			shankStructure[k[0]] = []
	
	return shankStructure	

def loadShankMapping(path):	
	import scipy.io	
	spikedata = scipy.io.loadmat(path)
	shank = spikedata['shank'] -1
	return shank

def loadSpikeData(path, index):
	# units shoud be the value to convert in s 
	import scipy.io
	import neuroseries as nts
	spikedata = scipy.io.loadmat(path)
	shank = spikedata['shank'] - 1
	shankIndex = np.where(shank == index)[0]

	spikes = {}	
	for i in shankIndex:	
		spikes[i] = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2], time_units = 's')

	a = spikes[0].as_units('s').index.values	
	if ((a[-1]-a[0])/60.)/60. > 20. : # VERY BAD		
		spikes = {}	
		for i in shankIndex:	
			spikes[i] = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2]*0.0001, time_units = 's')

	return spikes, shank

def loadEpoch(path, epoch):
	import scipy.io
	import neuroseries as nts
	sampling_freq = 1250	
	behepochs = scipy.io.loadmat(path+'/Analysis/BehavEpochs.mat')

	if epoch == 'wake':
		wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
		return nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)

	elif epoch == 'sleep':
		sleep_pre_ep, sleep_post_ep = [], []
		if 'sleepPreEp' in behepochs.keys():
			sleep_pre_ep = behepochs['sleepPreEp'][0][0]
			sleep_pre_ep = np.hstack([sleep_pre_ep[1],sleep_pre_ep[2]])
			sleep_pre_ep_index = behepochs['sleepPreEpIx'][0]
		if 'sleepPostEp' in behepochs.keys():
			sleep_post_ep = behepochs['sleepPostEp'][0][0]
			sleep_post_ep = np.hstack([sleep_post_ep[1],sleep_post_ep[2]])
			sleep_post_ep_index = behepochs['sleepPostEpIx'][0]
		if len(sleep_pre_ep) and len(sleep_post_ep):
			sleep_ep = np.vstack((sleep_pre_ep, sleep_post_ep))
		elif len(sleep_pre_ep):
			sleep_ep = sleep_pre_ep
		elif len(sleep_post_ep):
			sleep_ep = sleep_post_ep						
		return nts.IntervalSet(sleep_ep[:,0], sleep_ep[:,1], time_units = 's')

	elif epoch == 'sws':
		import os
		file1 = path.split("/")[-1]+'.sts.SWS'
		file2 = path.split("/")[-1]+'-states.mat'
		listdir = os.listdir(path)
		if file1 in listdir:
			sws = np.genfromtxt(path+'/'+file1)/float(sampling_freq)
			return nts.IntervalSet.drop_short_intervals(nts.IntervalSet(sws[:,0], sws[:,1], time_units = 's'), 0.0)

		elif file2 in listdir:
			sws = scipy.io.loadmat(path+'/'+file2)['states'][0]
			index = np.logical_or(sws == 2, sws == 3)*1.0
			index = index[1:] - index[0:-1]
			start = np.where(index == 1)[0]+1
			stop = np.where(index == -1)[0]
			return nts.IntervalSet.drop_short_intervals(nts.IntervalSet(start, stop, time_units = 's', expect_fix=True), 0.0)

	elif epoch == 'rem':
		import os
		file1 = path.split("/")[-1]+'.sts.REM'
		file2 = path.split("/")[-1]+'-states.mat'
		listdir = os.listdir(path)	
		if file1 in listdir:
			rem = np.genfromtxt(path+'/'+file1)/float(sampling_freq)
			return nts.IntervalSet(rem[:,0], rem[:,1], time_units = 's').drop_short_intervals(0.0)

		elif file2 in listdir:
			rem = scipy.io.loadmat(path+'/'+file2)['states'][0]
			index = (rem == 5)*1.0
			index = index[1:] - index[0:-1]
			start = np.where(index == 1)[0]+1
			stop = np.where(index == -1)[0]
			return nts.IntervalSet(start, stop, time_units = 's', expect_fix=True).drop_short_intervals(0.0)

def loadRipples(path):	
	"""
	# 0 : debut
	# 1 : milieu
	# 2 : fin
	# 3 : amplitude nombre de sd au dessus de bruit
	# 4 : frequence instantane
	
	if a .evt.py.rip exist, it's loaded in priority
	"""
	import neuroseries as nts
	import os
	name = path.split("/")[-1]
	files = os.listdir(path)
	if name + '.evt.py.rip' in files:
		tmp = np.genfromtxt(path + '/' + name + '.evt.py.rip')[:,0]
		ripples = tmp.reshape(len(tmp)//3,3)/1000
	else:	
		ripples = np.genfromtxt(path+'/'+path.split("/")[-1]+'.sts.RIPPLES')
	return (nts.IntervalSet(ripples[:,0], ripples[:,2], time_units = 's'), 
			nts.Ts(ripples[:,1], time_units = 's'))

def loadUpDown(path):
	import neuroseries as nts
	import os
	name = path.split("/")[-1]
	files = os.listdir(path)
	if name + '.evt.py.dow' in files:
		tmp = np.genfromtxt(path+'/'+name+'.evt.py.dow')[:,0]
		tmp = tmp.reshape(len(tmp)//2,2)/1000
		down_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
	if name + '.evt.py.upp' in files:
		tmp = np.genfromtxt(path+'/'+name+'.evt.py.upp')[:,0]
		tmp = tmp.reshape(len(tmp)//2,2)/1000
		up_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
	return (down_ep, up_ep)

def loadTheta(path):	
	import scipy.io
	import neuroseries as nts
	thetaInfo = scipy.io.loadmat(path)
	troughs = nts.Tsd(thetaInfo['thetaTrghs'][0][0][2].flatten(), thetaInfo['thetaTrghs'][0][0][3].flatten(), time_units = 's')
	peaks = nts.Tsd(thetaInfo['thetaPks'][0][0][2].flatten(), thetaInfo['thetaPks'][0][0][3].flatten(), time_units = 's')
	good_ep = nts.IntervalSet(thetaInfo['goodEp'][0][0][1], thetaInfo['goodEp'][0][0][2], time_units = 's')	
	tmp = (good_ep.as_units('s')['end'].iloc[-1] - good_ep.as_units('s')['start'].iloc[0])	
	if (tmp/60.)/60. > 20. : # VERY BAD
		good_ep = nts.IntervalSet(	good_ep.as_units('s')['start']*0.0001, 
									good_ep.as_units('s')['end']*0.0001,
									time_units = 's'
								)
	return good_ep	
	# troughs = troughs.restrict(good_ep)
	# peaks = peaks.restrict(good_ep)
	# return troughs, peaks

def loadSWRMod(path, datasets, return_index = False):
	import _pickle as cPickle
	tmp = cPickle.load(open(path, 'rb'))
	z = []
	index = []
	for session in datasets:
		neurons = np.array(list(tmp[session].keys()))
		sorte = np.array([int(n.split("_")[1]) for n in neurons])
		ind = np.argsort(sorte)			
		for n in neurons[ind]:
			z.append(tmp[session][n])						
		index += list(neurons[ind])
	z = np.vstack(z)
	index = np.array(index)
	if return_index:
		return (z, index)
	else:
		return z

def loadSpindMod(path, datasets, return_index = False):
	import _pickle as cPickle
	tmp = cPickle.load(open(path, 'rb'))
	z = {'hpc':[], 'thl':[]}
	index = {'hpc':[], 'thl':[]}
	for k in z.keys():
		for session in datasets:
			neurons = np.array(list(tmp[session][k].keys()))
			sorte = np.array([int(n.split("_")[1]) for n in neurons])
			ind = np.argsort(sorte)			
			for n in neurons[ind]:
				z[k].append(tmp[session][k][n])				
			index[k] += list(neurons[ind])
		z[k] = np.vstack(z[k])
		index[k] = np.array(index[k])
	if return_index:
		return (z, index)
	else:
		return z

def loadThetaMod(path, datasets, return_index = False):
	import _pickle as cPickle	
	tmp = cPickle.load(open(path, 'rb'))
	z = {'wake':[],'rem':[]}
	index = {'wake':[], 'rem':[]}

	for k in z.keys():
		for session in datasets:
			neurons = np.array(list(tmp[session][k].keys()))
			sorte = np.array([int(n.split("_")[1]) for n in neurons])
			ind = np.argsort(sorte)			
			for n in neurons[ind]:
				z[k].append(tmp[session][k][n])				
			index[k] += list(neurons[ind])
		z[k] = np.vstack(z[k])
		index[k] = np.array(index[k])
	if return_index:
		return (z, index)
	else:
		return z

def loadXML(path):
	from xml.dom import minidom
	xmldoc 		= minidom.parse(path)
	nChannels 	= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
	fs 			= xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data	
	shank_to_channel = {}
	groups 		= xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
	for i in range(len(groups)):
		shank_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
	return int(nChannels), int(fs), shank_to_channel

def loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
	import neuroseries as nts
	if type(channel) is not list:
		f = open(path, 'rb')
		startoffile = f.seek(0, 0)
		endoffile = f.seek(0, 2)
		bytes_size = 2		
		n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
		duration = n_samples/frequency
		interval = 1/frequency
		f.close()
		with open(path, 'rb') as f:
			data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
		timestep = np.arange(0, len(data))/frequency
		return nts.Tsd(timestep, data, time_units = 's')
	elif type(channel) is list:
		f = open(path, 'rb')
		startoffile = f.seek(0, 0)
		endoffile = f.seek(0, 2)
		bytes_size = 2
		
		n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
		duration = n_samples/frequency
		f.close()
		with open(path, 'rb') as f:
			data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
		timestep = np.arange(0, len(data))/frequency
		return nts.TsdFrame(timestep, data, time_units = 's')

def loadBunch_Of_LFP(path,  start, stop, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
	import neuroseries as nts	
	bytes_size = 2		
	start_index = int(start*frequency*n_channels*bytes_size)
	stop_index = int(stop*frequency*n_channels*bytes_size)
	fp = np.memmap(path, np.int16, 'r', start_index, shape = (stop_index - start_index)//bytes_size)
	data = np.array(fp).reshape(len(fp)//n_channels, n_channels)

	if type(channel) is not list:
		timestep = np.arange(0, len(data))/frequency
		return nts.Tsd(timestep, data[:,channel], time_units = 's')
	elif type(channel) is list:
		timestep = np.arange(0, len(data))/frequency		
		return nts.TsdFrame(timestep, data[:,channel], time_units = 's')

def loadSpeed(path):
	import neuroseries as nts
	import scipy.io
	raw = scipy.io.loadmat(path)
	return nts.Tsd(raw['speed'][:,0], raw['speed'][:,1], time_units = 's')

def writeNeuroscopeEvents(path, ep, name):
	f = open(path, 'w')
	for i in range(len(ep)):
		f.writelines(str(ep.as_units('ms').iloc[i]['start']) + " "+name+" start "+ str(1)+"\n")
		f.writelines(str(ep.as_units('ms').iloc[i]['end']) + " "+name+" end "+ str(1)+"\n")
	f.close()		
	return

def plotEpoch(wake_ep, sleep_ep, rem_ep, sws_ep, ripples_ep, spikes_sws):
	from pylab import figure, plot, legend, show
	figure()
	plot([wake_ep['start'][0], wake_ep['end'][0]], np.zeros(2), '-', color = 'blue', label = 'wake')
	[plot([wake_ep['start'][i], wake_ep['end'][i]], np.zeros(2), '-', color = 'blue') for i in range(len(wake_ep))]
	plot([sleep_ep['start'][0], sleep_ep['end'][0]], np.zeros(2), '-', color = 'green', label = 'sleep')
	[plot([sleep_ep['start'][i], sleep_ep['end'][i]], np.zeros(2), '-', color = 'green') for i in range(len(sleep_ep))]	
	plot([rem_ep['start'][0], rem_ep['end'][0]],  np.zeros(2)+0.1, '-', color = 'orange', label = 'rem')
	[plot([rem_ep['start'][i], rem_ep['end'][i]], np.zeros(2)+0.1, '-', color = 'orange') for i in range(len(rem_ep))]
	plot([sws_ep['start'][0], sws_ep['end'][0]],  np.zeros(2)+0.1, '-', color = 'red', label = 'sws')
	[plot([sws_ep['start'][i], sws_ep['end'][i]], np.zeros(2)+0.1, '-', color = 'red') for i in range(len(sws_ep))]	
	plot([ripples_ep['start'][0], ripples_ep['end'][0]],  np.zeros(2)+0.2, '-', color = 'black', label = 'ripples')
	[plot([ripples_ep['start'][i], ripples_ep['end'][i]], np.zeros(2)+0.2, '-', color = 'black') for i in range(len(ripples_ep))]	
	for n in spikes_sws.keys():
		plot(spikes_sws[n].index.values, np.zeros(spikes_sws[n].size)+0.4, 'o')


	legend()
	show()

def plotThetaEpoch(wake_ep, sleep_ep, rem_ep, sws_ep, rem_peaks, rem_troughs, wake_peaks, wake_troughs):
	from pylab import figure, plot, legend, show
	figure()
	plot([wake_ep['start'][0], wake_ep['end'][0]], np.zeros(2), '-', color = 'blue', label = 'wake')
	[plot([wake_ep['start'][i], wake_ep['end'][i]], np.zeros(2), '-', color = 'blue') for i in range(len(wake_ep))]
	plot([sleep_ep['start'][0], sleep_ep['end'][0]], np.zeros(2), '-', color = 'green', label = 'sleep')
	[plot([sleep_ep['start'][i], sleep_ep['end'][i]], np.zeros(2), '-', color = 'green') for i in range(len(sleep_ep))]	
	plot([rem_ep['start'][0], rem_ep['end'][0]],  np.zeros(2)+0.1, '-', color = 'orange', label = 'rem')
	[plot([rem_ep['start'][i], rem_ep['end'][i]], np.zeros(2)+0.1, '-', color = 'orange') for i in range(len(rem_ep))]
	plot([sws_ep['start'][0], sws_ep['end'][0]],  np.zeros(2)+0.1, '-', color = 'red', label = 'sws')
	[plot([sws_ep['start'][i], sws_ep['end'][i]], np.zeros(2)+0.1, '-', color = 'red') for i in range(len(sws_ep))]	

	plot(rem_peaks.index.values, np.zeros(rem_peaks.size)+0.4, 'o')
	plot(wake_peaks.index.values, np.zeros(wake_peaks.size)+0.4, 'o')
	plot(rem_troughs.index.values, np.zeros(rem_troughs.size)+0.3, '+')
	plot(wake_troughs.index.values, np.zeros(wake_troughs.size)+0.3, '+')

	legend()
	show()

def makeTSNE(data):
	from sklearn.manifold import TSNE
	tsne = TSNE(n_components=2,
				# early_exaggeration = 4, 
				# n_iter = 10000,
				perplexity = 10,
				# learning_rate = 500,
				# min_grad_norm = 1e-12,
				# n_iter_without_progress = 1000,
				# method = 'exact'
			)
	kluster = tsne.fit_transform(data)
	return kluster, tsne.kl_divergence_

def makeAllTSNE(data, n):
	dview = Pool(cpu_count())
	datatsne = dview.map_async(makeTSNE, [data.copy() for _ in range(n)]).get()
	klusters = []
	divergence = []
	for kl, dv in datatsne:
		klusters.append(kl)
		divergence.append(dv)
	klusters = np.dstack(klusters).T
	divergence = np.array(divergence)
	return klusters, divergence


def computeAngularTuningCurves(spikes, angle, ep, nb_bins = 180, frequency = 120.0):
	bins 			= np.linspace(0, 2*np.pi, nb_bins)
	idx 			= bins[0:-1]+np.diff(bins)/2
	tuning_curves 	= pd.DataFrame(index = idx, columns = spikes.keys())	
	# little bug here in neuroseries
	angle.name 		= 0
	angle 			= angle[~angle.index.duplicated(keep='first')]
	angle 			= angle.restrict(ep)
	# # Smoothing the angle here
	# tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	# tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	# angle			= nts.Tsd(tmp2%(2*np.pi))
	for k in spikes:
		spks 			= spikes[k]
		true_ep 		= nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), 
									end = np.minimum(angle.index[-1], spks.index[-1]))		
		spks 			= spks.restrict(true_ep)		
		angle_spike 	= angle.restrict(true_ep).realign(spks)
		spike_count, bin_edges = np.histogram(angle_spike, bins)
		occupancy, _ 	= np.histogram(angle, bins)
		spike_count 	= spike_count/occupancy
		tuning_curves[k] = spike_count*frequency

	return tuning_curves

def findHDCells(tuning_curves):
	"""
		Peak firing rate larger than 1
		and Rayleigh test p<0.001 & z > 100
	"""
	cond1 = tuning_curves.max()>1.0
	
	from pycircstat.tests import rayleigh
	stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
	for k in tuning_curves:
		stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
	cond2 = np.logical_and(stat['pval']<0.001,stat['z']>100)
	tokeep = np.where(np.logical_and(cond1, cond2))[0]
	return tokeep, stat

def decodeHD(tuning_curves, spikes, ep, bin_size = 200, px = None):
	"""
		See : Zhang, 1998, Interpreting Neuronal Population Activity by Reconstruction: Unified Framework With Application to Hippocampal Place Cells
		tuning_curves: pd.DataFrame with angular position as index and columns as neuron
		spikes : dictionnary of spike times
		ep : nts.IntervalSet, the epochs for decoding
		bin_size : in ms (default:200ms)
		px : Occupancy. If None, px is uniform
	"""		
	if len(ep) == 1:
		bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
	else:
		print("TODO, more than one epoch")
		sys.exit()
	
	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = spikes.keys())
	for k in spikes:
		spks = spikes[k].restrict(ep).as_units('ms').index.values
		spike_counts[k], _ = np.histogram(spks, bins)

	tcurves_array = tuning_curves.values
	spike_counts_array = spike_counts.values
	proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))

	part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
	if px is not None:
		part2 = px
	else:
		part2 = np.ones(tuning_curves.shape[0])
		# part2 = np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
	
	for i in range(len(proba_angle)):
		part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
		p = part1 * part2 * part3
		proba_angle[i] = p/p.sum() # Normalization process here

	proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)	
	proba_angle = proba_angle.astype('float')		
	decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')
	return decoded, proba_angle

def quickBin(dspike, ts, bins, index):
	rates = np.zeros((len(ts),len(bins)-1,len(dspike)))
	for i, t in enumerate(ts):
		tbins = t+bins
		for j, n in enumerate(index):
			tspk = dspike[n]
			a, _ = np.histogram(tspk, tbins)
			b = a / (np.diff(bins)*1e-3)
			b = b - b.mean()
			b = b / (b.std()+1)			
			rates[i,:,j] = b
	# a, b, c = rates.shape
	# rates = rates.reshape(a*b, c)
	# rates = rates - rates.mean(0)
	# rates = rates / rates.std(0)
	# rates = rates.reshape(a, b, c)
	return rates

def quickBin2(dspike, ts, bins, index, meanstd):
	rates = np.zeros((len(ts),len(bins)-1,len(dspike)))
	for i, t in enumerate(ts):
		tbins = t+bins
		for j, n in enumerate(index):
			tspk = dspike[n]
			a, _ = np.histogram(tspk, tbins)
			# b = a / (np.diff(bins)*1e-3)
			# b = b - b.mean()
			# b = b / (b.std()+1)			
			# b = a/(np.diff(bins)*1e-3)
			# c = (b - meanstd[j,0])/(meanstd[j,1]+1)
			rates[i,:,j] = a
	# a, b, c = rates.shape
	# rates = rates.reshape(a*b, c)
	# rates = rates - rates.mean(0)
	# rates = rates / rates.std(0)
	# rates = rates.reshape(a, b, c)
	return rates


def _fill_array(data, mask=None, fill_value=None):
    """
    Mask numpy array and/or fill array value without demasking.
    Additionally set fill_value to value.
    If data is not a MaskedArray and mask is None returns silently data.
    :param mask: apply mask to array
    :param fill_value: fill value
    """
    if mask is not None and mask is not False:
        data = np.ma.MaskedArray(data, mask=mask, copy=False)
    if np.ma.is_masked(data) and fill_value is not None:
        data._data[data.mask] = fill_value
        np.ma.set_fill_value(data, fill_value)
#    elif not np.ma.is_masked(data):
#        data = np.ma.filled(data)
    return data

def spectral_whitening(data, sr=None, smooth=None, filter=None,
                       waterlevel=1e-8):
    """
    Apply spectral whitening to data

    Data is divided by its smoothed (Default: None) amplitude spectrum.

    :param data: numpy array with data to manipulate
    :param sr: sampling rate (only needed for smoothing)
    :param smooth: length of smoothing window in Hz
        (default None -> no smoothing)
    :param filter: filter spectrum with bandpass after whitening
        (tuple with min and max frequency)
    :param waterlevel: waterlevel relative to mean of spectrum

    :return: whitened data
    """    
    mask = np.ma.getmask(data)
    N = len(data)
    nfft = next_fast_len(N)
    spec = fft(data, nfft)
    spec_ampl = np.sqrt(np.abs(np.multiply(spec, np.conjugate(spec))))
    if smooth:
        smooth = int(smooth * N / sr)
        spec_ampl = ifftshift(smooth_func(fftshift(spec_ampl), smooth))
    # save guard against division by 0
    wl = waterlevel * np.mean(spec_ampl)
    spec_ampl[spec_ampl < wl] = wl
    spec /= spec_ampl
    if filter is not None:
        spec *= _filter_resp(*filter, sr=sr, N=len(spec), whole=True)[1]
    ret = np.real(ifft(spec, nfft)[:N])
    return _fill_array(ret, mask=mask, fill_value=0.)




from sklearn.manifold import Isomap
def makeRingManifold(spikes, ep, angle, bin_size = 200):
	"""
	spikes : dict of hd spikes
	ep : epoch to restrict
	angle : tsd of angular direction
	bin_size : in ms
	"""
	neurons = np.sort(list(spikes.keys()))
	bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
	for i in neurons:
		spks = spikes[i].as_units('ms').index.values
		spike_counts[i], _ = np.histogram(spks, bins)

	rates = np.sqrt(spike_counts/(bin_size))

	angle = angle.restrict(ep)
	newangle = pd.Series(index = np.arange(len(bins)-1))
	tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
	newangle.loc[tmp.index] = tmp
	newangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

	tmp = rates.rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2).values

	imap = Isomap(n_neighbors = 20, n_components = 2, n_jobs = -1).fit_transform(tmp)

	iwak = imap

	H = newangle.values/(2*np.pi)

	HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T

	from matplotlib.colors import hsv_to_rgb

	RGB = hsv_to_rgb(HSV)

	figure()
	ax = subplot(111)
	noaxis(ax)
	ax.set_aspect(aspect=1)
	ax.scatter(iwak[~np.isnan(H),0], iwak[~np.isnan(H),1], c = RGB[~np.isnan(H)], marker = '.', alpha = 0.5, zorder = 2, linewidth = 0, s= 40)	

	# hsv
	display_axes = fig.add_axes([0.2,0.45,0.1,0.1], projection='polar')
	colormap = plt.get_cmap('hsv')
	norm = mpl.colors.Normalize(0.0, 2*np.pi)
	xval = np.arange(0, 2*pi, 0.01)
	yval = np.ones_like(xval)
	display_axes.scatter(xval, yval, c=xval, s=100, cmap=colormap, norm=norm, linewidths=0, alpha = 0.8)
	display_axes.set_yticks([])
	display_axes.set_xticks(np.arange(0, 2*np.pi, np.pi/2))
	display_axes.grid(False)

	show()

	return 

@jit(nopython=True)
def histo(spk, obins):
	n = len(obins)
	count = np.zeros(n)
	for i in range(n):
		count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
	return count


def compute_isomap(spikes, rip_tsd, idx, obins, neurons, bin_size, sws_ep, n_ex, times, rates_wak, i):
	####################################################################################################################
	# SWR
	####################################################################################################################						
	# BINNING
	tmp = rip_tsd.index.values[idx == i]
	subrip_tsd = pd.Series(index = tmp, data = np.nan)				
	rates_swr = []
	tmp2 = subrip_tsd.index.values/1000
	for j, t in enumerate(tmp2):				
		tbins = t + obins
		spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)
		for k in neurons:
			spks = spikes[k].as_units('ms').index.values
			spike_counts[k] = histo(spks, tbins)
			
		rates_swr.append(np.sqrt(spike_counts/(bin_size)))

	####################################################################################################################
	# RANDOM
	####################################################################################################################			
	# BINNING
	rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[j,'start']+500000, sws_ep.loc[j,'end']+500000, np.maximum(1,n_ex//len(sws_ep))) for j in sws_ep.index])))
	# rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[j,'start']+500000, sws_ep.loc[j,'end']+500000, 1) for j in sws_ep.index])))
	if len(rnd_tsd) > n_ex:
		rnd_tsd = rnd_tsd[0:n_ex]
	rates_rnd = []
	tmp3 = rnd_tsd.index.values/1000
	for j, t in enumerate(tmp3):				
		tbins = t + obins
		spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)	
		for k in neurons:
			spks = spikes[k].as_units('ms').index.values	
			spike_counts[k] = histo(spks, tbins)
	
		rates_rnd.append(np.sqrt(spike_counts/(bin_size)))

	###########
	# SMOOTHING
	tmp1 = rates_wak.values
	tmp1 = tmp1.astype(np.float32)
	# SMOOTHING
	tmp3 = []
	for rates in rates_swr:
		tmp3.append(rates.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1).loc[-500:500].values)
		# tmp3.append(rates.loc[-500:500].values)
	tmp3 = np.vstack(tmp3)
	tmp3 = tmp3.astype(np.float32)
	#SMOOTHING
	tmp2 = []
	for rates in rates_rnd:				
		tmp2.append(rates.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1).loc[-500:500].values)
		# tmp2.append(rates.loc[-500:500].values)
	tmp2 = np.vstack(tmp2)
	tmp2 = tmp2.astype(np.float32)

	n = len(tmp3)
	m = len(tmp1)
	tmp = np.vstack((tmp1, tmp3, tmp2))

	# ISOMAP WAKE SWR
	# tmp = np.vstack((tmp1, tmp3))
	imap = Isomap(n_neighbors = 200, n_components = 2).fit_transform(tmp)
	iwak = imap[0:m]
	iswr = imap[m:m+n].reshape(len(subrip_tsd),len(times),2)

	# # ISOMAP WAKE RND
	# tmp = np.vstack((tmp1, tmp2))
	# imap = Isomap(n_neighbors = 5, n_components = 2).fit_transform(tmp)
	irnd = imap[m+n:].reshape(len(rnd_tsd),len(times),2)
			
	print(i)
	return (iswr, irnd, iwak)


