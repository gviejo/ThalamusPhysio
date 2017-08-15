import numpy as np


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
		while t2[i2] < lbound and i2 < nt2:
			i2 = i2+1
		while t2[i2-1] > lbound and i2 > 1:
			i2 = i2-1

		rbound = lbound
		l = i2
		for j in range(nbins):
			k = 0
			rbound = rbound+binsize
			while t2[l] < rbound and l < nt2:
				l = l+1
				k = k+1

			C[j] += k

	for j in range(nbins):
		C[j] = C[j] / (nt1 * binsize)

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
	allcount = allcount/(float(len(t1))*binsize)
	return allcount

def xcrossCorr(t1, t2, binsize, nbins, nbiter, jitter):	
	allcount = crossCorr(t1, t2, binsize, nbins)
	# JITTERING 
	jitter_count = np.zeros((nbiter,nbins+1))
	for i in range(nbiter):		
		t1_jitter = t1+np.random.uniform(-jitter,+jitter,len(t1))
		jitter_count[i,:] = crossCorr(t1_jitter, t2, binsize, nbins)
	mean_jitter_count = jitter_count.mean(0)		
	
	return (allcount - mean_jitter_count)/np.std(allcount)

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