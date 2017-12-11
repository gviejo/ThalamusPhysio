

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
import _pickle as cPickle
import time
import os, sys
import ipyparallel
import neuroseries as nts

data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

# to know which neurons to keep
theta_mod, theta_ses    = loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta               = pd.DataFrame( index = theta_ses['rem'], 
                                    columns = ['phase', 'pvalue', 'kappa'],
                                    data = theta_mod['rem'])
tmp2            = theta.index[theta.isnull().any(1)].values
tmp3            = theta.index[(theta['pvalue'] > 0.01).values].values
tmp             = np.unique(np.concatenate([tmp2,tmp3]))
theta_modth = theta.drop(tmp, axis = 0)
neurons_index = theta_modth.index.values

bins1 = np.arange(-1005, 1010, 25)*1000     
times = np.floor(((bins1[0:-1] + (bins1[1] - bins1[0])/2)/1000)).astype('int')          
premeanscore = {i:{'rem':pd.DataFrame(index = [], columns = ['mean', 'std']),'rip':pd.DataFrame(index = times, columns = [])} for i in range(3)}# BAD
posmeanscore = {i:{'rem':pd.DataFrame(index = [], columns = ['mean', 'std']),'rip':pd.DataFrame(index = times, columns = [])} for i in range(3)}# BAD
bins2 = np.arange(-1012.5,1025,25)*1000
tsmax = {i:pd.DataFrame(columns = ['pre', 'pos']) for i in range(3)}


clients = ipyparallel.Client()
print(clients.ids)
dview = clients.direct_view()
def compute_pop_pca(session):
    data_directory = '/mnt/DataGuillaume/MergedData/'
    import numpy as np  
    import scipy.io 
    import scipy.stats      
    import _pickle as cPickle
    import time
    import os, sys  
    import neuroseries as nts
    from functions import loadShankStructure, loadSpikeData, loadEpoch, loadThetaMod, loadSpeed, loadXML, loadRipples, loadLFP, downsample, getPeaksandTroughs, butter_bandpass_filter
    import pandas as pd 

    # to know which neurons to keep
    data_directory = '/mnt/DataGuillaume/MergedData/'
    datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')    
    theta_mod, theta_ses    = loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
    theta               = pd.DataFrame( index = theta_ses['rem'], 
                                        columns = ['phase', 'pvalue', 'kappa'],
                                        data = theta_mod['rem'])
    tmp2            = theta.index[theta.isnull().any(1)].values
    tmp3            = theta.index[(theta['pvalue'] > 0.01).values].values
    tmp             = np.unique(np.concatenate([tmp2,tmp3]))
    theta_modth = theta.drop(tmp, axis = 0)
    neurons_index = theta_modth.index.values

    bins1 = np.arange(-1005, 1010, 25)*1000     
    times = np.floor(((bins1[0:-1] + (bins1[1] - bins1[0])/2)/1000)).astype('int')          
    premeanscore = {i:{'rem':pd.DataFrame(index = [], columns = ['mean', 'std']),'rip':pd.DataFrame(index = times, columns = [])} for i in range(3)}
    posmeanscore = {i:{'rem':pd.DataFrame(index = [], columns = ['mean', 'std']),'rip':pd.DataFrame(index = times, columns = [])} for i in range(3)}
    bins2 = np.arange(-1012.5,1025,25)*1000 
    tsmax = {i:pd.DataFrame(columns = ['pre', 'pos']) for i in range(3)}
    

# for session in datasets:
# for session in datasets[0:15]:
# for session in ['Mouse12/Mouse12-120815']:
    start_time = time.clock()
    print(session)
    generalinfo     = scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
    shankStructure  = loadShankStructure(generalinfo)   
    if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
        hpc_channel     = generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
    else:
        hpc_channel     = generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1  
    spikes,shank    = loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])       
    wake_ep         = loadEpoch(data_directory+session, 'wake')
    sleep_ep        = loadEpoch(data_directory+session, 'sleep')
    sws_ep          = loadEpoch(data_directory+session, 'sws')
    rem_ep          = loadEpoch(data_directory+session, 'rem')
    sleep_ep        = sleep_ep.merge_close_intervals(threshold=1.e3)        
    sws_ep          = sleep_ep.intersect(sws_ep)    
    rem_ep          = sleep_ep.intersect(rem_ep)
    speed           = loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)  
    speed_ep        = nts.IntervalSet(speed[speed>2.5].index.values[0:-1], speed[speed>2.5].index.values[1:]).drop_long_intervals(26000).merge_close_intervals(50000)
    wake_ep         = wake_ep.intersect(speed_ep).drop_short_intervals(3000000) 
    n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')   
    rip_ep,rip_tsd  = loadRipples(data_directory+session)
    hd_info         = scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
    hd_info_neuron  = np.array([hd_info[n] for n in spikes.keys()])     
    all_neurons     = np.array(list(spikes.keys()))
    mod_neurons     = np.array([int(n.split("_")[1]) for n in neurons_index if session.split("/")[1] in n])
    if len(sleep_ep) > 1:
        store           = pd.HDFStore("/mnt/DataGuillaume/population_activity_25ms/"+session.split("/")[1]+".h5")   
        # all_pop       = store['allwake']
        pre_pop         = store['presleep']
        pos_pop         = store['postsleep']
        store.close()

        store           = pd.HDFStore("/mnt/DataGuillaume/population_activity_100ms/"+session.split("/")[1]+".h5")           
        all_pop         = store['allwake']
        # pre_pop       = store['presleep']
        # pos_pop       = store['postsleep']
        store.close()
        
        def compute_eigen(popwak):
            popwak = popwak - popwak.mean(0)
            popwak = popwak / (popwak.std(0)+1e-8)
            from sklearn.decomposition import PCA   
            pca = PCA(n_components = popwak.shape[1])
            xy = pca.fit_transform(popwak.values)
            pc = pca.explained_variance_ > (1 + np.sqrt(1/(popwak.shape[0]/popwak.shape[1])))**2.0
            eigen = pca.components_[pc]         
            lambdaa = pca.explained_variance_[pc]
            return eigen, lambdaa

        def compute_score(ep_pop, eigen, lambdaa, thr):
            ep_pop = ep_pop - ep_pop.mean(0)
            ep_pop = ep_pop / (ep_pop.std(0)+1e-8)          
            a = ep_pop.values
            score = np.zeros(len(ep_pop))
            for i in range(len(eigen)):
                 if lambdaa[i] >= thr:                  
                    score += (np.dot(a, eigen[i])**2.0 - np.dot(a**2.0, eigen[i]**2.0))                 
            score = nts.Tsd(t = ep_pop.index.values, d = score)
            return score

        def compute_rip_score(tsd, score, bins):                                
            times = np.floor(((bins[0:-1] + (bins[1] - bins[0])/2)/1000)).astype('int')         
            rip_score = pd.DataFrame(index = times, columns = [])
            for r,i in zip(tsd.index.values,range(len(tsd))):
                xbins = (bins + r).astype('int')
                y = score.groupby(pd.cut(score.index.values, bins=xbins, labels = times)).mean()
                if ~y.isnull().any():
                    rip_score[r] = y

            return rip_score

        def get_xmin(ep, minutes):
            duree = (ep['end'] - ep['start'])/1000/1000/60
            tmp = ep.iloc[np.where(np.ceil(duree.cumsum()) <= minutes + 1)[0]]
            return nts.IntervalSet(tmp['start'], tmp['end'])            



        pre_ep          = nts.IntervalSet(sleep_ep['start'][0], sleep_ep['end'][0])
        post_ep         = nts.IntervalSet(sleep_ep['start'][1], sleep_ep['end'][1])
    
        pre_sws_ep      = sws_ep.intersect(pre_ep)
        pos_sws_ep      = sws_ep.intersect(post_ep)
        pre_sws_ep      = get_xmin(pre_sws_ep.iloc[::-1], 30)
        pos_sws_ep      = get_xmin(pos_sws_ep, 30)

        if pre_sws_ep.tot_length('s')/60 > 5.0 and pos_sws_ep.tot_length('s')/60 > 5.0:                 
            for hd in range(3):
                if hd == 0 or hd == 2:
                    index = np.where(hd_info_neuron == 0)[0]
                elif hd == 1:
                    index = np.where(hd_info_neuron == 1)[0]
                if hd == 0:
                    index = np.intersect1d(index, mod_neurons)
                elif hd == 2:
                    index = np.intersect1d(index, np.setdiff1d(all_neurons, mod_neurons))

                allpop = all_pop[index].copy()          
                prepop = nts.TsdFrame(pre_pop[index].copy())                
                pospop = nts.TsdFrame(pos_pop[index].copy())
                # prepop25ms = nts.TsdFrame(pre_pop_25ms[index].copy())
                # pospop25ms = nts.TsdFrame(pos_pop_25ms[index].copy())
                if allpop.shape[1] and allpop.shape[1] > 5:                                 
                    eigen,lambdaa   = compute_eigen(allpop)                 
                    seuil           = 1.2
                    if np.sum(lambdaa > seuil):
                        pre_score       = compute_score(prepop, eigen, lambdaa, seuil)
                        pos_score       = compute_score(pospop, eigen, lambdaa, seuil)
                                        
                        prerip_score    = compute_rip_score(rip_tsd.restrict(pre_sws_ep), pre_score, bins1)
                        posrip_score    = compute_rip_score(rip_tsd.restrict(pos_sws_ep), pos_score, bins1)
                        
                        # pre_score_25ms    = compute_score(prepop25ms, eigen)
                        # pos_score_25ms    = compute_score(pospop25ms, eigen)                                      
                        # prerip25ms_score = compute_rip_score(rip_tsd.restrict(pre_ep),  pre_score_25ms, bins2)
                        # posrip25ms_score = compute_rip_score(rip_tsd.restrict(post_ep), pos_score_25ms,  bins2)
                        # prerip25ms_score = prerip25ms_score - prerip25ms_score.mean(0)    
                        # posrip25ms_score = posrip25ms_score - posrip25ms_score.mean(0)    
                        # prerip25ms_score = prerip25ms_score / prerip25ms_score.std(0) 
                        # posrip25ms_score = posrip25ms_score / posrip25ms_score.std(0)
                        # prerip25ms_score = prerip25ms_score.loc[-500:500]         
                        # posrip25ms_score = posrip25ms_score.loc[-500:500]                 
                        # sys.exit()                    
                        # tmp = pd.concat([pd.DataFrame(prerip25ms_score.idxmax().values, columns = ['pre']),pd.DataFrame(posrip25ms_score.idxmax().values, columns = ['pos'])],axis = 1)
                        # tmp = pd.DataFrame(data = [[prerip25ms_score.mean(1).idxmax(), posrip25ms_score.mean(1).idxmax()]], columns = ['pre', 'pos'])
                        # tsmax[hd] = tsmax[hd].append(tmp, ignore_index = True)

                        premeanscore[hd]['rip'][session] = prerip_score.mean(1)
                        posmeanscore[hd]['rip'][session] = posrip_score.mean(1)

                        # if len(rem_ep.intersect(pre_ep)) and len(rem_ep.intersect(post_ep)):
                        #   premeanscore[hd]['rem'].loc[session,'mean'] = pre_score.restrict(rem_ep.intersect(pre_ep)).mean()
                        #   posmeanscore[hd]['rem'].loc[session,'mean'] = pos_score.restrict(rem_ep.intersect(post_ep)).mean()
                        #   premeanscore[hd]['rem'].loc[session,'std'] =  pre_score.restrict(rem_ep.intersect(pre_ep)).std()
                        #   posmeanscore[hd]['rem'].loc[session,'std'] =  pos_score.restrict(rem_ep.intersect(post_ep)).std()


    return [premeanscore, posmeanscore, tsmax]


# sys.exit()

a = dview.map_sync(compute_pop_pca, datasets)


prescore = {i:pd.DataFrame(index = times) for i in range(3)}
posscore = {i:pd.DataFrame(index = times) for i in range(3)}
for i in range(len(a)):
    for j in range(3):
        if len(a[i][0][j]['rip'].columns):
            s = a[i][0][j]['rip'].columns[0]
            prescore[j][s] = a[i][0][j]['rip']
            posscore[j][s] = a[i][1][j]['rip']

# prescore = premeanscore
# posscore = posmeanscore

from pylab import *
titles = ['non hd mod', 'hd', 'non hd non mod']
figure()
for i in range(3):
    subplot(1,3,i+1)
    
    times = prescore[i].index.values
    # for s in premeanscore[i]['rip'].index.values:     
    #   plot(times, premeanscore[i]['rip'].loc[s].values, linewidth = 0.3, color = 'blue')
    #   plot(times, posmeanscore[i]['rip'].loc[s].values, linewidth = 0.3, color = 'red')
    plot(times, gaussFilt(prescore[i].mean(1).values, (1,)), label = 'pre', color = 'blue', linewidth = 2)
    plot(times, gaussFilt(posscore[i].mean(1).values, (1,)), label = 'post', color = 'red', linewidth = 2)
    legend()
    title(titles[i])

show()

sys.exit()

#########################################
# search for peak in 25 ms array
########################################
tsmax = {i:pd.DataFrame(columns = ['pre', 'pos']) for i in range(2)}
for i in range(len(a)):
    for hd in range(2):
        tsmax[hd] = tsmax[hd].append(a[i][2][hd], ignore_index = True)


from pylab import *
plot(tsmax[0]['pos'], np.ones(len(tsmax[0]['pos'])), 'o')
plot(tsmax[0]['pos'].mean(), [1], '|', markersize = 10)
plot(tsmax[1]['pos'], np.zeros(len(tsmax[1]['pos'])), 'o')
plot(tsmax[1]['pos'].mean(), [0], '|', markersize = 10)

sys.exit()

#########################################
# SAVING
########################################

store = pd.HDFStore("../figures/figures_articles/figure3/pca_analysis_3.h5")
for i,j in zip(range(3),('nohd_mod', 'hd', 'nohd_nomod')):
    store.put(j+'pre_rip', prescore[i])
    store.put(j+'pos_rip', posscore[i])
store.close()



# a = dview.map_sync(compute_population_correlation, datasets[0:15])
# for i in range(len(a)):
#   if type(a[i]) is dict:
#       s = list(a[i].keys())[0]
#       premeanscore.loc[s] = a[i][s]['pre']
#       posmeanscore.loc[s] = a[i][s]['pos']

from pylab import *
titles = ['non hd', 'hd']
figure()
for i in range(2):
    subplot(1,3,i+1)
    
    times = premeanscore[i]['rip'].columns.values
    # for s in premeanscore[i]['rip'].index.values:     
    #   plot(times, premeanscore[i]['rip'].loc[s].values, linewidth = 0.3, color = 'blue')
    #   plot(times, posmeanscore[i]['rip'].loc[s].values, linewidth = 0.3, color = 'red')
    plot(times, gaussFilt(premeanscore[i]['rip'].mean(0).values, (1,)), label = 'pre', color = 'blue', linewidth = 2)
    plot(times, gaussFilt(posmeanscore[i]['rip'].mean(0).values, (1,)),label = 'post', color = 'red', linewidth = 2)
    legend()
    title(titles[i])
subplot(1,3,3)
bar([1,2], [premeanscore[0]['rem'].mean(0)['mean'], premeanscore[1]['rem'].mean(0)['mean']])
bar([3,4], [posmeanscore[0]['rem'].mean(0)['mean'], posmeanscore[1]['rem'].mean(0)['mean']])
xticks([1,2], ['non hd', 'hd'])
xticks([3,4], ['non hd', 'hd'])


            
show()

figure()
subplot(121)
times = premeanscore[0]['rip'].columns.values
for s in premeanscore[0]['rip'].index.values:       
    print(s)
    plot(times, premeanscore[0]['rip'].loc[s].values, linewidth = 1, color = 'blue')
plot(premeanscore[0]['rip'].mean(0))
subplot(122)    
for s in posmeanscore[0]['rip'].index.values:       
    plot(times, posmeanscore[0]['rip'].loc[s].values, linewidth = 1, color = 'red')
plot(posmeanscore[0]['rip'].mean(0))
show()



