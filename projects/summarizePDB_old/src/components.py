import numpy as np
from matplotlib import pyplot as plt
from msmbuilder.decomposition import tICA, PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize
import scipy
import scipy.spatial
import scipy.cluster
from scipy import linalg

##########################
# higher-level functions #
##########################

def filter(traj,ids,pc_thresh=0.75,ic_thresh=1.0,fun='logcosh',algo='parallel',do_plot=False,title=''):
# identify and remove outliers
    # initialize
    traj_new, ids_new = traj_slice(traj,ids)
    if(ic_thresh < 1.0):
        it=0
        iterate=True
        # iterate
        while iterate :
            print('=================')
            print('Iteration #',it)
            subtitle=title+'outlier_iter'+str(it)
            traj_old, ids_old = traj_slice(traj_new, ids_new)
            v_ica, m_ica, x_ica = analyses(traj_old,ids_old,pc_thresh=pc_thresh,fun=fun,algo=algo,do_plot=do_plot,title=subtitle)
            idx_remains = np.arange(len(ids_old))
            n_outlier=0
            for iPC in np.arange(len(x_ica[0,:])):
                keyword=subtitle+'_IC'+str(iPC)
                idx_remains, idx_outlier = get_outlier(x_ica[:,iPC],idx_remains,vmax=ic_thresh)
                if(len(idx_outlier)>0):
                    save_traj(traj_old.slice(idx_outlier),ids_old[idx_outlier],keyword=keyword,verbose='full')
                    n_outlier+=1
            if(n_outlier==0):
                iterate=False
                print('converged in ',it,' iterations')
            else:
                it+=1
                traj_new, ids_new = traj_slice(traj_old.slice(idx_remains), ids_old[idx_remains])
    return traj_new,ids_new

def cluster(traj,ids,pc_thresh=0.75,fun='logcosh',algo='parallel',analysis_type='ica',do_plot=True,title=''):
    # perform ICA
    v,m,x = analyses(traj,ids,pc_thresh=pc_thresh,fun=fun,algo=algo,analysis_type='ica',do_plot=False,title=title)
    negent_ave, negent_var = ave_score(x,len(x[0,:]),fun=fun)
    # get pairwise distances. note that metric can be a function you've defined
    p = scipy.spatial.distance.pdist(x, metric='minkowski',p=2,w=negent_ave)
    # perform ward clustering
    clusters = scipy.cluster.hierarchy.linkage(p, method='ward')
    if(do_plot):
        plot_cluster(clusters,x,figname=title)
    return clusters

def analyses(traj,ids,pc_thresh=0.75,fun='logcosh',algo='parallel',analysis_type='ica',do_plot=True,c=None,title=''):
    """
    purpose: perform pca, to reduce dimensionality, then ica to identify relevant projections
     options: 
       analysis_type: 'pca' or 'ica'
       do_plot: True or False
    """
    v_pca, l_pca, x = traj2pc(traj, n_components=len(ids),var_trunc=pc_thresh)
    if(do_plot):
        print('Principal Component Analysis (step 0)')
        plot_stats_CA(x,l_pca,len(l_pca),threshold=0,analysis_type='pca',figname=title)
    if(analysis_type == 'ica'):
        v_ica, m_ica, x_ica = traj2ic(x,n_components=len(l_pca),fun=fun,algo=algo)
        x = x_ica
        if(len(l_pca)>1 and do_plot):
            print('Independent Component Analysis (step 1)')
            plot_stats_CA(x,m_ica,len(l_pca),threshold=0,figname=title)
    if(do_plot):
        print('projection of data in component space')
        biplots(x,n=np.minimum(10,len(l_pca)),nbins=30,c=c,figname=title)
    if(analysis_type == 'ica'):
        return v_ica, m_ica, x
    else:
        return v_pca, l_pca, x

def save_ICmode(traj,ids,v,m,x,nICs=np.arange(1),vmax=0.2,mode='oscillatory',exclude=True,keyword='mode',pc_thresh=0.75,verbose='minimal'):
    """
    purpose: write traj along iIC
    options:
     > mode: 'oscillatory': 
               or 'sorted':
     > exclude: exclude snapshots that lie further than vrange away in any other direction
    """
    # we have an issue here, when traj has less than nframe... to be solved.
    nframe=20
    xyz     = traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3)
    xyz_mean = np.mean(xyz, axis=0)
    v_pca, l_pca, x_pca = traj2pc(traj, n_components=len(ids),var_trunc=pc_thresh)
    xyz_ic = np.dot(v,v_pca)
    for iIC in nICs:
        keyword_tmp=keyword+'_'+mode+'_IC'+str(iIC+1)
        index = get_sorted_index(x,iIC,nICs=nICs,vmax=vmax,exclude=exclude)
        traj_tmp,ids_tmp = traj_slice(traj,ids,index=index)
        if(mode=='oscillatory'):
            traj_tmp = traj[0:nframe]
            for iframe in np.arange(0,nframe,1):
                phase = iframe*2.*np.pi/nframe
                traj_tmp.xyz[iframe:iframe+1,:] = 10*xyz_ic[iIC:iIC+1,:].reshape(1,traj.n_atoms,3)*np.cos(phase)
                traj_tmp.xyz[iframe:iframe+1,:] += xyz_mean.reshape(traj.n_atoms,3)
        save_traj(traj_tmp,ids_tmp,keyword=keyword_tmp,verbose=verbose)

def save_traj(traj,ids,keyword='traj',save_mean=False,verbose='minimal'):
    """
    purpose: write current state to file
    options: 
        > save_mean: write sample mean (default:False)
        > verbose: 'minimal' or 'full'
    """
    if(len(ids) > 0):
        if(save_mean):
            filename=keyword+'_mean.pdb'
        else:
            filename=keyword+'.pdb'
        # prepare traj to be written
        if(save_mean):
            n_frames= traj.n_frames
            n_atoms = traj.n_atoms
            xyz     = traj.xyz.reshape(n_frames, n_atoms * 3)
            xyz_mean = np.mean(xyz, axis=0)
            xyz_std2 = np.std(xyz, axis=0)**2
            xyz_bfac = np.sum(xyz_std2.reshape(n_atoms, 3), axis=1)*8*(np.pi)**2
            traj_mean = traj[0]
            traj_mean.xyz = xyz_mean.reshape(n_atoms, 3)
            traj_mean.save(filename, bfactors=np.clip(xyz_bfac, -9, 99))
        else:
            traj.save(filename)
        # add info of ID list at beginning
        with open(filename,'r') as f:
            save = f.read()
        with open(filename, 'w') as f:
            f.write("REMARK ID list: ")
            for item in ids:
                f.write("%s " % item)
            f.write('\n')
        with open(filename, 'a') as f:
            f.write(save)
        if(verbose=='full'):
            print('wrote ',filename,' : ',ids)

def traj2pc(traj,n_components=1,negent_sort=False,var_trunc=-1):
    n_cmpnt = n_components
    xyz = get_xyz_centered(traj)
    v_pc, l_pc = get_pca(xyz,n_components=n_components)
    x_pc = proj(v_pc,xyz)
    # truncate is asked to
    if(var_trunc > 0):
        n_cmpnt = get_truncate_order(l_pc,var_trunc)
        v_pc, l_pc, x_pc = truncate_svd(v_pc, l_pc, x_pc, n=n_cmpnt)
    # sort by decreasing order of negentropy if asked to
    if(negent_sort):
        Jscore,Jtmp = ave_score(x_pc,n_cmpnt)
        index = np.argsort(Jscore)[::-1]
        v_pc = v_pc[:,index]
        l_pc = l_pc[index]
        x_pc = x_pc[:,index]
    return v_pc,l_pc,x_pc

def traj2ic(x_pc,n_components=1,fun='logcosh',algo='parallel',negent_sort=True):
    ica = FastICA(whiten=True,algorithm=algo,fun=fun)
    x_ic = ica.fit_transform(x_pc)
    m_ic = ica.mixing_
    v_ic = ica.components_
    if(negent_sort):
        Jscore,Jtmp = ave_score(x_ic,n_components)
        index = np.argsort(Jscore)[::-1]
        x_ic = x_ic[:,index]
        m_ic = m_ic[:,index]
        v_ic = v_ic[index,:]
    return v_ic,m_ic,x_ic

##############
# Clustering #
##############

def cluster_split(traj,ids,l,n_clusters,title='',save_mean=True):
    assignments = get_assignment(l,n_clusters) 
    for i_cluster in np.arange(1,n_clusters+1):
        keyword=title+'cluster_'+str(i_cluster)
        print('> '+keyword)
        idx_in = get_idx(assignments,vrange=[i_cluster-1,i_cluster+1])
        save_traj(traj.slice(idx_in),ids[idx_in],keyword=keyword,save_mean=save_mean,verbose='full')
def get_assignment(l,n_clusters):
    return scipy.cluster.hierarchy.fcluster(l, t=n_clusters, criterion='maxclust')

def get_ncluster(l):
    for i in np.arange(l.shape[0]-2):
        m = l[::-1]
        score = (m[i,2]-m[i+1,2])/m[0,2]
        if(score>0.1):
            return i+2

#############
# Filtering #
#############

def get_sorted_index(x,iIC=1,nICs=np.arange(1),vmax=0.2,exclude=True):
    index = np.argsort(x[:,iIC])
    id_kp=index
    if(exclude):
        for jIC in nICs:
            if(jIC!=iIC):
                id1,id2 = get_outlier(x[:,jIC],index,vmax=vmax)
                id1 = np.setdiff1d(id_kp,id2)
                id_kp = id1
    index = np.argsort(x[id_kp,iIC])
    return index

def get_idx(x,vrange=[-0.2,0.2]):
    return [i for i,v in enumerate(x) if v > vrange[0] and v < vrange[1]]

def get_outlier(x,index,vmax=0.9):
    idx_outlier = [i for i,v in enumerate(abs(x)) if v > vmax]
    idx_remains = np.setdiff1d(index,idx_outlier)
    return idx_remains, idx_outlier

def get_truncate_order(L,threshold=0.9):
    var = L**2
    var /= np.sum(var)
    n_components = 1
    if(threshold==1.0):
        n_components = len(var)
    else:
        for i in np.arange(0,len(var)-1,1):
            var_current = np.cumsum(var)[i]
            var_next = np.cumsum(var)[i+1]
            if(var_current < threshold and var_next > threshold):
                n_components=i+1
    return n_components

def truncate_svd(U,L,V,n=-1):
    if(n==-1):
        n=L.shape[0]
    #Ul = U[:,0:n]
    Ul = U[0:n,:]
    Ll = L[0:n]
    Vl = V[:,0:n]
    return Ul, Ll, Vl

#################
# Preprocessing #
#################

def get_xyz_centered(traj):
    # Extract coordinates
    xyz = traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3)
    # Center 
    xyz_mean = np.mean(xyz, axis=0)
    xyz_ctrd = xyz - xyz_mean
    return xyz_ctrd

def traj_slice(traj,ids,index=[]):
    if(len(index) == 0):
        index = np.arange(len(ids))
    return traj.slice(index), ids[index]

############################## 
# Component Analyses Methods #
##############################

def get_svd(traj):
# Singular Value Decomposition
# if traj is (n_sample, n_features), Vh: components. Otherwhise, U
    U,s,Vh = linalg.svd(traj)
    return U,s,Vh

def get_pca(traj,n_components=1):
# Principal Component Analysis
    pca = PCA(n_components=n_components,svd_solver='full')
    pca.fit([traj])
    return pca.components_, pca.singular_values_

def get_ica(traj,n_components=1,fun='logcosh'):
# Independent Component Analysis
    ica = FastICA(n_components,fun=fun)
    ica.fit(traj)
    return ica.components_

def get_tica(traj,n_components=1,lag_time=100):
# time-structure ICA
    tica = tICA(n_components,lag_time)
    tica.fit([traj])
    return tica.components_, tica.eigenvalues_

def proj(component,traj):
# returns coordinates of samples in component space
    proj = np.dot(traj,component.T)
    normed = normalize(proj,axis=0)
    return normed

###################### 
# Plotting functions #
######################

def plot_summary(v,prj,neg,n_components,nbins=20,nfig=3,title='notitle'):
    print(title)
    plot_negent(neg,figsize=(nfig*4,nfig))
    plot_component(v,figsize=(nfig*4,nfig*2))
    biplots(prj,n=n_components,figsize=(nfig*4,nfig*4),nbins=nbins)

def plot_negent(negent,figsize=(12,3)):
# plot component negentropies
    fig = plt.figure(figsize=figsize)
    plt.title('negentropy')
    plt.bar(np.arange(len(negent)),negent)
    plt.tight_layout()
    plt.show()

def plot_component(component,figsize=(12,6)):
# plot components
    fig = plt.figure(figsize=figsize)
    m,n = component.shape
    plt.title('components')
    for i in np.arange(m):
        j = i+1
        plt.subplot(1,m,j)
        plt.bar(np.arange(n),component[i,:])
    plt.tight_layout()
    plt.show()

def biplots(prj,n=1,plottype='hexbin',nbins=10,figsize=-1,c=None,figname=''):
# plot populations in component space
    if c is not None:
        plottype='scatter'
    if(plottype=='scatter'):
        cmap='rainbow'
    else:
        cmap='plasma'
    if(figsize < 0 ):
        if(n == 1):
            figsize=1
        else:
            figsize=4
    figsize=figsize*6
    labels = get_labels(n) 
    fig = plt.figure(figsize=(figsize,figsize), dpi= 160, facecolor='w', edgecolor='k')
    nrow=n
    ncol=n
    #color_hexbin='plasma'
    nbins_coarse = int(nbins/1)
    nbox=1 #nrow
    for i in np.arange(0,n,1):
        for j in np.arange(0,n,1):
            if(i<j):
                ax = fig.add_subplot(nrow,ncol,nbox)
                plt.grid()
                if(j<n):
                    if(i == 0):
                        ax.set_xlabel(labels[j])   
                    if(j == n - 1):
                        ax.set_ylabel(labels[i])
                    ax.xaxis.tick_top()
                    ax.yaxis.tick_right()
                    ax.xaxis.set_label_position('top')
                    ax.yaxis.set_label_position('right')
                    Ax = prj[:,j]
                    Ay = prj[:,i]
                    if(plottype == 'scatter'):
                        plt.scatter(Ax, Ay, c=c, cmap=cmap)
                    else:
                        plt.hexbin(Ax, Ay, gridsize=nbins, cmap=cmap, mincnt=1)
            elif(i==j):
                ax = fig.add_subplot(nrow,ncol,nbox)
                plt.grid()
                Ax = prj[:,i]
                plt.hist(Ax,bins=nbins_coarse)
            else:
                if(i == 1):
                    if(j == 0):
                        ax = fig.add_subplot(nrow,ncol,nbox)
                        ax.set_xlabel('cluster color')
                        xy = range(1,np.max(c)+1,1)
                        z = xy
                        sc = plt.scatter(xy, xy, c=z, vmin=1, vmax=np.max(c), cmap=cmap)
                        plt.colorbar(sc)
            nbox=nbox+1
    plt.tight_layout()
    plt.show()
    if(figname):
        fig.savefig(figname+'_biplot.png')

def plot_stats_CA(prj,l,n_components=1,fun='logcosh',threshold=0.9,niter=100,span=True,figsize=-1,analysis_type='ica',figname=''):
    # Preplotting...
    if(figsize < 0 ):
        if(n_components == 1):
            figsize=1
        else:
            figsize=2
    figsize=figsize*6
    labels = get_labels(n_components)
    prj_GVs = np.random.randn(len(prj[:,0]),n_components)/np.sqrt(len(prj[:,0]))
    GVscore, GVscore_var = ave_score(prj_GVs,n_components,niter,fun)
    CVscore, CVscore_var = ave_score(prj,n_components,niter,fun)
    fig = plt.figure(figsize=(figsize, figsize/2), dpi= 160, facecolor='w', edgecolor='k')
    nrow=1 #nrow=2
    ncol=2 #ncol=1
    idx=1
    if(n_components>1):
        ncol=3
        if(analysis_type == 'ica'):
            # - MIXING
            plt.subplot(nrow,ncol,idx)
            plt.title('|mixing matrix|')
            plt.xlabel('IC')
            plt.ylabel('PC')
            plt.imshow(abs(l),cmap='plasma')
            plt.colorbar()
        else:
            # - VARIANCE
            var = l**2
            var /= np.sum(var)
            plt.subplot(nrow,ncol,idx)
            plt.grid()
            plt.title('Variance ratio per component')
            plt.xlabel('ID')
            plt.ylabel('variance ratio')
            plt.plot(range(1,1+len(var)), var, 'ko')
            #plt.plot(range(1,1+len(var)), np.cumsum(var), 'k+')
            plt.axhline(y=threshold, color='r', linestyle='-')
        idx+=1
    # - NEGENTROPY
    plt.subplot(nrow,ncol,idx)
    plt.grid()
    plt.title('Negentropy of component')
    plt.xlabel('ID')
    plt.ylabel('negentropy')
    plt.errorbar(np.arange(1,n_components+1,1),GVscore,yerr=np.sqrt(GVscore_var))
    plt.errorbar(np.arange(1,n_components+1,1),CVscore,yerr=np.sqrt(CVscore_var))
    plt.plot(np.arange(1,n_components+1,1),GVscore,'x-')
    plt.plot(np.arange(1,n_components+1,1),CVscore,'o-')
    idx+=1
    if(span):
        # - SPAN 
        plt.subplot(nrow,ncol,idx)
        plt.grid()
        plt.title('Population of component')
        plt.ylabel('sorted coordinates')
        for y_arr, label in zip(prj.T, labels):
            plt.plot(np.sort(y_arr), '-', label=label)
        if(n_components < 10):
            plt.legend()
    #
    plt.tight_layout()
    plt.show()
    if(figname):
        fig.savefig(figname+'_'+analysis_type+'_stats.png')

def plot_cluster(clusters,x,figsize=12,figname=''):
    fig = plt.figure(figsize=(figsize, figsize), dpi= 160, facecolor='w', edgecolor='k')
    nrow=2
    ncol=1
    # look at the number of natural clusters using the linkage object
    plt.subplot(nrow,ncol,1)
    plt.scatter(np.arange(1,len(x)), clusters[:,2][::-1])
    plt.title('Objective function change', fontsize=15)
    plt.ylabel('Variance', fontsize=13)
    plt.xlabel('Number of macrostates', fontsize=13)
    # see dendogram
    plt.subplot(nrow,ncol,2)
    scipy.cluster.hierarchy.dendrogram(clusters, labels=np.arange(len(x)))
    plt.xticks(fontsize=13)
    plt.title('Ward linkage', fontsize=15)
    #
    plt.tight_layout()
    plt.show()
    if(figname):
        fig.savefig(figname+'_cluster.png')


def plot_mixing(m_ica):
    plt.figure(figsize=(12, 6), dpi= 160, facecolor='w', edgecolor='k')
    plt.title('mixing matrix')
    plt.xlabel('IC')
    plt.ylabel('PC')
    plt.imshow(m_ica,cmap='plasma')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def get_labels(n):
    labels = []
    for i in np.arange(0,n,1):
        labels.append('v'+str(i+1))
    return labels

##############
# NEGENTROPY #
# related functions )
##############

def ave_score(X,n,niter=100,fun='logcosh'):
    # notice that X components are column-oriented
    score_ave=[]
    score_var=[]
    for i in np.arange(0,n,1):
        score_tmp = []
        for j in np.arange(0,niter,1):
            score_tmp.append(negent_score(X[:,i],fun))
        score_ave.append(np.mean(score_tmp))
        score_var.append(np.var(score_tmp))
    return score_ave,score_var

def negent_score(X,fun='logcosh'):
    # We compute J(X) = [E(G(X)) - E(G(Xgauss))]**2
    # We consider X (and Xgauss) to be white, in the sense that E(X,X.T)=I
    # The expectation being approximated by the sample mean in our case: np.dot(X,X.T)/n=I
    # In practice, we assume that X has already been normalized by its length [np.dot(X,X.T)=I]
    # so we rescale by np.sqrt(n) before we take the expectation value of G(X).
    length=len(X)
    Xscale = X*np.sqrt(length)
    Xgauss = np.random.randn(length)
    if(fun == 'logcosh'):
        n1 = np.mean(f_logcosh(Xscale)) #np.sum(f_logcosh(Xscale))
        n2 = np.mean(f_logcosh(Xgauss)) #np.sum(f_logcosh(Xgauss))
    elif(fun == 'exp'):
        n1 = np.mean(f_exp(Xscale))     #np.sum(f_exp(Xscale))
        n2 = np.mean(f_exp(Xgauss))     #np.sum(f_exp(Xgauss))
    elif(fun == 'rand'):
        n1 = np.mean(f_logcosh(Xgauss)) #np.sum(f_logcosh(Xgauss))
        n2 = 0 
    negent = (n2-n1)**2
    return negent

def f_logcosh(X):
    return np.log(np.cosh(X))

def f_exp(X):
    return -np.exp(-(X**2)/2)

