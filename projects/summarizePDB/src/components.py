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
    """filter: identify and remove outliers
    
    Parameters
    ----------
    traj : mdtraj object
    
    ids : list of string
        must be of length traj.n_frames

    pc_thresh : float, optional (default: 0.75)
        threshold value used to truncate PCA at desired variance ratio

    ic_thresh : float, optional (default: 1.0)
        threshold value used to identify outliers along an independent component

    fun : string, optional (default: 'logcosh')
     [to be finished]
    
    """
    traj_new, ids_new = traj_slice(traj,ids)
    if(ic_thresh < 1.0):
        it=0
        iterate=True
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
    """cluster: cluster projection of data in component space
    
    Description
    -----------
    
    See description of get_distance for the definition of the distance in component space.
    This distance is used to perform the (Ward) hierarchical clustering.

    Input
    -----
        traj : TBD
        ids : TBD

    Parameters
    ----------
        analysis_type : string, optional (default: 'ica')
            Possible values are:
            . 'ica' : negentropy-weigthed distance in IC space 
            . 'pca' : variance-weigthed distance in PC space
            . 'pcaJ': negentropy-weighted distance in PC space
        === Component analysis parameters ===
        pc_thresh :
        fun : 
        algo :
        === Verbose and plot parameters ===
        do_plot : 
        title : 

    Returns
    -------
        clusters: ndarray
            [result of scipy.cluster.hierarchy.linkage with method='ward']

    """
    distance, x, weight = get_distance(traj,ids,analysis_type=analysis_type,pc_thresh=pc_thresh,fun=fun,algo=algo) 
    clusters = scipy.cluster.hierarchy.linkage(distance, method='ward')
    if(do_plot):
        plot_cluster(clusters,x,figname=title)
    return clusters

def get_distance(traj,ids,analysis_type='ica',pc_thresh=0.75,fun='logcosh',algo='parallel'):
    """get_distance: compute distance in component space
    
    Description
    -----------

    Consider N data points X_i to decompose.
    Note Q_k the k-thcomponent and x_ik the coordinates of X_i on component Q_k, such that:
    [1] X_i = \sum_k x_ik Q_k
    Each component k is attributed a weight w_k, either the variance of the projection of the data along it, or its negentropy.
    The distance between a pair of data points (i,j) in the component space is defined as:
    [2] d_ij = (\sum_k w_k ( x_jk - x_ik )**2 )**1/2
    
    === more verbosity to come ===
    
    """
    atype='pca' 
    if(analysis_type=='ica'):
        atype='ica'
    v,m,x = analyses(traj,ids,pc_thresh=pc_thresh,fun=fun,algo=algo,analysis_type=atype,do_plot=False)
    if(analysis_type=='pca'):
        weight = m
    else:
        negent_ave, negent_var = ave_score(x,len(x[0,:]),fun=fun)
        weight = negent_var
    distance = scipy.spatial.distance.pdist(x, metric='minkowski',p=2,w=weight)
    return distance, x, weight

def analyses(traj,ids,pc_thresh=0.75,fun='logcosh',algo='parallel',analysis_type='ica',do_plot=True,c=None,title=''):
    """analyses: perform PCA to reduce dimensionality, and optionaly ICA to identify relevant components
    
    Description
    -----------

    Parameters
    ----------

    Returns
    -------

    """
    v_pca, l_pca, x_pca = traj2pc(traj, n_components=len(ids),var_trunc=pc_thresh)
    x = x_pca
    if(do_plot):
        print('Principal Component Analysis (step 0)')
        score = score_mode(traj,ids,v=None,pc_thresh=pc_thresh)
        plot_stats_CA(x_pca,l_pca,len(l_pca),score=score,threshold=0,analysis_type='pca',figname=title)
    if(analysis_type == 'ica'):
        v_ica, m_ica, x_ica = traj2ic(x_pca,n_components=len(l_pca),fun=fun,algo=algo)
        x = x_ica
        if(len(l_pca)>1 and do_plot):
            print('Independent Component Analysis (step 1)')
            score = score_mode(traj,ids,v=v_ica,pc_thresh=pc_thresh)
            plot_stats_CA(x_ica,m_ica,len(l_pca),score=score,threshold=0,figname=title)
    if(do_plot):
        print('projection of data in component space')
        biplots(x,prj2=x_pca,n=np.minimum(10,len(l_pca)),nbins=30,c=c,figname=title)
    if(analysis_type == 'ica'):
        return v_ica, m_ica, x_ica
    else:
        return v_pca, l_pca, x_pca

def save_mode(traj,ids,prj,n=np.arange(1),v=None,pc_thresh=0.75,keyword='mode',movie='oscillatory',nframe=20,verbose='minimal'):
    """ save_mode : writes PDB file with information on a component mode

    Description
    -----------

    Parameters
    ----------
    mode: string, optional
        . 'oscillatory' (default)
        . 'sorted' (not implemented yet - see options in function 'get_sorted_index')
        . 'projected' (not implemented yet)
    """
    if v is None:
        mode_type='PC'
    else:
        mode_type='IC'
    key=keyword+'_'+movie+'_'+mode_type
    xyz_mean, b_factors = get_xyz_mean(traj)
    xyz_mode = get_mode(traj, ids, v=v, pc_thresh=pc_thresh)
    for ic in n:
        index = get_sorted_index(prj,ic,nICs=n)
        traj_IC, ids_IC = traj_slice(traj,ids,index=index)
        if(movie=='oscillatory'):
            traj_IC = traj[0:nframe]
            for iframe in np.arange(0,nframe,1):
                amplitude = np.amax(np.abs(prj[:,ic]))
                phase = iframe*2.*np.pi/nframe
                traj_IC.xyz[iframe:iframe+1,:] = amplitude*xyz_mode[ic,:].reshape(1,traj.n_atoms,3)*np.sin(phase)
                traj_IC.xyz[iframe:iframe+1,:] += xyz_mean.reshape(traj.n_atoms,3)
        save_traj(traj_IC,ids_IC,keyword=key+str(ic+1),verbose=verbose)

def get_mode(traj,ids,v=None,pc_thresh=0.75):
    """ get_mode : returns mode along one component.
    
    Description
    -----------
    Given a decomposition of the data X_i = \sum_k x_ik Q_k, we call Q_k the modes.
    It is a difference vector with a definition that varies in PC and IC space.
    In PC-space, x_ik corresponds to the V-matrix of SVD, and Q is the variance-weigthed U-matrix.
    In IC-space, x_ij corresponds to the sources, and Q is the unmixed Q matrix of PC-space.

    Parameters
    ----------
    If v is None, then we output the PC modes. 
    Otherwise, v is assumed to be the unmixing matrix pre-computed with ICA.
    """
    v_pca, l_pca, x_pca = traj2pc(traj, n_components=len(ids),var_trunc=pc_thresh)
    xyz_mode = np.dot(np.diag(l_pca),v_pca)
    if v is not None:
        xyz_mode = np.dot(v,xyz_mode)
    return xyz_mode

def score_mode(traj,ids,v=None,pc_thresh=0.75,score_type='elasticity'):
    """ score_mode
    """
    if v is None:
        mode_type='PC'
    else:
        mode_type='IC'
    xyz_mean, b_factors = get_xyz_mean(traj)
    xyz_mode = get_mode(traj, ids, v=v, pc_thresh=pc_thresh)
    n = xyz_mode.shape[0]
    mode_score = np.zeros(n)
    for ic in np.arange(0,n,1):
        score = compute_mode_score(xyz_mode[ic,:],mean=xyz_mean,score_type=score_type)
        mode_score[ic] = score
    return mode_score

def compute_mode_score(xyz,mean=None,score_type='harmonicity'):
    """ compute_mode_score: provide some energy score per mode

    Arguments
    ---------
    score_type : string, optional
        - 'harmonicity' (default) computes the harmonic score
        - 'elasticity' computes the elastic score

    """
    if(score_type=='harmonicity'):
        score = compute_harmonic_score(xyz)
    else:
        score = compute_elastic_score(xyz,mean=mean)
    return score

def compute_harmonic_score(xyz):
    """ compute_harmonic_score

    Description
    -----------

    Assuming the mode is harmonic, its free energy is given by
    F = -logZ with Z = <exp(- 0.5* X H X)>
    and the Hessian is inverse of covariance H**-1 = <XXt>
    Up to a constant, we got: F = -0.5 * log Det <XXt>

    """
    sign, score = np.linalg.slogdet(np.outer(xyz,xyz))
    score = -0.5*score
    return score

def compute_elastic_score(xyz,mean=None):
    """ compute_elastic_score :
    mean should not be None, but maybe something could be worked out for that case too...

    Description
    -----------

    the idea is to compute the elastic energy of the mode.
    For atom pairs within a distance cutoff, compute (dij - dij_0)**2
    Then sum: E = sum_ij (dij - dij_0)**2
    (requires additional input: xyz_mean)

    """
    natoms = int(len(mean)/3)
    dist0 = scipy.spatial.distance.pdist(mean.reshape(natoms,3), metric='euclidean')
    xyz = mean + xyz
    dist1 = scipy.spatial.distance.pdist(xyz.reshape(natoms,3), metric='euclidean')
    dist0_masked = np.ma.masked_where(dist0>5, dist0)
    dist0_kept = np.ma.compressed(dist0_masked)
    dist1_masked = np.ma.masked_array(dist1, dist0_masked.mask)
    dist1_kept = np.ma.compressed(dist1_masked)
    diff = (dist1_kept - dist0_kept)**2
    return np.mean(diff)

def save_traj(traj,ids,keyword='traj',save_mean=False,verbose='minimal'):
    """ save_traj : writes traj to PDB, or just its sample mean, with ids list.
    """
    if(len(ids) > 0):
        if(save_mean):
            filename=keyword+'_mean.pdb'
        else:
            filename=keyword+'.pdb'
        write_trj_to_file(filename,traj,save_mean=save_mean)
        write_ids_to_file(filename,ids)
        if(verbose=='full'):
            print('wrote ',filename,' : ',ids)

def write_trj_to_file(filename,traj,save_mean=False):
    """ write_trj_to_file: write traj to file, or just its sample mean
    """
    if(save_mean):
        xyz_mean, b_factors = get_xyz_mean(traj)
        traj_mean = traj[0]
        traj_mean.xyz = xyz_mean.reshape(traj.n_atoms, 3)
        traj_mean.save(filename, bfactors=np.clip(b_factors, -9, 99))
    else:
        traj.save(filename)

def write_ids_to_file(filename,ids):
    """ write_ids_to_file: add info of ID list at beginning of file
    """
    with open(filename,'r') as f:
        save = f.read()
    with open(filename, 'w') as f:
        f.write("REMARK ID list: ")
        for item in ids:
            f.write("%s " % item)
        f.write('\n')
    with open(filename, 'a') as f:
        f.write(save)

def traj2pc(traj,n_components=1,negent_sort=False,var_trunc=-1):
    """ traj2pc : 
    """
    n_cmpnt = n_components
    xyz = get_xyz(traj,centered=True)
    v_pc, l_pc = get_pca(xyz,n_components=n_components)
    x_pc = proj(v_pc,xyz)
    # truncate if asked to
    if(var_trunc > 0):
        n_cmpnt = get_truncate_order(l_pc,var_trunc)
        v_pc, l_pc, x_pc = truncate_svd(v_pc, l_pc, x_pc, n=n_cmpnt)
    # sort by decreasing order of negentropy if asked to
    if(negent_sort):
        Jscore,Jtmp = ave_score(x_pc,n_cmpnt)
        index = np.argsort(Jscore)[::-1]
        v_pc = v_pc[index,:]
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
    """ truncate_svd : truncates all SVD matrices with first n components
    Conventions
    -----------
    Careful we do not really follow the intuitive ordering...
    With our notations, with X (n_sample,n_xyz), this is what is implicitely done:
        X_exact  = (U.T L V.T).T 
        X_approx = (U[0:n,:].T L[0:n] V[:,0:n].T).T
    """
    if(n==-1):
        n=L.shape[0]
    Ul = U[0:n,:]
    Ll = L[0:n]
    Vl = V[:,0:n]
    return Ul, Ll, Vl

#################
# Preprocessing #
#################

def get_xyz(traj,centered=False):
    """ get_xyz : yields atomic coordinates
    Input: MDtraj trajectory object 
    Returns xyz, centered if asked
    """
    xyz = traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3)
    if(centered):
        xyz_mean, bfac = get_xyz_mean(traj)
        xyz = xyz - xyz_mean
    return xyz

def get_xyz_mean(traj):
    """ get_xyz_mean : gives coordinate sample mean, and B factors
    Input: MDtraj trajectory objects
    Returns sample mean of traj.xyz, and B factors
    """
    xyz       = traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3)
    xyz_mean  = np.mean(xyz, axis=0)
    xyz_std2  = np.std(xyz, axis=0)**2
    b_factors = np.sum(xyz_std2.reshape(traj.n_atoms, 3), axis=1)*8*(np.pi)**2
    return xyz_mean, b_factors

def traj_slice(traj,ids,index=[]):
    if(len(index) == 0):
        index = np.arange(len(ids))
    return traj.slice(index), ids[index]

############################## 
# Component Analyses Methods #
##############################

def get_svd(traj):
    """ get_svd : Singular Value Decomposition
    if traj is (n_sample, n_features), Vh: components. Otherwhise, U
    """
    U,s,Vh = linalg.svd(traj)
    return U,s,Vh

def get_pca(traj,n_components=1):
    """ get_pca : Principal Component Analysis
    Returns
    -------
    pca.components_ : (n_frame, n_components)
    pca.singular_values_ : (n_components)
    """
    pca = PCA(n_components=n_components,svd_solver='full')
    pca.fit([traj])
    return pca.components_, pca.singular_values_

def get_ica(traj,n_components=1,fun='logcosh'):
    """ get_ica : Independent Component Analysis
    """
    ica = FastICA(n_components,fun=fun)
    ica.fit(traj)
    return ica.components_

def get_tica(traj,n_components=1,lag_time=100):
    """ get_tica : time-structured ICA
    """
    tica = tICA(n_components,lag_time)
    tica.fit([traj])
    return tica.components_, tica.eigenvalues_

def proj(component,xyz):
    """ proj : project data on components, and normalize
    Description
    -----------
    xyz is (n_sample,n_xyz), component is (n_component,n_xyz)
    normed = xyz.(component.T) is (n_sample,n_component)
    """
    proj = np.dot(xyz,component.T)
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

def biplots(prj,prj2=None,n=1,plottype='hexbin',nbins=10,figsize=-1,c=None,figname=''):
    """ biplots : plot populations in component space

    Description
    -----------
    For n components, shows biplots between all pairs of prj in upper triangle.
    If prj2 is provided, shows biplots between all pairs of prj2 in lower one.
    The possibility to color based on input assignment is offered.

    """
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
    nbins_coarse = int(nbins/1)
    nbox=1 
    for i in np.arange(0,n,1):
        for j in np.arange(0,n,1):
            if(i<j):
                ax = fig.add_subplot(nrow,ncol,nbox)
                plt.grid()
                #if(j<n):
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
                if prj2 is not None:
                    Ay = prj2[:,i]
                    plt.hist(Ay,bins=nbins_coarse,rwidth=0.4)
            else:
                if prj2 is None:
                    if(i == 1):
                        if(j == 0):
                            ax = fig.add_subplot(nrow,ncol,nbox)
                            ax.set_xlabel('cluster color')
                            xy = range(1,np.max(c)+1,1)
                            sc = plt.scatter(xy, xy, c=xy, vmin=1, vmax=np.max(c), cmap=cmap)
                            plt.colorbar(sc)
                else:
                    ax = fig.add_subplot(nrow,ncol,nbox)
                    plt.grid()
                    if(j == 0):
                        ax.set_ylabel(labels[i])
                    if(i == n - 1):
                        ax.set_xlabel(labels[j])
                    Ax = prj2[:,j]
                    Ay = prj2[:,i]
                    if(plottype == 'scatter'):
                        plt.scatter(Ax, Ay, c=c, cmap=cmap)
                    else:
                        plt.hexbin(Ax, Ay, gridsize=nbins, cmap=cmap, mincnt=1)
            nbox=nbox+1
    plt.tight_layout()
    plt.show()
    if(figname):
        fig.savefig(figname+'_biplot.png')

def plot_stats_CA(prj,l,n_components=1,score=None,fun='logcosh',threshold=0.9,niter=100,span=True,figsize=-1,analysis_type='ica',figname=''):
    """ plot_stats_CA : ...
    """
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
    ncol=1
    if(span):
        ncol+=1
    if score is not None:
        ncol+=1
    idx=1
    if(n_components>1):
        ncol+=1
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
    if score is not None:
        idx+=1
        # - SCORE
        plt.subplot(nrow,ncol,idx)
        plt.grid()
        plt.title('Score of component')
        plt.xlabel('ID')
        plt.ylabel('elastic energy')
        plt.plot(range(1,1+len(score)), score, 'ko')
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

