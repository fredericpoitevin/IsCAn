import numpy as np
from matplotlib import pyplot as plt
from msmbuilder.decomposition import tICA, PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize
from scipy import linalg

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

def biplots(prj,n=1,plottype='hexbin',nbins=10,figsize=(24,24)):
# plot populations in component space
    labels = get_labels(n) 
    fig = plt.figure(figsize=figsize, dpi= 160, facecolor='w', edgecolor='k')
    plt.title('population')
    nrow=n
    ncol=n
    color_hexbin='plasma'
    #nbins = 40
    nbins_coarse = int(nbins/1)
    nbox=1 #nrow
    for i in np.arange(0,n,1):
        for j in np.arange(0,n,1):
            ax = fig.add_subplot(nrow,ncol,nbox)
            plt.grid()
            if(i<j):
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
                        plt.scatter(Ax, Ay, cmap=color_hexbin)
                    else:
                        plt.hexbin(Ax, Ay, gridsize=nbins, cmap=color_hexbin, mincnt=1)
            elif(i==j):
                Ax = prj[:,i]
                plt.hist(Ax,bins=nbins_coarse)
                #plt.hist(Ay,bins=nbins,range=xlim_array,log=True,rwidth=0.4)
            else:
                if(j == 0):
                    ax.set_ylabel(labels[i])
                if(i == n - 1):
                    ax.set_xlabel(labels[j])
                Ax = prj[:,j]
                Ay = prj[:,i]
                if(plottype == 'scatter'):
                    plt.scatter(Ax, Ay, cmap=color_hexbin)
                else:
                    plt.hexbin(Ax, Ay, gridsize=nbins, cmap=color_hexbin, mincnt=1)
            nbox=nbox+1
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

