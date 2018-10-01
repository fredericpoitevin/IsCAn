#!/Users/fpoitevi/anaconda3/bin/python
# IsCAn
#
# purpose: window-scans data PC-space to do ICA
# license: MIT License
# author: Frederic Poitevin
# e-mail: frederic.poitevin@gmail.com
#

import time

import matplotlib.pyplot as plt
import matplotlib.axis as axis

import numpy as np
import numpy.random
from numpy import linalg as nLA

from scipy import linalg as sLA
from scipy.stats import ortho_group
import scipy.stats as ss

from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA

import mdtraj as md
import simtk.openmm as mm
from msmbuilder.decomposition import tICA

########
# LOAD #
########

def load_xyz(filename):
    # purpose: returns coordinates from a MD trajectory
    traj = md.load(filename)
    traj.superpose(traj, 0)
    xyz = traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3)
    return xyz

def load(filename):
    # purpose: returns array previously saved
    # e.g.: > data = IsCAn.load_xyz('data.pdb')
    #       > np.save('data.npy',data)
    data = np.load(filename)
    print(f"DATA is assumed to be (n_feature,n_sample), is it? ",data.shape)
    return data

###############
# PRE-PROCESS #
###############

def center(data):
    # assuming (feature,sample), the sample_mean is
    print(f"> Retrieving sample mean and centering DATA on it")
    #mean = np.mean(data,axis=0)
    #centered = data - mean.T
    #return mean.T, centered
    mean = np.mean(data,axis=1)
    centered = (data.T - mean).T
    return mean, centered

def whiten(data,whiten='PCA',n=-1):
    # here we perform a SVD of the data,
    # truncate to order n
    # and whiten according to whiten
    U,L,V = perform_svd(data)
    if(n==-1):
        n=L.shape[0]
    if(whiten=='PCA'):
        whitened = V[:,0:n]
    elif(whiten=='ZCA'):
        whitened = np.dot(U[:,0:n],V[:,0:n].T)
    return whitened

########
# PLOT #
########

def plot(data,title=''):
    dimension=len(data.shape)
    fig = plt.figure()
    plt.title(title)
    if(dimension==1):
        plt.plot(data)
    else:
        ratio = data.shape[1]/data.shape[0]
        if(ratio>=5 or  ratio<=1/5):
            plt.plot(data)
        else:
            plt.imshow(data)

def plot_truncate_order(L,trange=np.arange(0,1,0.1)):
    x,y = scan_truncate_order(L,trange=trange)
    plt.ylabel('explained variance ratio')
    plt.xlabel('truncating order (Lorder)')
    plt.plot(y,x)

def biplots(prj,n=2,plottype='hexbin',nbins=10,figsize=6):
    labels = get_labels(n) 
    fig = plt.figure(figsize=(figsize, figsize), dpi= 160, facecolor='w', edgecolor='k')
    nrow=n
    ncol=n
    color_hexbin='plasma'
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

#######
# PCA #
#######

def perform_svd(data):
    # purpose: SVD of data. 
    # note: L is an array here, and V is transposed
    print(f"> Performing SVD\n ")
    U,L,V = nLA.svd(data,full_matrices=True)
    return U,L,V.T

def backward_svd(U,L,V,n=-1):
    # exact backward of perform_svd
    if(n==-1):
        n=L.shape[0]
    Un = U[:,0:n]
    Ln = np.diag(L[0:n])
    Vn = V[:,0:n]
    return np.dot(Un,np.dot(Ln,Vn.T)).T

def get_rank(L,tol=1e-10):
    return nLA.matrix_rank(np.diag(L),tol)

def get_truncate_order(L,threshold=0.9):
    var = L**2
    var /= np.sum(var)
    nPCs = 1
    if(threshold==1.0):
        nPCs = len(var)
    else:
        for i in np.arange(0,len(var)-1,1):
            var_current = np.cumsum(var)[i]
            var_next = np.cumsum(var)[i+1]
            if(var_current < threshold and var_next > threshold):
                nPCs=i+1
    return nPCs

def truncate_svd(U,L,V,n=-1):
    if(n==-1):
        n=L.shape[0]
    Ul = U[:,0:n]
    Ll = L[0:n]
    Vl = V[:,0:n]
    return Ul, Ll, Vl

def scan_truncate_order(L,trange=np.arange(0,1.1,0.1)):
    order = []
    for thresh in trange:
        n = get_truncate_order(L,threshold=thresh)
        order.append(n)
    return trange,order

#######
# ICA #
#######

def window_scan(data,Lorder=10,window_size_max=1,algorithm='parallel',fun='logcosh'):
    wstart = []
    wend   = []
    score  = []
    for window_size in np.arange(1,window_size_max): 
        for window_start in np.arange(Lorder-window_size):
            window_end = window_start + window_size
            window=data[:,window_start:window_end]
            wstart.append(window_start)
            wend.append(window_end)
            score.append(get_window_score(window,algorithm=algorithm,fun=fun))
    return wstart, wend, score

def get_window_score(window,algorithm='parallel',fun='logcosh'):
    window_size = window.shape[1]
    PCscore, PCscore_var = ave_score(window,window_size)
    S, A, W = perform_ica(window,algorithm=algorithm,fun=fun)
    ICscore, ICscore_var = ave_score(S,window_size)
    score = []
    score.append(PCscore)
    score.append(ICscore)
    score.append((ICscore-np.amax(PCscore))/np.sqrt(np.amax(PCscore_var)))
    score = (ICscore-np.amax(PCscore))/np.sqrt(np.amax(PCscore_var))
    return score

def perform_ica(X,algorithm='parallel',fun='logcosh'):
    #                   X = S A.T <=> S = X W.T
    # X (n_samples, n_features)    / S (n_samples, n_components)
    # A (n_features, n_components) / W (n_components, n_features)
    #ica = FastICA(whiten=False,algorithm=icalgo,fun=nongauss,max_iter=ica_iter, tol=ica_tol)
    ica = FastICA(whiten=True,algorithm=algorithm,fun=fun)
    S = ica.fit_transform(X) # Fit and apply the unmixing matrix to recover the sources
    A = ica.mixing_          # The mixing matrix
    W = ica.components_      # The unmixing matrix
    return S,A,W

def ave_score(X,n,niter=100,fun='logcosh'):
    score_ave=[]
    score_var=[]
    for i in np.arange(0,n,1):
        score_tmp = []
        for j in np.arange(0,niter,1):
            score_tmp.append(negent_score(X[:,i],fun=fun))
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

#######################
# AUTOMATIC RUN BELOW #
#######################
#def main():

    # Initialize IsCAn
    #iscan = IsCAn()

    #iscan.run()

#if __name__ == '__main__':
#    main()
