import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.spatial as sp_spatial
import scipy.cluster as sp_cluster
from sklearn import cluster as sl_cluster
#
def cluster(traj,method='ward',n_clusters=2):
    fluct,dist = get_distance_fluctuation(traj)
    if(method=='ward'):
        clusters = sp_cluster.hierarchy.linkage(fluct, method='ward')
        return clusters
    elif(method=='spectral'):
        similarity  = np.exp(-0.5*(fluct/fluct.mean())**2)
        #similarity *= np.exp(-0.5*(dist/10.)**2)
        similarity  = sp_spatial.distance.squareform(similarity)
        spectral = sl_cluster.SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
        spectral.fit_predict(similarity)
        return spectral
    else:
        print("please use either ward or spectral")
#
def get_assignment(l,n_clusters,method='ward'):
    if(method=='ward'):
        return sp_cluster.hierarchy.fcluster(l, t=n_clusters, criterion='maxclust')
    elif(method=='spectral'):
        return l.labels_.astype(np.int)
    else:
        print("please use either ward or spectral")
#
def filter_assignment(traj,assignment,wsize=10):
    ass_new = assignment
    size_range = np.arange(4,wsize+1)
    for isize in size_range:
        ass_new = clean_assignment(traj,ass_new,wsize=isize)
    for isize in size_range[::-1]:
        ass_new = clean_assignment(traj,ass_new,wsize=isize)
    return ass_new

#
def clean_assignment(traj,assignment,wsize=10):
    ires = get_residue_id(traj)
    ass_new = assignment
    for i in np.arange(int(wsize/2),traj.n_atoms-int(wsize/2)):
        ilo = i - int(wsize/2)
        ihi = i + int(wsize/2) - 1
        #if(ires[ihi] - ires[ilo] < wsize):
        if(ass_new[ihi]==ass_new[ilo] and ass_new[i]!=ass_new[ilo]):
            ass_new[i] = ass_new[ilo]
    return ass_new
#
def save_cluster_in_bfac(traj,filename,assignment):
    traj.save(filename, bfactors=assignment)
#
def get_distance_fluctuation(traj):
    nframes = traj.n_frames
    natoms  = traj.n_atoms
    ndist   = int(natoms*(natoms-1)/2)
    #
    xyz = traj.xyz.reshape(nframes, natoms * 3)
    dist1 = np.zeros(ndist)
    dist2 = np.zeros(ndist)
    for iframe in np.arange(nframes):
        dist0   = sp_spatial.distance.pdist(xyz[iframe,:].reshape(natoms,3), metric='euclidean')
        dist1  += dist0
        dist2  += dist0**2
    dist1 /= nframes
    dist2 /= nframes
    fluct = np.sqrt(dist2-dist1**2)
    return fluct,dist1

def get_residue_id(traj):
    ires = np.zeros(traj.n_atoms)
    for i in np.arange(traj.n_atoms):
        ires[i] = int(str(traj.topology.residue(i))[3:])
    return ires
