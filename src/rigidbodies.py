import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.spatial as sp_spatial
import scipy.cluster as sp_cluster
 
########################
# DOMAIN DECOMPOSITION #
########################
 
def rigidbodies(traj,pdb_output=None,pdb_keep_all=False,pdb_filter=False,cluster_output=False,cutoff=None,ndomains=2,similarity_type='distance_fluctuation',similarity_binarize=None):
    """rigidbodies

    Description
    -----------
    Rigid body decomposition through Ward clustering of a pariwise atom similarity matrix

    Parameters
    ----------
    - traj: MDtraj object
    - cutoff: maximum variance to identify the number of domains
    - ndomains: if cutoff is None, ndomains are assigned
    - similarity_type: 'distance_fluctuation' or 'fluctuation_range'
    - similarity_binarize: threshold to binarize matrix
    - pdb_output: if not None, write traj.slice(0) to PDB file, with domain assignment in B factor column
    - cluster_output: if True, return cluster information
    """
    clusters,similarity = cluster(traj,similarity_type=similarity_type,similarity_binarize=similarity_binarize)
    if cutoff is None:
        cutoff=1.0*np.amax(sp_spatial.distance.squareform(similarity))
    assignment = get_assignment(clusters,cutoff=cutoff,ndomains=ndomains)
    if pdb_output is not None:
        print(">WRITING ",pdb_output)
        if(pdb_keep_all):
            t = traj
        else:
            t = traj.slice(0)
        if(pdb_filter):
            assignment_new = assignment
            for wsize in np.arange(6,26,4):
                assignment_new = filter_assignment(t,assignment_new,wsize=wsize)
            assignment = assignment_new
        save_cluster_in_bfac(t,pdb_output,assignment)
    if(cluster_output):
        print(">OUTPUT CLUSTER INFO")
        return clusters,similarity,assignment

def cluster(traj,similarity_type='distance_fluctuation',similarity_binarize=None):
    """ cluster: 

    Description
    -----------
    domain decomposition through clustering of an atom pair similrity matrix

    Parameters
    ----------
    - traj: MDtraj object
    - similarity_type: string, optional
        . 'distance_fluctuation'
        . 'distance_range'
    - similarity_binarize: None or float
        if not None, the similarity matrix will be binarize around the value given (absolute or N sigma ?)

    """
    similarity = get_similarity(traj,similarity_type=similarity_type,similarity_binarize=similarity_binarize)
    clusters = sp_cluster.hierarchy.linkage(similarity, method='ward')
    return clusters, sp_spatial.distance.squareform(similarity)
 
def get_similarity(traj,similarity_type='distance_fluctuation',similarity_binarize=None):
    """ get_similarity
    """
    similarity = compute_similarity(traj,similarity_type='distance_fluctuation')
    similarity = binarize_similarity(similarity,similarity_binarize=similarity_binarize)
    return similarity

def binarize_similarity(similarity,similarity_binarize=None):
    """ binarize_similarity
    """
    if similarity_binarize is not None:
        print("... similarity stats before binarizing:")
        mean   = np.mean(similarity)
        std    = np.std(similarity)
        thresh = mean + similarity_binarize*std
        print("    mean: ",mean," +\- ",std," => threshold set at ",thresh)
        similarity[similarity >  thresh] = 1.0
        similarity[similarity <= thresh] = 0.0
    return similarity

def compute_similarity(traj,similarity_type='distance_fluctuation'):
    """ compute_similarity
    """
    dist1,dist2 = compute_similarity_tools(traj,similarity_type=similarity_type)
    for iframe in np.arange(traj.n_frames): 
        dist1,dist2 = compute_similarity_tools(traj.slice(iframe),d1=dist1,d2=dist2,tool='update',similarity_type=similarity_type)
    similarity = compute_similarity_tools(traj,d1=dist1,d2=dist2,tool='wrap',similarity_type=similarity_type)
    return similarity

def compute_similarity_tools(traj,d1=None,d2=None,tool='init',similarity_type='distance_fluctuation'):
    """ compute_similarity_tools
    """
    if(tool == 'init'):
        natoms = traj.n_atoms
        ndist  = int(natoms*(natoms-1)/2)
        if(similarity_type == 'distance_fluctuation'):
            dist1 = np.zeros(ndist)
            dist2 = np.zeros(ndist)
        elif(similarity_type == 'distance_range'):
            dist1 = np.zeros(ndist)
            dist2 = 1000*np.ones(ndist)
        return dist1,dist2
    elif(tool == 'update'):
        dist0 = sp_spatial.distance.pdist(traj.xyz.reshape(traj.n_atoms,3), metric='euclidean')
        if(similarity_type == 'distance_fluctuation'):
            dist1 = d1 + dist0
            dist2 = d2 + dist0**2
        elif(similarity_type == 'distance_range'):
            dist1 = np.maximum(dist0,d1)
            dist2 = np.minimum(dist0,d2)
        return dist1,dist2
    elif(tool == 'wrap'):
        if(similarity_type == 'distance_fluctuation'):
            nframes = traj.n_frames
            dist1 = d1/nframes
            dist2 = d2/nframes
            similarity = np.sqrt(dist2-dist1**2)
        elif(similarity_type == 'distance_range'):
            similarity = dmax - dmin
        return similarity

def get_assignment(l,cutoff=None,ndomains=2):
    """ get_assignment
    """
    if cutoff is not None:
        ndomains = get_nclusters(l,cutoff)
    return sp_cluster.hierarchy.fcluster(l, t=ndomains, criterion='maxclust')
 
def get_nclusters(clusters,cutoff):
    """ get_nclusters
    """
    nclusters=1
    keepongoing=True
    n = clusters.shape[0]
    for i in np.arange(1,n): #np.arange(0,n):
        score = clusters[i,2] - clusters[i-1,2]
        if(score > cutoff and keepongoing):
            nclusters = n-i + 1
            print("Number of domains: ",nclusters," (",score,")")
            keepongoing=False
    return nclusters

def filter_assignment(traj,assignment,wsize=10):
    """ filter_assignment
    """
    ass_new = assignment
    size_range = np.arange(4,wsize+1)
    for isize in size_range:
        ass_new = clean_assignment(traj,ass_new,wsize=isize)
    for isize in size_range[::-1]:
        ass_new = clean_assignment(traj,ass_new,wsize=isize)
    return ass_new
 
def clean_assignment(traj,assignment,wsize=10):
    """ clean_assignment
    """
    ass_new = assignment
    for i in np.arange(int(wsize/2),traj.n_atoms-int(wsize/2)):
        ilo = i - int(wsize/2)
        ihi = i + int(wsize/2) - 1
        if(ass_new[ihi]==ass_new[ilo] and ass_new[i]!=ass_new[ilo]):
            ass_new[i] = ass_new[ilo]
    return ass_new
 
def save_cluster_in_bfac(traj,filename,assignment):
    """ save_cluster_in_bfac
    """
    traj.save(filename, bfactors=assignment)

def get_residue_id(traj):
    """ get_residue_id
    """
    ires = np.zeros(traj.n_atoms)
    for i in np.arange(traj.n_atoms):
        ires[i] = int(str(traj.topology.residue(i))[3:])
    return ires


