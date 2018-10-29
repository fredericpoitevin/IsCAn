import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import mdtraj as md
import scipy
import scipy.spatial

def load_dataset(pdb_filename,ids_filename='',keep_mode='intersect',superpose=False,pdb_clean=False,neighbour_cutoff=5.0,Nsigma=3):
    """ load_dataset 

    Description
    -----------
    Loads the content of a PDB file containing a trajectory.
    Careful: the file needs to have as its irst line a "REMARK ID list:" tag followed by a list of names (or ids) for each of the frame
    
    Parameters
    ----------
    - ids_filename:  string, optional
        if provided, either only keep its ids (keep_mode intersect), or ignore them on the contrary (keep_mode ignore)
    - keep_mode: string, optional
        see ids_filename
    - superpose: True or False
    - pdb_clean: True or False
        compute local chain elastic energy of each atom, and remove isolated or outliers.

    """
    ids  = load_ids(pdb_filename)
    traj = load_traj(pdb_filename,superpose=superpose,pdb_clean=pdb_clean,neighbour_cutoff=neighbour_cutoff,Nsigma=3)
    if(ids_filename):
        ids_keep = load_ids(ids_filename)
        if(keep_mode=='intersect'):
            mask = np.in1d(ids,ids_keep)
        elif(keep_mode=='ignore'):
            mask = np.logical_not(np.in1d(ids,ids_keep))
        else:
            print("Error...")
        ids_new  = ids[mask]
        traj_new = traj[mask]
        if(superpose):
            traj_new.superpose(traj, 0)
    else:
        ids_new  = ids
        traj_new = traj
    if(len(ids_new) != traj_new.n_frames):
        print('Warning: load_dataset inconsistency')
    return traj_new,ids_new

def load_ids(filename):
    """ load_ids : 
    """
    line = genfromtxt(filename, max_rows=1, delimiter=' ', dtype=(str))
    cif_id = line[3:len(line)]
    return cif_id

def load_traj(filename,superpose=False,pdb_clean=False,neighbour_cutoff=5.0,Nsigma=3):
    """ load_traj : 
    """
    traj = md.load(filename)
    if(superpose):
        traj.superpose(traj, 0)
    if(pdb_clean):
        traj = clean_pdb(traj,neighbour_cutoff=neighbour_cutoff,Nsigma=3)
    return traj

def clean_pdb(traj,neighbour_cutoff=5.0,Nsigma=3):
    """ pdb_clean
    """
    print("Initial number of atoms ",traj.n_atoms)
    traj.superpose(traj, 0)
    atom_indices = pdb_clean_get_atom_indices(traj,neighbour_cutoff=neighbour_cutoff,Nsigma=3)
    traj.atom_slice(atom_indices,inplace=True)
    traj.superpose(traj, 0)
    print("... after cleaning: ",traj.n_atoms)
    return traj

def pdb_clean_get_atom_indices(traj,neighbour_cutoff=5.0,Nsigma=3):
    """ pdb_clean_get_atom_indices

    Description
    -----------
    this is not ideal, but here is the idea:
    for each atom i, consider the two atoms that precede and follow it in the sequence, 
    and keep those that are within cutoff distance from atom i, in average.
    If none is kept, atom i is dropped.
    If more than one is kept, a score is given to atom i:
        E_i = \sum_j^neighbours ( max(dij) - min(dij))**2
    Then if E_i > mean(E_i) + Nsigma*std(E_i), then i is dropped.

    Parameters
    ----------
    - traj: MDtraj object
    - neighbour_cutoff: float, optional
    """
    indices = []
    scores  = np.zeros(traj.n_atoms)
    for i in np.arange(0,traj.n_atoms,1):
        i_neigh = get_i_neigh(traj,i,neighbour_cutoff=neighbour_cutoff)
        for j in i_neigh:
            i_list = [i,j]
            scores[i] += get_i_score(traj,i_list)/len(i_neigh)
    score_cutoff = np.mean(scores) + Nsigma*np.std(scores)
    for i in np.arange(0,traj.n_atoms,1):
        if(scores[i] < score_cutoff):
            indices.append(i)
    return indices

def get_i_neigh(traj,i,neighbour_cutoff=5.0):
    """ get_i_neigh
    """
    i_min = 0
    i_max = traj.n_atoms - 1
    i_neigh = []
    if( i != i_min ):
        i_list = [i-1,i]
        dist = get_dist(traj,i_list,mean=True)
        if(dist <= neighbour_cutoff):
            i_neigh.append(i-1)
    if( i != i_max ):
        i_list = [i,i+1]
        dist = get_dist(traj,i_list,mean=True)
        if(dist <= neighbour_cutoff):
            i_neigh.append(i+1)
    return i_neigh

def get_dist(traj,i_list,mean=False):
    """ get_dist
    """
    tij = traj.atom_slice(i_list,inplace=False)
    if(mean):
        xyz_ij_mean = np.mean(tij.xyz.reshape(tij.n_frames, tij.n_atoms * 3),axis=0)
        dist = scipy.spatial.distance.pdist(xyz_ij_mean.reshape(tij.n_atoms,3), metric='euclidean')
    else:
        dist = []
        for t in np.arange(0,tij.n_frames,1):
            frame = tij.slice(t)
            dist_t = scipy.spatial.distance.pdist(frame.xyz.reshape(tij.n_atoms,3), metric='euclidean')
            dist.append(dist_t)
    return dist

def get_i_score(traj,i_list):
    """ get_i_score
    """
    dist  = get_dist(traj,np.sort(i_list))
    score = (np.amax(dist)-np.amin(dist))**2
    return score

def merge_pdb_list(filelist='',output='merged.pdb',superpose=False):
    if(filelist):
        with open(output,'w') as fwritten:
            for item in filelist:
                with open(item,'r') as fread:
                    item_read = fread.read()
                fwritten.write(item_read)
                fwritten.write('ENDMDL\n')
        traj = load_traj(output,superpose=superpose)
        traj.save(output)
