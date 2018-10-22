import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import mdtraj as md

def load_dataset(pdb_filename,ids_filename='',keep_mode='intersect',superpose=False):
    """
    purpose: load both traj and ids from file.
    options: if ids_filename is provided, either only keep its ids (keep_mode intersect), 
             or ignore them on the contrary (keep_mode ignore)
    """
    ids  = load_ids(pdb_filename)
    traj = load_traj(pdb_filename,superpose=superpose)
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
    line = genfromtxt(filename, max_rows=1, delimiter=' ', dtype=(str))
    cif_id = line[3:len(line)]
    return cif_id

def load_traj(filename,superpose=False):
    traj = md.load(filename)
    if(superpose):
        traj.superpose(traj, 0)
    return traj

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
