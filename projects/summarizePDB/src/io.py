import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import mdtraj as md

def load_dataset(keyword,superpose=False):
    filename='data/'+keyword+'.id'
    ids = load_ids(filename)
    filename='data/'+keyword+'.pdb'
    traj = load_traj(filename,superpose=superpose)
    if(len(ids) != traj.n_frames):
        print('Warning: load_dataset inconsistency')
    return traj,ids

def load_ids(filename):
    cif_id = genfromtxt(filename, max_rows=1, delimiter=' ', dtype=(str))
    return cif_id

def load_traj(filename,superpose=False):
    traj = md.load(filename)
    if(superpose):
        traj.superpose(traj, 0)
    return traj
