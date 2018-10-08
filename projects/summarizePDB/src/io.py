import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import mdtraj as md

def load_dataset(filename,superpose=False):
    ids  = load_ids(filename)
    traj = load_traj(filename,superpose=superpose)
    if(len(ids) != traj.n_frames):
        print('Warning: load_dataset inconsistency')
    return traj,ids

def load_ids(filename):
    line = genfromtxt(filename, max_rows=1, delimiter=' ', dtype=(str))
    cif_id = line[3:len(line)]
    return cif_id

def load_traj(filename,superpose=False):
    traj = md.load(filename)
    if(superpose):
        traj.superpose(traj, 0)
    return traj

