import numpy as np
from numpy.linalg import svd

def get_linear_response_force(traj_source,traj_target,filename='test.pdb'):
    """ get_linear_reponse_force

    Description
    -----------
        [(-dx)----X0----(+dx)].......[X1]
            Cov0 = dx dx.T & H0 = Cov0**-1
            X   = X1 - X0
        The linear response force to go from X0 to X1 is:
            F = H0 X

    Parameters
    ----------
    - traj_source : 2-frame traj where first frame is mean and second is elementary displacement
    - traj_target : structural mean to target 
    """
    dx, X   = get_displacements(traj_source,traj_target)
    hessian = compute_hessian_from_displacement(dx)
    force   = compute_force_from_hessian(hessian,X,output='norm')
    write_force_to_pdbfile(traj_source,traj_target,force,filename)

def write_force_to_pdbfile(t_source,t_target,force,filename):
    """ write_force_to_pdbfile
    """
    traj = t_source
    traj.xyz[1,:,:] = t_target.xyz[0,:,:]
    traj.save(filename,bfactors=np.clip(force, -9, 99))

def compute_force_from_hessian(hessian,X,output='norm'):
    """ compute_force_from_hessian
    """
    force = np.dot(hessian,X)
    if(output=='norm'):
        force = force**2
        force = np.sqrt(np.sum(force.reshape(int(force.shape[0]/3), 3), axis=1))
    return force

def compute_hessian_from_displacement(dx):
    """ compute_hessian_from_displacement
    """
    covariance = np.outer(dx,dx)
    u, s, vh   = svd(covariance,full_matrices=True)
    hessian    = np.outer(u[:,0],vh[0,:])/s[0]
    return hessian

def get_displacements(traj_source,traj_target):
    """ get_displacements
    """
    xyz_source = traj_source.xyz.reshape(traj_source.n_frames, traj_source.n_atoms*3)
    xyz_target = traj_target.xyz.reshape(traj_target.n_frames, traj_target.n_atoms*3)
    dx = xyz_source[1,:] - xyz_source[0,:]
    X  = xyz_target[0,:] - xyz_source[0,:]
    return dx, X
