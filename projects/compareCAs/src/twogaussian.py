# adapted from:
# http://msmbuilder.org/development/examples/tICA-vs-PCA.html
#
import numpy as np
from matplotlib import pyplot as plt
import simtk.openmm as mm

def plot_pot():
    xx, yy = np.meshgrid(np.linspace(-2,2), np.linspace(-3,3))
    zz = 0 # We can only visualize so many dimensions
    ww = 5 * (xx-1)**2 * (xx+1)**2 + yy**2 + zz**2
    c = plt.contourf(xx, yy, ww, np.linspace(-1, 15, 20), cmap='viridis_r')
    plt.contour(xx, yy, ww, np.linspace(-1, 15, 20), cmap='Greys')
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('$y$', fontsize=18)
    plt.colorbar(c, label='$E(x, y, z=0)$')
    plt.tight_layout()
    plt.show()

def plot_traj(trajectory):
    ylabels = ['x', 'y', 'z']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(trajectory[:, i])
        plt.ylabel(ylabels[i])
    plt.xlabel('Simulation time')
    plt.tight_layout()
    plt.show()

def propagate(n_steps=10000):
    system = mm.System()
    system.addParticle(1)
    force = mm.CustomExternalForce('5*(x-1)^2*(x+1)^2 + y^2 + z^2')
    force.addParticle(0, [])
    system.addForce(force)
    integrator = mm.LangevinIntegrator(500, 1, 0.02)
    context = mm.Context(system, integrator)
    context.setPositions([[0, 0, 0]])
    context.setVelocitiesToTemperature(500)
    x = np.zeros((n_steps, 3))
    for i in range(n_steps):
        x[i] = (context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                ._value)
        integrator.step(1)
    return x
