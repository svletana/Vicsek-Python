'''This script performs a simulation of the
 Vicsek model and creates a movie from the data'''

import vicsek as vi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def vicsek_movie(L, pos_x, pos_y, pos_z, steps):
    '''Create mp4 movie from Vicsek simulation data.
    L: box side length
    pos_x, pos_y, pos_z: array of xyz positions over time
    steps: number of simulation steps performed'''
    # Setting the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sct, = ax.plot([], [], [], "o", markersize=5)

    # callable function for FuncAnimation
    def update(i, pos_x, pos_y, pos_z):
        x = pos_x[i]
        y = pos_y[i]
        z = pos_z[i]
        sct.set_data(x, y)
        sct.set_3d_properties(z)

    # Setting the axes properties
    lim = L * .5
    lims = [-lim, lim]
    ax.set_xlim3d(lims)
    ax.set_xlabel('X')
    ax.set_ylim3d(lims)
    ax.set_ylabel('Y')
    ax.set_zlim3d(lims)
    ax.set_zlabel('Z')
    ax.set_title('Vicsek Swarm')

    # Perform animation and save
    args = (pos_x, pos_y, pos_z)
    line_ani = animation.FuncAnimation(fig, update, frames=steps,
                                       fargs=args)
    line_ani.save("vicsek_ani.mp4", writer='imagemagick', fps=30)


#  initialize simulation
dt, N, R, L, v, ns, steps, data_name = 1, 30, 1, 2, .3, 2, 200, "data"
sim = vi.Vicsek3D(dt=dt, N=N, R=R, L=L, v=v, ns=ns,
                  steps=steps, data_name=data_name)
sim.do_simulation()

# read simulation data
df = pd.read_pickle("data")
# get positions only
poss = {}
for i, frm in enumerate(df):
    poss[i] = frm['pos']
# isolate each coordinate
# each array contains a number "steps" of subarrays,
# where each one corresponds to the x, y or z coordinates
# of the N particles for said frame
pos_x = []
pos_y = []
pos_z = []
for p in poss:
    pos_x.append([poss[p][i][0] for i in range(N)])
    pos_y.append([poss[p][i][1] for i in range(N)])
    pos_z.append([poss[p][i][2] for i in range(N)])

# animate!
vicsek_movie(L, pos_x, pos_y, pos_z, steps)
