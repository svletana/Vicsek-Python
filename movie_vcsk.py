'''This script performs a simulation of the
 Vicsek model and creates a movie from the data'''

import vicsek as vi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def vicsek_movie(L, pos_x, pos_y, steps):
    '''Create mp4 movie from Vicsek simulation data.
    L: box side length
    pos_x, pos_y, pos_z: array of xyz positions over time
    steps: number of simulation steps performed'''
    # Setting the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sct, = ax.plot([], [], "o", markersize=2)

    # callable function for FuncAnimation
    def update(i, pos_x, pos_y):
        x = pos_x[i]
        y = pos_y[i]
        sct.set_data(x, y)

    # Setting the axes properties
    lim = L * .5
    lims = [-lim, lim]
    ax.set_xlim(lims)
    ax.set_xlabel('X')
    ax.set_ylim(lims)
    ax.set_ylabel('Y')
    ax.set_title('Vicsek Swarm')

    # Perform animation and save
    args = (pos_x, pos_y)
    line_ani = animation.FuncAnimation(fig, update, frames=steps,
                                       fargs=args)
    line_ani.save("vicsek_ani.mp4", writer='imagemagick', fps=30)


# variables
N, L, ns, steps = 300, 25, .1, 500
# things I'll keep fixed
dt, R, v = 1, 1, .3
data_name, save_bool = "data", True
#  initialize simulation
sim = vi.Vicsek2D(dt=dt, N=N, R=R, L=L, v=v, ns=ns, steps=steps,
                  data_name=data_name, save_bool=save_bool)
sim.do_simulation()

# read simulation data
df = pd.read_pickle("data")
# get positions only
poss = {}
for i, frm in enumerate(df):
    poss[i] = frm['pos']
# isolate each coordinate
# each array contains a number "steps" of subarrays,
# where each one corresponds to the xy coordinates
# of the N particles for said frame
pos_x = []
pos_y = []
for p in poss:
    pos_x.append([poss[p][i][0] for i in range(N)])
    pos_y.append([poss[p][i][1] for i in range(N)])

# animate!
vicsek_movie(L, pos_x, pos_y, steps)
