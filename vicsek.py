import time
import numpy as np
from scipy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Vicsek3D:
    '''Implementation of 3D Vicsek model'''
    def __init__(self, dt, N, R, L, v, ns, steps, data_name):
        self.dt = dt  # frame
        self.N = N  # number of particles
        self.R = R  # radius of interaction
        self.L = L  # box side length
        self.v = v  # velocity absolute value
        self.ns = ns  # noise
        self.steps = steps  # number of simulation steps
        self.state = []  # array for system state
        self.data_name = data_name  # filename of simulation data

        # write simulation metadata
        self.metadata()

        # initialize positions
        self.pos = (2 * np.random.random((self.N, 3)) - 1) * self.L * 0.5

        # initialize velocities
        self.vel = (2 * np.random.random((self.N, 3)) - 1) * self.v * 0.5

        # initialize orientations (3D space --> two angles)
        # alpha between 0 and pi, phi between 0 and 2pi
        alpha = np.random.random(self.N) * np.pi
        phi = np.random.random(self.N) * 2 * np.pi
        self.theta = np.array([[i, j] for i, j in zip(alpha, phi)])

        # initialize noise
        self.randomize_noise()

        # write initial state
        self.write_state()

    def metadata(self):
        '''Write metadata file'''
        with open("metadata_vicsek.txt", 'w') as f:
            f.write(time.asctime())
            f.write("\nVariable - var. name in code - value\n\n")
            f.write("Number of particles (N): {}\n".format(self.N))
            f.write("Box length (L): {}\n".format(self.L))
            f.write("Time step (dt): {}\n".format(self.dt))
            f.write("Radius of interaction (R): {}\n".format(self.R))
            f.write("Velocity absolute value (v): {}\n".format(self.v))
            f.write("Noise Value (ns): {}\n".format(self.ns))
            f.write("Steps (steps): {}\n".format(self.steps))
            f.write("=" * 40)

    def write_state(self):
        '''Record system state'''
        curr_state = {"pos": self.pos,
                      "vel": self.vel,
                      "angle": self.theta,
                      "noise": self.noise}
        self.state.append(curr_state)

    def save_data(self):
        '''Save simulation data to pickle file'''
        pd.to_pickle(self.state, self.data_name)

    def randomize_noise(self):
        '''Randomize noise'''
        self.noise = (2 * np.random.random((self.N, 2)) - 1) * self.ns

    def step_angle_p(self, p):
        '''Angle time step for particle p'''
        # get indices of particles within interaction range
        interacting = []
        for i, j in enumerate(self.pos):
            if (norm(self.pos[p]-j) < self.R and i != p):
                interacting.append(i)
        # compute average orientation from interacting particles
        if interacting == []:
            return self.theta[p]
        else:
            avg_theta = np.mean([self.theta[i] for i in interacting], axis=0)
            new_theta = avg_theta + self.noise[p]
            return new_theta

    def step_vel_p(self, p, new_theta):
        '''Velocity time step for particle p'''
        alpha = new_theta[0]
        phi = new_theta[1]
        new_vel = np.array([np.cos(alpha) * np.sin(phi),
                            np.sin(alpha) * np.sin(phi),
                            np.cos(phi)])
        new_vel *= self.v
        return new_vel

    def step_pos_p(self, p, new_vel):
        '''Position time step for particle p'''
        new_pos = self.pos[p] + new_vel * self.dt
        limit = self.L * .5
        # periodic boundary conditions x
        if new_pos[0] > limit:
            new_pos[0] -= 2 * limit
        elif new_pos[0] < -limit:
            new_pos[0] += 2 * limit
        # periodic boundary conditions y
        if new_pos[1] > limit:
            new_pos[1] -= 2 * limit
        elif new_pos[1] < -limit:
            new_pos[1] += 2 * limit
        # periodic boundary conditions z
        if new_pos[2] > limit:
            new_pos[2] -= 2 * limit
        elif new_pos[2] < -limit:
            new_pos[2] += 2 * limit
        return new_pos

    def full_step(self):
        '''Compute a full time step'''
        # make room for new state
        new_angles = np.zeros((self.N, 2))
        new_positions = np.zeros((self.N, 3))
        new_velocities = np.zeros((self.N, 3))

        # compute new state for each particle
        for p in range(self.N):
            new_angles[p] = self.step_angle_p(p)
            new_velocities[p] = self.step_vel_p(p, new_angles[p])
            new_positions[p] = self.step_pos_p(p, new_velocities[p])

        self.theta = new_angles
        self.vel = new_velocities
        self.pos = new_positions
        self.randomize_noise()

    def do_simulation(self):
        '''Perform simulation and save data'''
        for frame in range(1, self.steps+1):
            self.full_step()
            self.write_state()
        self.save_data()

    def show_state(self):
        '''Draw current state'''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = [r[0] for r in self.pos]
        y = [r[1] for r in self.pos]
        z = [r[2] for r in self.pos]
        ax.scatter(x, y, z)
        plt.show()
