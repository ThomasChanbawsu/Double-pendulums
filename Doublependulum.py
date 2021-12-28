#!/usr/bin/env python3
'''
This program file allows the user to input a set of double pendulums
containing the initial information of all the angles and speeds, and produces
a simulation of its motion over time. The model of the simulation is based on
using normal euler's method on the complicated second order differential
equations. The state space of the each independent pendulum was also plotted.

Created by Thomas Chan (Physics Mphys student at University of Manchester)
22/12/2021
'''
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


# optimise the initial settings of the double pendulum as shown
MASS1 = 1
MASS2 = 1
LENGTH1 = 1
LENGTH2 = 1
MAX_LENGTH = LENGTH1 + LENGTH2
GRAVITATIONAL_CONSTANT = 9.81
# Fill in your choice of thetas and thetadots in the variable PENDULUM_LIST
# The information for each double pendulum goes [theta1, theta1dot, theta2
# ,theta2dot]. You can also fill in multiple double pendulums
PENDULUM_LIST = [[10, 0 , -10, 0]]
TIMESPAN = 0.005
time = np.arange(0, 15, TIMESPAN)
HISTORY_LEN = 250
# sidenote: this code uses normal Euler's method, which only works for small
# times. If t gets very large, the error accumulated by Euler's method gets
# very large and the motion gets unnecessarily complex.

def convert_to_rad(pendulum_list):
    '''
    converts the initial angles in degrees to radians

    Returns
    -------
    theta1 (radian), theta2 (radian)

    '''
    return np.deg2rad(pendulum_list)

def Lagrange_eq_theta_1(theta1, theta2, theta2_dot, theta2_ddot):
    '''
    returns the second derivative of theta1 of double pendulum based on the
    Lagrange's equation

    Parameters
    ----------
    theta1 : float nd.array
    theta2 : float nd.array
    theta2_dot : float nd.array
    theta2_ddot : float nd.array

    Returns
    -------
    theta1ddot : float nd.array

    '''
    mass_fraction = MASS2 / (MASS1+MASS2)
    theta_difference = theta1 - theta2
    sho_term = - GRAVITATIONAL_CONSTANT * np.sin(theta1)/ LENGTH1
    ddot_term = - theta2_ddot * np.cos(theta_difference)
    dot_term = -(theta2_dot ** 2) * np.sin(theta_difference)
    non_sho_term = mass_fraction * (ddot_term + dot_term) * LENGTH2 / LENGTH1
    theta1ddot = non_sho_term + sho_term
    return theta1ddot

def Lagrange_eq_theta_2(theta2, theta1, theta1_dot, theta1_ddot):
    '''
    returns the second derivative of theta2 of double pendulum based on the
    Lagrange's equation

    Parameters
    ----------
    theta2 : float nd.array
    theta1 : float nd.array
    theta1_dot : float nd.array
    theta1_ddot : float nd.array

    Returns
    -------
    theta2ddot : float nd.array


    '''
    theta_difference = theta1 - theta2
    sho_term = - GRAVITATIONAL_CONSTANT * np.sin(theta2)/ LENGTH2
    ddot_term = - theta1_ddot * np.cos(theta_difference)
    dot_term = (theta1_dot ** 2) * np.sin(theta_difference)
    theta2_ddot = LENGTH1 * (ddot_term + dot_term) / LENGTH2 + sho_term
    return theta2_ddot

def numerical_int(timespan, previous_value, slope_value):
    '''
    carries out the numerical computation for linear acceleration at each spot

    Parameters
    ----------
    timespan : float
    previous_value : float
    slope_value : float

    Returns
    -------
    float

    '''
    return previous_value + slope_value * timespan


def integration_process():
    '''
    Performs the whole integration process of double pendulum motion and
    returns an array of all the theta1 and theta2 values at each single second
    in the given timespan

    Returns
    -------
    motion_data: list of ndarrays
                each index of the list represents a double pendulum from the
                set the user inputted
                The ndarray contains the data of all thetas and
                thetadots at different times t of the motion

    '''
    pendulum_list = convert_to_rad(PENDULUM_LIST)
    motion_data = []
    for pendulum in pendulum_list:
        theta1, theta1dot, theta2, theta2dot = pendulum
        theta1_ddot = 0
        theta2_ddot = 0
        pendulum = np.array(pendulum)

        for _ in time:
            theta1_ddot = Lagrange_eq_theta_1(theta1, theta2, theta2dot,
                                             theta2_ddot)
            next_theta1dot = numerical_int(TIMESPAN, theta1dot, theta1_ddot)
            next_theta1 = numerical_int(TIMESPAN, theta1, theta1dot)

            theta2_ddot = Lagrange_eq_theta_1(theta2, theta1, theta1dot,
                                             theta1_ddot)
            next_theta2dot = numerical_int(TIMESPAN, theta2dot, theta2_ddot)
            next_theta2 = numerical_int(TIMESPAN, theta2, theta2dot)

            instant_array = (next_theta1, next_theta1dot, next_theta2,
                             next_theta2dot)

            pendulum = np.vstack((pendulum, instant_array))

            theta1 = next_theta1
            theta2 = next_theta2
            theta1dot = next_theta1dot
            theta2dot = next_theta2dot
        motion_data.append(pendulum)
    return motion_data

def positions(theta_1_array, theta_2_array):
    '''
    Converts the theta values into corresponding positions of the bob's
    position on the grid for plotting

    Parameters
    ----------
    theta_array : float nd.array

    Returns
    -------
    bob_1_coordinates : float nd.array
                            first column is x position, second is y position
    bob_2_coordinates : float nd.array
                            first column is x position, second is y position

    '''
    x_1 = LENGTH1 * np.sin(theta_1_array)
    y_1 = - LENGTH2 * np.cos(theta_1_array)

    x_2 = x_1 + LENGTH2 * np.sin(theta_2_array)
    y_2 = y_1 - LENGTH2 * np.cos(theta_2_array)

    bob_coordinates = np.column_stack((x_1, y_1, x_2, y_2))

    return bob_coordinates

def hamiltonian(theta1, theta2, theta1_dot, theta2_dot):
    '''
    calculates the hamiltonian (aka also energy) of the double pendulum
    system. It's purpose is to track the errors made by our integration method

    Parameters
    ----------
    theta1 : float 1d array
    theta2 : float 1d array
    theta1_dot : float 1d array
    theta2_dot : float 1d array

    Returns
    -------
    float array

    '''
    T1 = 0.5 * (MASS1 + MASS2) * (LENGTH1 **2) * (theta1_dot ** 2)
    T2 = 0.5 * MASS2 * (LENGTH2 ** 2) * (theta2_dot ** 2)
    diff = theta1 - theta2
    T3 = MASS2 * LENGTH1 * LENGTH2 * theta1_dot * theta2_dot * np.cos(diff)
    T = T1 + T2 + T3
    V1 = -(MASS1 + MASS2) * GRAVITATIONAL_CONSTANT * LENGTH1 * np.cos(theta1)
    V2 = - MASS2 * GRAVITATIONAL_CONSTANT * LENGTH2 * np.cos(theta2)
    V = V1 + V2
    return T+V

def plot_angles(data):
    '''
    plots out the angular displacements made by both bobs over time. It also
    plots out the energy of the double pendulum based on Euler's method
    approximation and the actual initial energy.

    Parameters
    ----------
    data : float ndarray
            must contain theta1, theta2

    Returns
    -------
    None.

    '''
    energy = hamiltonian(data[:,0], data[:,2], data[:,1], data[:,3])
    initial_energy = energy[0]
    figure = plt.figure()
    grid = figure.add_gridspec(2, 1)
    graph = figure.add_subplot(grid[0, 0])
    graph.plot(time, data[:-1,0], '-', label = r'$\theta_1$')
    graph.plot(time, data[:-1,2], '-', label = r'$\theta_2$')
    graph.set_title('Amplitude of both pendulums across time t')
    graph.set_xlabel('Angular Displacement (in radians)')
    graph.set_ylabel('Time (seconds)')
    graph.grid()
    graph.legend()
    graph2 = figure.add_subplot(grid[1, 0])
    graph2.plot(time, energy[:-1], '-', label = "energy (Euler's method")
    graph2.axhline(y = initial_energy, linestyle = '--', color = 'red',
                   label = 'actual energy')
    graph2.set_title('Energy of double pendulum over time')
    graph2.set_xlabel('time (seconds)')
    graph2.set_ylabel('energy (J)')
    graph2.grid()
    graph2.legend()
    plt.tight_layout()
    plt.show()

data = integration_process()
coordinate_list = []
for pendulum_data in data:
    coordinates = positions(pendulum_data[:,0], pendulum_data[:,2])
    plot_angles(pendulum_data)
    coordinate_list.append(coordinates)
coordinate_list = np.array(coordinate_list)



fig = plt.figure(figsize = (10,10))
grid = fig.add_gridspec(2, 2)
# main axes for the animation of the double pendulums
ax = fig.add_subplot(grid[0, 0:2], autoscale_on=False,
                xlim = (-MAX_LENGTH-1, MAX_LENGTH+1),
                ylim = (-MAX_LENGTH-1, MAX_LENGTH+1))
ax.set_aspect('equal')
ax.grid()
ax.set_title('Swinging motion of the double pendulum over time', fontsize = 16)
lines = [ax.plot([],[], 'o-', lw=2, ms = 10,
                  mfc = 'black')[0] for _ in range(len(data))]
traces = [ax.plot([], [], '-', lw = 1, color = 'r', alpha = 0.5)[0]
          for _ in range(len(data))]
TIME_TEMPLATE = 'time = %.1fs'
time_text = ax.text(0.1, 0.9, '', transform = ax.transAxes)
histories = [(deque(maxlen=HISTORY_LEN), deque(maxlen=HISTORY_LEN))
             for _ in range(len(data))]

# axes for state space of the pendulum
ax2 = fig.add_subplot(grid[1,0], autoscale_on=False,
                          xlim = (-5, 5),
                          ylim = (-5, 5))
ax2.set_title('State space of the pendulum', fontsize = 16)
ax2.grid()
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel(r'speed $\dot{\theta}$')
state_lines = [ax2.plot([], [], 'o-', lw=1, ms = 1,
                    label = f'pendulum {no}')[0] for no in range(len(data))]
ax2.legend()

ax3 = fig.add_subplot(grid[1,1], autoscale_on=False,
                          xlim = (-5, 5),
                          ylim = (-5, 5))
ax3.set_title(r'State space of the second pendulum', fontsize = 16)
ax3.grid()
ax3.set_xlabel(r'$\theta_2$')
ax3.set_ylabel(r'spped $\dot{\theta_2}$')
state_lines2 = [ax3.plot([], [], 'o-', lw=1, ms = 1,
                    label = f'pendulum {no}')[0] for no in range(len(data))]
ax3.legend()


def animate(i):
    '''
    A local scope function. This is used in the Funcanimation where it sets
    the data for each iterable plot line and trace line during the animation
    of the double pendulum. Note that this function return a lists of
    iterables.

    Parameters
    ----------
    i : iterable

    Returns
    -------
    iterable : iterable list

    '''
    for j, line in enumerate(lines):
        trace = traces[j]
        state_line = state_lines[j]
        state_line2 = state_lines2[j]
        history_x, history_y = histories[j]
        first_bob_x, first_bob_y = coordinate_list[j][i, 0:2]
        second_bob_x, second_bob_y = coordinate_list[j][i, 2:4]
        thisx = [0, first_bob_x, second_bob_x]
        thisy = [0, first_bob_y, second_bob_y]
        state_angle = data[j][0:i+1, 0]
        state_angledot = data[j][0:i+1,1]
        state_angle2 = data[j][0:i+1, 2]
        state_angledot2 = data[j][0:i+1, 3]

        if i == 0:
            history_x.clear()
            history_y.clear()

        history_x.appendleft(thisx[2])
        history_y.appendleft(thisy[2])
        histories[j] = (history_x, history_y)
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        state_line.set_data(state_angle, state_angledot)
        state_line2.set_data(state_angle2, state_angledot2)

    iterable = lines + traces + state_lines + state_lines2
    time_text.set_text(TIME_TEMPLATE % (i*TIMESPAN))
    iterable.append(time_text)
    return iterable


anim = animation.FuncAnimation(fig, animate, frames = len(time),
                 interval=TIMESPAN, blit= True)

plt.show()

    
