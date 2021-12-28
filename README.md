# Double-pendulums
simulation of motion of multiple double pendulums

This program file allows the user to input a set of double pendulums
containing the initial information of all the angles and speeds, and produces
a simulation of its motion over time. The model of the simulation is based on
using normal euler's method on the complicated second order differential
equations. The state space of the each independent pendulum was also plotted. In addition,
a function that plots the angular displacement and energy was also added. You will notice that
the amplitude or energy for the double pendulum appears to blow up over a period of time. This is
reasonable due to the growing truncating error as a result of using normal Euler's method in
simulating its motion.

I am also currently looking into better approximation methods for the double pendulum motion. If anyone
knows a bit about chaos theory or bifurcation, I'm delighted if you can show me how to incorporate it
in my program (e.g creating a bifurcation diagram for the double pendulum)


Created by Thomas Chan (Physics Mphys student at University of Manchester)
22/12/2021
