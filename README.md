# Double-pendulums
simulation of motion of multiple double pendulums

This program file allows the user to input a set of double pendulums
containing the initial information of all the angles and speeds, and produces
a simulation of its motion over time. The model of the simulation is based on
using normal euler's method on the complicated second order differential
equations. The state space of the each independent pendulum was also plotted. In addition,
a function that plots the angular displacement and energy was also added. You will notice that
the amplitude or energy for the double pendulum blows up over a period of time. This is
reasonable due to the growing truncating error as a result of using normal Euler's method in
simulating its motion. In the end, this simulation is based on an approximation model. One way to
get more accurate results is if we further decrease the size of the time step, though this generally 
means a tradeoff for longer running time for the computer. Another way is utilising higher order
approximation methods, a helpful one being the famous runge-kutta 4 method. These type of methods produce
an error that is around the order of 4-5, making it significantly more accurate than the normal euler method.

I am  currently looking into better approximation methods for the double pendulum motion, especially the 
RK4 method. If anyoneknows a bit about chaos theory or bifurcation, I'm delighted if you can show me how to incorporate it
in my program (e.g creating a bifurcation diagram for the double pendulum, though this may be pretty complicated and advanced
for me at the moment.)


Created by Thomas Chan (2nd year Physics Mphys student at University of Manchester)
22/12/2021
