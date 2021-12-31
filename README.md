# Introduction of the files/project
simulation of motion of multiple double pendulums

This program file allows the user to input a set of double pendulums
containing the initial information of all the angles and speeds, and produces
a simulation of its motion over time. The model of the simulation is based on
using numerical methods on the complicated second order differential
equations. The methods used include the normal euler method and the runge kutta 4th order method.
Additional plots include the angular displacement of pendulums over time, plots of the angles against their respective angular speed
as well as the energy of the system estimated by the method over time.

# Physics of the double pendulum
The double pendulum is a simple system that consists of a second bob attached to an upper bob, which is fixed by a pivot. The motion of a double pendulum
is often studied by scientists as it reflects behaviours of a chaotic system. One distinctive features with chaotic systems is their sensitivity to the initial
conditions your system starts with. What this implies is if you place the two double pendulums at very close but not the same conditions of $\theta$ and $\dot{\theta}$, you observe two different trajectories traced by them over time. Inevitably, this chaotic nature makes it hard to predict the movements of a system over time.
This discovery was first made by meteorologist Edward Lorenz when he studied about a set of equations in an attempt to model atmospheric convections. These later came to be known as the Lorenz system, which established the popular media idea of the 'butterfly effect'.

You will notice that
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
