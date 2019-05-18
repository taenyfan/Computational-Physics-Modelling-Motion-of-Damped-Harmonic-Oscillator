#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:01:39 2018

@author: aiyuan
COMPUTATIONAL PHYSICS PROJECT 2
MODELLING SIMPLE HARMONIC MOTION
PART II

PROGRAM THAT CALCULATES THE SOLUTIONS OF DAMPED HARMONICE MOTION 
WITH AN INSTANTANEOUS FORCE APPLIED AT DIFFERENT PHASES OF THE MOTION
USING THE VERLET METHOD AND STEP SIZE 0.001
"""

import math
import numpy as np
import matplotlib.pyplot as plt

#==============
#  Functions
#==============
def get_verlet(x_i,v_i,t,k,m,b,h,n,F_inst,t_inst, F_duration):
    """
    Calculates x,v,a,E at every step using Verlet Method
    Returns values of x,v,a,E calculated 
    """           
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    E = np.zeros(n)
    F = np.zeros(n)
    F[int(t_inst/h) : int(t_inst/h) + int(F_duration/h)] = F_inst # assign instantaneous force 
    for i in range(int(n)):
        if i == 0:
            x[i] = x_i
            v[i] = v_i
            a[i] = - (b*v[i]/m) - (k*x[i]/m) + F[i]/m
        elif i == 1:    # Improved Euler Method for First Step
            v[i] = v[i-1] + h*a[i-1]
            x[i] = x[i-1] + h*v[i-1] + 0.5*h*h*a[i-1]
            a[i] = - (b*v[i]/m) - (k*x[i]/m) + F[i]/m
        else:           # Verlet Method for Second Step Onwards
            x[i] = 2*(2*m-k*h*h)/(2*m+b*h)*x[i-1] + (b*h-2*m)/(2*m+b*h)*x[i-2] + 2*F[i-1]*h*h/(2*m+b*h)
            if i >= 3:  # can only calculate v[i] when both x[i+1] and x[i-1] is known
                v[i-1] = (x[i] - x[i-2])/2/h
    for i in range(n):
        E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
    return (x,v,a,E)

def get_t(h,n):
    """
    Calculate the time values from steps and step size
    """
    t = np.zeros(n)
    for i in range(n):
        if i==0:
            t[i] = 0
        else:
            t[i] = t[i-1] + h
    return t

#=======================
#   INITIAL CONDITIONS
#========================
k = 0.89
m = 2.16
b = 0.3
x_i = 0
v_i = -1
h = 0.001
n = 80000
F_inst = 0      # Force applied
t_inst = 0   # Time instant when force applied
F_duration = 0     # Duration force is applied for

#==========================
#   CALCULATE TIME VALUES
#==========================
t = get_t(h,n)

#=========================
#  CALCULATE SOLUTIONS
#========================
x_unforced = get_verlet(x_i,v_i,t,k,m,b,h,n,F_inst,t_inst,F_duration)[0]

F_inst = -15
t_inst = 21
F_duration = 0.05

x_forced = get_verlet(x_i,v_i,t,k,m,b,h,n,F_inst,t_inst,F_duration)[0]

#======================
#  PLOT SOLUTIONS
#======================
plt.plot(t, x_forced, '-.', label = 'Forced')
plt.plot(t, x_unforced, '--', label = 'Unforced')
plt.grid()
plt.legend()
plt.title('$b=%.1f, h=%.3f, n=%i$' % (b,h,n))
plt.xlabel('$Time$ $/s$')
plt.ylabel('$Displacement$ $/m$')
