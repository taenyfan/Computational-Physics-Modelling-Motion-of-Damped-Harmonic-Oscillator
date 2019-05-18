#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:26:26 2018
@author: jiachen

COMPUTATIONAL PHYSICS PROJECT 2
MODELLING SIMPLE HARMONIC MOTION
PART I

INTERACTIVE PROGRAM THAT:
    1.CALCULATES SOLUTIONS OF DAMPED UNFORCED HARMONIC MOTION GIVEN INITIAL CONDITIONS USING:
        ANALYTICAL METHOD
        EULER METHOD
        IMPROVED EULER METHOD
        VERLET METHOD
        EULER CROMER METHOD
    2.SHOWS ACCURACY OF EACH NUMERICAL METHOD BY COMPARING ENERGY SOLUTIONS TO ANALYTICAL METHOD 
    3.SHOWS EFFECT OF STEP SIZE ON VERLET METHOD
    4.CALCULATES SOLUTIONS OF LIGHT,CRITICAL AND HEAVY DAMPING USING VERLET METHOD
    
"""

import numpy as np
import math
import matplotlib.pyplot as plt

#=======================
#      FUNCTIONS
#=======================
def get_user_input(s):
    """
    string -> float
    Get a valid float value for 's' from user input 
    """
    userInput = input('Please input a value for %s in SI units' % s)
    while True:
        try:
            float(userInput)
            break
        except:
            userInput = input('Input for %s is not a number. Please re enter a valid value. ' % s)
    return float(userInput)

def get_degree(m,k,b):
    """
    float, float, float -> int
    Determine degree of damping from b,m,k values
    Returns int that represents degree of damping
    """
    degree = 0
    if b ==0: degree = 0
    elif b < 2*math.sqrt(m*k): degree = 1
    elif b == 2*math.sqrt(m*k): degree = 2
    else: degree = 3
    return degree
        
def get_ana(degree,x_i,v_i,t,k,m,b,h,n):
    """
    float,float,np,float,float,float,float,int -> (np,np,np)
    Calculates x,v,E at every step using Analytical Solution
    Returns values x,v,E calculated
    """
    x = np.zeros(n)
    v = np.zeros(n)
    E = np.zeros(n)
    
    # if free oscillation or light damping
    if degree == 0 or degree==1:
        # calculate amplitude term
        amp = math.sqrt(x_i*x_i + (v_i*v_i*m/k)) 
        # calculate phase angle term
        if x_i/amp >= 0:
            # fix phi to be within 0 and 2pi
            phi = [math.asin(x_i/amp), math.pi-math.asin(x_i/amp)]
            # compare possible phi values with initial velocity
            if abs(v_i - amp*(math.sqrt(k/m))*math.cos(phi[0])) < 1e-5:
                phi = float(phi[0])
            elif abs(v_i - amp*(math.sqrt(k/m))*math.cos(phi[1])) < 1e-5:
                phi = float(phi[1])
        else:
            # possible phi values between 0 and 2pi
            phi = [math.pi-math.asin(x_i/amp), 2*math.pi+math.asin(x_i/amp)]
            # compare possible phi values with initial velocity
            if abs(v_i - amp*(math.sqrt(k/m))*math.cos(phi[0])) < 1e-5:
                phi = float(phi[0])
            elif abs(v_i - amp*(math.sqrt(k/m))*math.cos(phi[1])) < 1e-5:
                phi = float(phi[1])
        if degree == 0:
            # calculate angular frequency 
            w = math.sqrt(k/m)
            # caclculate x,v,E 
            for i in range(n):
                x[i] = amp * math.sin( w*t[i] + phi )
                v[i] = amp * w * math.cos( w*t[i] + phi )
                E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
        elif degree == 1:
            # calculate gamma
            g = b/m
            # calculate angular frequency
            w = math.sqrt(k/m - g*g/4)
            # calculate x,v,E 
            for i in range(n):
                x[i] = amp * math.exp( -g/2*t[i] ) * math.sin( w*t[i] + phi )
                v[i] = -g/2 * amp * math.exp(-g/2*t[i]) * math.sin(w*t[i] + phi) + amp * w * math.exp(-g/2*t[i]) * math.cos(w*t[i] + phi)
                E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
#                
    # critical damping      
    elif degree == 2:
        # calculate gamma
        g = b/m
        # calculate A and B term
        A = x_i
        B = v_i + g/2*x_i
        
        for i in range(n):
            x[i] = A * math.exp(-g/2*t[i]) + B * t[i] * math.exp(-g/2*t[i])
            v[i] = -g/2 * A * math.exp(-g/2*t[i]) + B * math.exp(-g/2*t[i]) - g/2 * B * t[i] * math.exp(-g/2*t[i])
            E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
    else:
        # calculate gamma and alpha
        g = b/m
        a = math.sqrt(g*g/4 - k/m)
        # calculate A and B term
        B = (1/2/a) * ( (-g/2 + a)*x_i - v_i )
        A = x_i - B
        # calculate x,v,E
        for i in range(n):
            x[i] = A*math.exp( (-g/2 + a)*t[i] ) + B*math.exp( (-g/2-a)*t[i] )
            v[i] = (-g/2+a)*A*math.exp( (-g/2 + a)*t[i] ) - (g/2+a)*B*math.exp( (-g/2-a)*t[i] )
            E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
    return (x,v,E)
        
def get_euler(x_i,v_i,t,k,m,b,h,n):
    """
    Calculates x, v, a, E at every step using Euler Method
    Returns values of x, v, a, E calculated 
    """
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    E = np.zeros(n)
    for i in range(n):
        if i == 0:
            x[i] = x_i
            v[i] = v_i
            a[i] = - (b*v[i]/m) - (k*x[i]/m)
            E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
        else:
            v[i] = v[i-1] + h*a[i-1]
            x[i] = x[i-1] + h*v[i-1]
            a[i] = - (b*v[i]/m) - (k*x[i]/m)
            E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
    return (x,v,a,E)

def get_euler_improved(x_i,v_i,t,k,m,b,h,n):
    """
    Calculates x,v,a,E at every step using Improved Euler Method
    Returns values of x,v,a,E calculated 
    """ 
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    E = np.zeros(n)
    for i in range(n):
        if i == 0:
            x[i] = x_i
            v[i] = v_i
            a[i] = - (b*v[i]/m) - (k*x[i]/m)
            E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
        else:
            v[i] = v[i-1] + h*a[i-1]
            x[i] = x[i-1] + h*v[i-1] + 0.5*h*h*a[i-1]
            a[i] = - (b*v[i]/m) - (k*x[i]/m)
            E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
    return (x,v,a,E)
   
def get_verlet(x_i,v_i,t,k,m,b,h,n):
    """
    Calculates x,v,a,E at every step using Verlet Method
    Returns values of x,v,a,E calculated 
    """           
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    E = np.zeros(n)
    for i in range(int(n)):
        if i == 0:
            x[i] = x_i
            v[i] = v_i
            a[i] = - (b*v[i]/m) - (k*x[i]/m)
        elif i == 1:    # Improved Euler Method for First Step
            v[i] = v[i-1] + h*a[i-1]
            x[i] = x[i-1] + h*v[i-1] + 0.5*h*h*a[i-1]
            a[i] = - (b*v[i]/m) - (k*x[i]/m)
        else:           # Verlet Method for Second Step Onwards
            x[i] = 2*(2*m-k*h*h)/(2*m+b*h)*x[i-1] + (b*h-2*m)/(2*m+b*h)*x[i-2]
            if i >= 3:  # can only calculate v[i] when both x[i+1] and x[i-1] is known
                v[i-1] = (x[i] - x[i-2])/2/h
            if i == n-1: # need to calculate extra point of x if want to find velocity of last point
                v[i] = ((2*(2*m-k*h*h)/(2*m+b*h)*x[i] + (b*h-2*m)/(2*m+b*h)*x[i-1])-x[i-1])/2/h
    for i in range(n):
        E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
    return (x,v,a,E)
            
def get_euler_cromer(x_i,v_i,t,k,m,b,h,n):
    """
    Calculates x,v,a,E at every step using Euler Cromer Method
    Returns values of x,v,a,E calculated 
    """
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    E = np.zeros(n)
    for i in range(n):
        if i == 0:
            x[i] = x_i
            v[i] = v_i
            a[i] = - (b*v[i]/m) - (k*x[i]/m)
            E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
        else:
            v[i] = v[i-1] + h*a[i-1]
            x[i] = x[i-1] + h*v[i]
            a[i] = - (b*v[i]/m) - (k*x[i]/m)
            E[i] = 0.5*m*v[i]*v[i] + 0.5*k*x[i]*x[i]
    return (x,v,a,E)        

def E_sum(E):
    """
    np array -> np array
    Calcualte E_sum[i] = E[0]+E[1]+E[2]+...+E[i]
    Returns all values of E_sum[i] calculated
    """
    E_sum = np.zeros(len(E))
    for i in range(len(E)):
        if i == 0:
            E_sum[i] = E[i]
        else:
            E_sum[i] = E[i] + E_sum[i-1]
    return E_sum

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
#    INTRODUCTION 
#=======================
text = 'This program calculates the solutions of damped harmonic motion of an oscillator using  analytical and other numerical methods, if given the initial conditions of displacement, velocity and values for spring constant, mass of object, damping term. Also needed are the values of step size and number of steps taken. A step size of 0.001 is recommended.'
print(text)

#========================
#   INITIAL CONDITIONS
#========================
k = get_user_input('Spring Constant, k')
m = get_user_input('Mass of Object, m')
b = get_user_input('Damping Term, b')
x_i = get_user_input('Initial Displacement, x')
v_i = get_user_input('Initial Velocity, v')
h = get_user_input('Steph Size, h')
n = int(get_user_input('Number of Steps, n'))

#====================================
#  DETERMINE DEGREE OF DAMPING
# i.e. free, light, critical or heavy
#====================================
degree = get_degree(m,k,b)      # 0-free 1-light 2-critical 3-heavy

#==========================
#   CALCULATE TIME VALUES
#==========================
t = get_t(h,n)

#=========================
#  CALCULATE SOLUTIONS
#========================
# get x values by analytical solution
(x_ana, v_ana, E_ana) = get_ana(degree,x_i,v_i,t,k,m,b,h,n)
# write to file
data_ana = np.array([x_ana, v_ana, E_ana]).T  # transpose into columns
with open('Analytical Solution Data', 'w+') as file_object:
    np.savetxt(file_object, data_ana, fmt=['%f','%f','%f'])

# get x values by eulor method
(x_euler, v_euler, a_euler, E_euler) = get_euler(x_i,v_i,t,k,m,b,h,n)
# write to file
data_euler = np.array([x_euler, v_euler, a_euler, E_euler]).T
with open('Euler Solution Data', 'w+') as file_object:
    np.savetxt(file_object, data_euler, fmt=['%f','%f','%f','%f'])

# get x values by improved eulor method
(x_euler_improved, v_euler_improved, a_euler_improved, E_euler_improved) = get_euler_improved(x_i,v_i,t,k,m,b,h,n)
# write to file
data_euler_improved = np.array([x_euler_improved, v_euler_improved, a_euler_improved, E_euler_improved]).T
with open('Improved Euler Solution Data', 'w+') as file_object:
    np.savetxt(file_object, data_euler_improved, fmt=['%f','%f','%f','%f'])

# get x values by verlet method
(x_verlet, v_verlet, a_verlet, E_verlet) = get_verlet(x_i,v_i,t,k,m,b,h,n)
# write to file
data_verlet = np.array([x_verlet, v_verlet, a_verlet, E_verlet]).T
with open('Verlet Solution Data', 'w+') as file_object:
    np.savetxt(file_object, data_verlet, fmt=['%f','%f','%f','%f'])

# get x values by eulor cromer method
(x_euler_cromer, v_euler_cromer, a_euler_cromer, E_euler_cromer) = get_euler_cromer(x_i,v_i,t,k,m,b,h,n)
# write to file
data_euler_cromer = np.array([x_euler_cromer, v_euler_cromer, a_euler_cromer, E_euler_cromer]).T
with open('Euler Cromer Solution Data', 'w+') as file_object:
    np.savetxt(file_object, data_euler_cromer, fmt=['%f','%f','%f','%f'])

#=====================================
#        DISPLACEMENT 
#=====================================
# displacement against time for every method used 
plt.plot(t,x_ana,label='Analytic Method')
plt.plot(t,x_euler, label='Euler Method')
plt.plot(t,x_euler_improved, label='Improved Euler Method')
plt.plot(t,x_verlet, label='Verlet Method')
plt.plot(t,x_euler_cromer, label='Euler Cromer Method')
plt.grid()
plt.legend( fontsize=10)
plt.title('$b=%.1f, h=%.1f, n=%i$' % (b,h,n))
plt.ylabel('$Displacement$ $/m$')
plt.xlabel('$Time$ $/s$')
plt.show()

#====================================
#       PHASE SPACE 
#====================================
# read in data from files 
data_ana = np.loadtxt('Analytical Solution Data')
(x_ana, v_ana, E_ana) = (data_ana[:,0], data_ana[:,1], data_ana[:,2])

data_euler = np.loadtxt('Euler Solution Data')
(x_euler, v_euler, E_euler) = (data_euler[:,0], data_euler[:,1], data_euler[:,3])

data_euler_improved = np.loadtxt('Improved Euler Solution Data')
(x_euler_improved, v_euler_improved, E_euler_improved) = (data_euler_improved[:,0], data_euler_improved[:,1], data_euler_improved[:,3])

data_verlet = np.loadtxt('Verlet Solution Data')
(x_verlet, v_verlet, E_verlet) = (data_verlet[:,0], data_verlet[:,1], data_verlet[:,3])

data_euler_cromer = np.loadtxt('Euler Cromer Solution Data')
(x_euler_cromer, v_euler_cromer, E_euler_cromer) = (data_euler_cromer[:,0], data_euler_cromer[:,1], data_euler_cromer[:,3])

# plot phase space
plt.plot(x_ana, v_ana, label='Analytic Method')
plt.plot(x_euler, v_euler, label='Euler Method')
plt.plot(x_euler_improved, v_euler_improved, label='Improved Euler Method')
plt.plot(x_verlet, v_verlet, label='Verlet Method')
plt.plot(x_euler_cromer, v_euler_cromer, label='Euler Cromer Method')
plt.legend(loc='lower right', fontsize=8)
plt.title('$b=%.1f, h=%.1f, n=%i$' % (b,h,n))
plt.xlabel('$Displacement$ $/m$')
plt.ylabel('$Velocity$ $/ms^-1$')
plt.grid()
plt.show()

#===========================
#          ENERGY
#===========================
# plot energy against time for every method
plt.plot(t, E_ana, label='Analytic Method')
plt.plot(t, E_euler, label='Euler Method')
plt.plot(t, E_euler_improved, label='Improved Euler Method')
plt.plot(t, E_verlet, label='Verlet Method')
plt.plot(t, E_euler_cromer, label='Euler Cromer Method')
plt.title('$b=%.1f, h=%.1f, n=%i$' % (b,h,n))
plt.xlabel('$Time$ $/s$')
plt.ylabel('$Energy$ $/J$')
plt.xlim(0)
plt.grid()
plt.legend()
plt.show()


#=================================
#   SUMMED ENERGY DIFFERENCE
#=================================
# calculate the total summ of energy from every previous point in time
E_ana_sum = E_sum(E_ana)
E_euler_sum = E_sum(E_euler)
E_euler_improved_sum = E_sum(E_euler_improved)
E_verlet_sum = E_sum(E_verlet)
E_euler_cromer_sum = E_sum(E_euler_cromer)

# plot summed energy difference graphs
# ln(summed energy of numerical method - summed energy of analytic solution) against time
plt.plot(t, np.log(abs(E_euler_sum - E_ana_sum)),  label='Euler Method')
plt.plot(t, np.log(abs(E_euler_improved_sum - E_ana_sum)), label='Improved Euler Mehod')
plt.plot(t, np.log(abs(E_verlet_sum - E_ana_sum)), label='Verlet Method')
plt.plot(t, np.log(abs(E_euler_cromer_sum - E_ana_sum)), label='Euler Cromer Method') 
plt.grid()
plt.legend( fontsize=10)
plt.title('$b=%.1f, h=%.1f, n=%i$' % (b,h,n))
plt.ylabel('$ln(|\Delta$ $Energy|$) $/J$')
plt.xlabel('$Time$ $/s$')
plt.show()

""" OPTIONAL CODE TO SEE NUMERICAL VALUES OF SUMMED ENERGY DIFFERENCE AFTER N STEPS
# print numerical value E_method_sum - E_ana_sum after n steps
print('After %i steps, b = %f, h = %f,' % (n,b,h))
print('abs(E_euler_sum - E_ana_sum) =', (abs(E_euler_sum - E_ana_sum))[-1])
print('abs(E_euler_improved_sum - E_ana_sum) =', (abs(E_euler_improved_sum - E_ana_sum))[-1])
print('abs(E_verlet_sum - E_ana_sum) =', (abs(E_verlet_sum - E_ana_sum))[-1])
print('abs(E_euler_cromer_sum - E_ana_sum) =', (abs(E_euler_cromer_sum - E_ana_sum))[-1])
"""

# Note: This Part takes Some Time to Calculate. Can be Left out
#==============================================================
#       DETERMINING ACCURACY OF STEP SIZES
#==============================================================
# determine how varying step size affects accuracy of verlet method 
# Note: Use Zero Damping for This Part
# step sizes used : 1, 0.1,0.01,0.001,0.0001,0.00001,0.000001'
# accuracy : difference of total sum of E of numerical solution from that that of analytical solution after 10s

# store energy sum for different h values after 10s
E_verlet_sum_h = []

# store energy sum for different h values after 10s
E_ana_sum_h = [] 

# calculate energy sum values for verlet method
for i in [1, 0.1,0.01,0.001,0.0001,0.00001,0.000001]:
    h = i
    n = int(10/i)
    E_verlet = get_verlet(x_i,v_i,t,k,m,b,h,n)[3]
    E_verlet_sum = E_sum(E_verlet)
    E_verlet_sum_h.append(E_verlet_sum[-1])

# calculate energy sum values for analytical method
for i in [1, 0.1,0.01,0.001,0.0001,0.00001,0.000001]:
    E_ana_sum_h.append( E_ana[0] * (10/i) ) 

print('h : 1, 0.1,0.01,0.001,0.0001,0.00001,0.000001')   
print('Energy Difference after 10s :', np.array(E_verlet_sum_h) - np.array(E_ana_sum_h))


#=====================================
#    SOLUTION GRAPHS FOR DIFFFERENT
#        DAMPING TERMS
#====================================
# Verlet Method, Step Size = 0.001, Steps = 20000
h = 0.001
n = 20000
t = get_t(h,n)
b_crit = 2 * math.sqrt(k * m)

# Calculate Solutions
# Lightly Damped Oscillation, b = 0.5*b_crit
b = 0.5 * b_crit
(x_light, v_light, a_light, E_light) = get_verlet(x_i,v_i,t,k,m,b,h,n)
# Critically Damped Oscillation, b = b_crit
b = b_crit
(x_crit, v_crit, a_crit, E_crit) = get_verlet(x_i,v_i,t,k,m,b,h,n)
# Heavily Damped Oscillation, b = 2*b_crit
b = 2 * b_crit 
(x_heavy, v_heavy, a_heavy, E_heavy) = get_verlet(x_i,v_i,t,k,m,b,h,n)

# Plot Displacement Graphs
plt.plot(t, x_light, '--', label='$b=0.5b_{crit}$' )
plt.plot(t, x_crit, '-.', label='$b=b_{crit}$')
plt.plot(t, x_heavy, ':', label = '$b=2b_{crit}$')
plt.title('$b_{crit}=%.3f, h=%.3f, n=%i$' % (b,h,n))
plt.xlabel('$Time$ $/s$')
plt.ylabel('$Displacement$ $/m$')
plt.grid()
plt.legend()
plt.show()

# Plot Energy Graphs
plt.plot(t, E_light, '--', label='$b=0.5b_{crit}$' )
plt.plot(t, E_crit, '-.', label='$b=b_{crit}$')
plt.plot(t, E_heavy, ':', label = '$b=2b_{crit}$')
plt.title('$b_{crit}=%.3f, h=%.3f, n=%i$' % (b,h,n))
plt.xlabel('$Time$ $/s$')
plt.ylabel('$Energy$ $/J$')
plt.grid()
plt.legend()
plt.show()

# Plot Phase Space
plt.plot(x_light, v_light, '--', label='$b=0.5b_{crit}$' )
plt.plot(x_crit, v_crit, '-.', label='$b=b_{crit}$')
plt.plot(x_heavy, v_heavy, ':', label = '$b=2b_{crit}$')
plt.title('$b_{crit}=%.3f, h=%.3f, n=%i$' % (b,h,n))
plt.xlabel('$Displacement$ $/m$')
plt.ylabel('$Velocity$ $/ms^-1$')
plt.grid()
plt.legend()
plt.show()











































##************THIS PART TAKES A LONG TIME TO CALCULATE*************************************
##==============================================================
##       DETERMINING ACCURACY OF DIFFERENT STEP SIZES
##==============================================================
## determine how varying step size affects accuracy of verlet method 
## Note: Use Zero Damping for This Part
## step sizes used : 1, 0.1,0.01,0.001,0.0001,0.00001,0.000001'
## accuracy : difference of total sum of E of numerical solution from that that of analytical solution after 10s
#
## store energy sum for different h values after 10s
#E_verlet_sum_h = []
#
## store energy sum for different h values after 10s
#E_ana_sum_h = [] 
#
## calculate energy sum values for verlet method
#for i in [1, 0.1,0.01,0.001,0.0001,0.00001,0.000001]:
#    h = i
#    n = int(10/i)
#    E_verlet = get_verlet(x_i,v_i,t,k,m,b,h,n)[3]
#    E_verlet_sum = E_sum(E_verlet)
#    E_verlet_sum_h.append(E_verlet_sum[-1])
#
## calculate energy sum values for analytical method
#for i in [1, 0.1,0.01,0.001,0.0001,0.00001,0.000001]:
#    E_ana_sum_h.append( E_ana[0] * (10/i) ) 
#
#print('h : 1, 0.1,0.01,0.001,0.0001,0.00001,0.000001')   
#print('Energy Difference after 10s :', np.array(E_verlet_sum_h) - np.array(E_ana_sum_h))
#
##*************************END OF THIS PART**********************************************











