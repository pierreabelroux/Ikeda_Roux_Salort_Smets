
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 28-01-2022

"""
This code contains the parameters and functions used to produce the figures of the article "Theoretical study of the emergence of periodic solutions for the inhibitory NNLIF neuron model with synaptic delay" by Kota Ikeda, Pierre Roux, Delphine Salort and Didier Smets.

Running the code will produce all the figures at once, but it is possible to comment some lines in the part "## Production of the figures" in order to produce only some of the figures.

The numerical scheme used in this code for computing a approximations of the solutions of the NNLIF model was proposed by Jingwei Hu, Jian-Guo Liu, Yantong Xie and Zhennan Zhou in the article "A structure preserving numerical scheme for Fokker-Planck equations of neuron networks: numerical analysis and exploration".
"""

__author__ = "Pierre Roux"
__credits__ = ["Pierre Roux", "Kota Ikeda", "Delphine Salort", "Didier Smets"]
__license__ = "GNU General public license v3.0"
__version__ = "2.0"
__maintainer__ = "Pierre Roux"
__email__ = "pierre.rouxmp@laposte.net"
__status__ = "Final"


## Imports

import numpy as np
import scipy.linalg as lng
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint, quad, simps
import scipy.optimize as opt

# Mesh ploting :
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

start_time = time.time()



## Debugging

def get_k(t1,t):
    """
    Given a time grid t and a particular time t1, returns the index k of the time t[k] which is the closest to t1.
    """
    return np.argmin( (t - t1)**2 )


## Plot function

def display_1d_multliple(plots, xlab=None, ylab=None, title=None) :
    """
    Plot y versus x as a line with appropriate legend depending on the options. The "plots" argument is formated like
    plots = [ [ x, y2, 'royalblue', '-.', "label y1" ],
              [ x, y2, 'purple', '--', "label y2" ],
                    ...
              [ x, yn, 'darkgoldenrod', ':', "label yn" ] ]
    """
    fig, ax = plt.subplots()
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.setp(ax.spines.values(), linewidth=3)
    
    for p in plots:
        x, y, col, style, lab = p[0], p[1], p[2], p[3], p[4]
        ax.plot(x,y, color=col, linestyle=style, linewidth=3, label=lab)
    
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)
    if xlab != None :
        ax.set_xlabel(xlab, fontsize=20)
    
    if ylab != None :
        ax.set_ylabel(ylab, fontsize=20)
    
    if title != None :
            ax.set_title(title)
    
    if len(plots)>1 or lab!=None:
        ax.legend(fontsize=20)
    
    fig.show()
    
    return None
    

## Functions underpinning the scheme

def m(v, N, b, a):
    """
    Computes an approximation of M(v,t).
    
    Args:
        v (float): space point value.
        N (float): value of the firing rate.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
    
    Returns:
        float: approximation of M(v,t). 
    """
    return np.exp( -(v-b*N)**2/(2*a) )


def md(v, w, N, b, a):
    """
    Computes an approximation of M(w,t)/M(v,t) in a numerically stable way.
    
    Args:
        v (float): first space point value.
        w (float): second space point value.
        N (float): value of the firing rate.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
    
    Returns:
        float: approximation of M(w,t)/M(v,t).
    """
    return np.exp( (  (v-b*N)**2  -(w-b*N)**2 )/(2*a) )



def matrix_scheme(v, N, b, a, j_r):
    """
    Matrix defining the schemes for the NNLIF model. In order to avoid numerical errors while computing exponentials of large numbers, many terms are rewritten in appropriate form with the function md. The matrix is smaller than the space vector v in both rows and collumns because the extremal values of the solution never change because of the Diritchlet conditions.
    
    Args:
        v (numpy.ndarray): vector of the space grid.
        N (float): value of the firing rate.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
        j_r (int): index of V_R in the space vector v.
    
    Returns:
        numpy.ndarray: matrix used in the schemes.
    """
    M  = np.zeros((v.size-2,v.size-2))
    
    M[0,0] = 2 / ( 1 + md(v[2],v[1],N,b,a) )
    M[0,1] = - 2/ ( 1 + md(v[1], v[2], N, b, a) )
    
    for l in range(1,v.size-3):
        M[l,l] =  2/( 1 + md(v[l], v[l+1], N, b, a) )  +   2/( 1 + md(v[l+2], v[l+1], N, b, a) )
        M[l,l-1] = -  2/( 1 + md(v[l+1], v[l], N, b, a) ) 
        M[l,l+1] = -  2/( 1 + md(v[l+1], v[l+2], N, b, a) ) 
    
    k =-3
    M[-1,-1] = 2 /( 1 + md(v[k], v[k+1], N, b, a) )  + 1
    M[-1,-2] = - 2 /( 1 + md(v[k+1], v[k], N, b, a) )
    
    M[j_r-1,-1] = - 1
    
    return M


##  One step of the scheme for the non-delayed NNLIF, d = 0

def iteration_explicit(p_ex, v, dt, b, a, j_r):
    """
    Given the approximation p_ex of p(v,t) on the space grid v, computes the solution p(v,t+dt) of the non-delayed NNLIF model with parameters a and b at the time t+dt. The method is an explicit scheme, so dt must be small in order to avoid numerical instabilities.
    
    Args:
        p_ex (numpy.ndarray): solution at the time t on the grid v.
        v (numpy.ndarray): vector of the space grid.
        dt (float): time step of the simulation.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
        j_r (int): index of V_R in the vector v.
    
    Returns:
        numpy.ndarray: solution at time t+dt on the grid v.
        float: new firing rate N(t+dt). 
    """
    dv = v[1]-v[0]
    
    N_ex = a/dv * p_ex[-2]
    I = np.eye(v.size-2)
    M = matrix_scheme(v, N_ex, b, a, j_r)

    p_ex[1:-1] = np.dot( I - a*dt/dv**2 * M , p_ex[1:-1] )
    
    return p_ex, N_ex


def iteration_semi_implicit(p_si, v, dt, b, a, j_r):
    """
    Given the approximation p_si of p(v,t) on the space grid v, computes the solution p(v,t+dt) of the non-delayed NNLIF model with parameters a and b at the time t+dt. The method is a semi-implicit scheme.
    
    Args:
        p_si (numpy.ndarray): solution at the time t on the grid v.
        v (numpy.ndarray): vector of the space grid.
        dt (float): time step of the simulation.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
        j_r (int): index of V_R in the vector v.
    
    Returns:
        numpy.ndarray: solution at time t+dt on the grid v.
        float: new firing rate N(t+dt). 
    """
    dv = v[1]-v[0]
    
    N_si = a/dv * p_si[-2]
    I = np.eye(v.size-2)
    M = matrix_scheme(v, N_si, b, a, j_r)
    
    p_si[1:-1] = lng.solve( I + a*dt/dv**2 * M , p_si[1:-1] )
    
    return p_si, N_si


## Stationary states and relative entropy

def steady_state(v, b, a, j_r) :
    """
    Computes on a space grid v the stationary state of the NNLIF model with parameters a and b. The function starts from a Gaussian initial condition and iterates the semi-implicit scheme a large number of times. Be careful that when b is positive and large enough there can be multiple stationary states or not any.
    
    Args:
        p_si (numpy.ndarray): solution at the time t on the grid v.
        v (numpy.ndarray): vector of the space grid.
        dt (float): time step of the simulation.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
        j_r (int): index of V_R in the vector v.
    
    Returns:
        numpy.ndarray: solution at time t+dt on the grid v.
        float: new firing rate N(t+dt). 
    """
    dv = v[1]-v[0]
    
    p_inf = np.zeros(v.size)
    p_inf[1:-1] = np.exp( -(v[1:-1]-v[int(v.size* 2/3) ] )**2/(2*a) )
    p_inf = p_inf / (sum(p_inf)*dv)
    
    dt_loc = dv/2
    for k in range(1000):
        p_inf, N_inf = iteration_semi_implicit(p_inf, v, dt_loc, b, a, j_r)
        
    return p_inf, N_inf


def G_quad(x):
    """
    Quadratic choice of the convex function for the relative entropy computation.
    """
    return (x-1)**2


def G_log(x):
    """
    Logarithmic choice of the convex function for the relative entropy computation.
    """
    return x*np.log(x)

    
def relative_entropy(G, p, p_inf, v) :
    """
    Computes the relative entropy associated to a convex function G of the solution p with respect to the stationary state p_inf, both given on a same space grid v. The relative entropy is the integral on the space domain of the function G( p / p_inf ) p_inf.
    
    Args:
        G: convex function which defines the relative entropy.
        p (numpy.ndarray): approximation of the solution p(v,t) on the grid v.
        p_inf (numpy.ndarray): approximation of the stationary state on the grid v.
        v (numpy.ndarray): vector of the space grid.

    
    Returns:
        float: numerical value of the relative entropy. 
    
    """
    return simps(   np.abs( p_inf[1:-1] * G( p[1:-1] / p_inf[1:-1] )), v[1:-1])


##  One step of the scheme for the delayed NNLIF, d > 0

def iteration_explicit_delay(p_ex, N, v, dt, b, a, j_r):
    """
    Given the approximation p_ex of p(v,t) on the space grid v, and the past values of the firing rate N(s), s in [t-d,t], computes the solution p(v,t+dt) of the delayed NNLIF model with parameters a and b at the time t+dt. The value of the delay is given by the length of the vector N and the value of dt (hence, dt must be the same at each step of the run). The method is an explicit scheme, so dt must be small in order to avoid numerical instabilities.
    
    Args:
        p_ex (numpy.ndarray): solution at the time t on the grid v.
        N (numpy.ndarray): past values of the firing rate on [t-d,t].
        v (numpy.ndarray): vector of the space grid.
        dt (float): time step of the simulation.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
        j_r (int): index of V_R in the vector v.
    
    Returns:
        numpy.ndarray: solution at time t+dt on the grid v.
        numpy.ndarray: updated past values of the firing rate on [t+dt-d,t+dt]. 
    """
    dv = v[1]-v[0]
    
    N_tmp = a/dv * p_ex[-2]
    I = np.eye(v.size-2)
    M = matrix_scheme(v, N[0], b, a, j_r)

    p_ex[1:-1] = np.dot( I - a*dt/dv**2 * M , p_ex[1:-1] )
    
    N[:-1] = N[1:]
    N[-1] = N_tmp
    
    return p_ex, N


def iteration_semi_implicit_delay(p_si, N, v, dt, b, a, j_r):
    """
    Given the approximation p_si of p(v,t) on the space grid v, and the past values of the firing rate N(s), s in [t-d,t], computes the solution p(v,t+dt) of the delayed NNLIF model with parameters a and b at the time t+dt. The value of the delay is given by the length of the vector N and the value of dt (hence, dt must be the same at each step of the run). The method is a semi-implicit scheme.
    
    Args:
        p_si (numpy.ndarray): solution at the time t on the grid v.
        N (numpy.ndarray): past values of the firing rate on [t-d,t].
        v (numpy.ndarray): vector of the space grid.
        dt (float): time step of the simulation.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
        j_r (int): index of V_R in the vector v.
    
    Returns:
        numpy.ndarray: solution at time t+dt on the grid v.
        numpy.ndarray: updated past values of the firing rate on [t+dt-d,t+dt]. 
    """
    dv = v[1]-v[0]
    
    N_tmp = a/dv * p_si[-2]
    I = np.eye(v.size-2)
    M = matrix_scheme(v, N[0], b, a, j_r)
    
    p_si[1:-1] = lng.solve( I + a*dt/dv**2 * M , p_si[1:-1] )
    
    N[:-1] = N[1:]
    N[-1] = N_tmp
    
    return p_si, N


## Scheme for the Delay Differential equation

def solution_ode(c0, N_d, N_list, t, b):
    """
    Given the firing rate N_list on [0,T] of the PDE and its initial values N_d on [-d,0], computes the solution of the ODE
         c'(t) + c(t) = b N(t-d)
    with initial condition c(0)=c0. The approximation of c on [0,T] is returned on the time grid t.
    
    Args:
        c0 (float): initial condition c(0)=c0.
        N_d (numpy.ndarray): values of N(t) on [-d,0].
        N_list (numpy.ndarray): values of N(t) on the grid t representing [0,T].
        t (numpy.ndarray): time grid.
        b (float): connectivity parameter.
    
    Returns:
        numpy.ndarray: approximation of the solution c(t) on the time grid t.
    """
    dt = t[1]-t[0]
    j_d = N_d.size

    c = np.zeros(t.size)
    c[0] = c0

    for k in range(1,t.size):
        if k<j_d:
            c[k] = (2-dt)/(2+dt) * c[k-1] + 2*dt/(2+dt) * b * N_d[k]
        else:
            c[k] = (2-dt)/(2+dt) * c[k-1] + 2*dt/(2+dt) * b * N_list[k-j_d]
    
    return c

def soliton(v, c, a):
    """
    Computes the solitonic wave of mean c and variance a on the space grid v.
    """
    return 1/np.sqrt(2*np.pi*a) * np.exp( - (v - c)**2 / (2*a) )

def solution_ode_extended(m1_0, var_0, N_d, N_list, t, b, V_R, V_F, a):
    """
    Gives the solution to the complete systems defining the mean and variance of the PDE solution.
    """
    dt = t[1]-t[0]
    j_d = N_d.size

    m1 = np.zeros(t.size)
    var = np.zeros(t.size)
    m1[0] = m1_0
    var[0] = var_0
    
    for k in range(1,t.size):
        if k<j_d:
            m1[k] = (2-dt)/(2+dt) * m1[k-1] + 2*dt/(2+dt) * ( b * N_d[k] + (V_R-V_F)*N_list[k] )
            var[k] = (1-dt)/(1+dt) * var[k-1]   + dt/(1+dt) * ( 2*a + (V_F-V_R)*( 2*m1[k] - V_R-V_F )*N_list[k]  )
        else:
            m1[k] = (2-dt)/(2+dt) * m1[k-1] + 2*dt/(2+dt) * ( b * N_list[k-j_d] + + (V_R-V_F)*N_list[k] )
            var[k] = (1-dt)/(1+dt) * var[k-1]   + dt/(1+dt) * ( 2*a + (V_F-V_R)*( 2*m1[k] - V_R-V_F )*N_list[k]  )
    
    return m1, var

def N_dde(c, a, V_F):
    """
    Function giving the firing rate of the DDE with the value c and the parameters a and V_F.
    """
    return 1/np.sqrt(2*np.pi*a) * (V_F - c) * np.exp( - (V_F - c)**2 / (2*a) )

def solution_dde(c0, c_d, t, b, a, V_F):
    """
    Given the parameters b, a, V_F, the values c_d of c(t) on [-d,0] and the initial condition c(0)=c0, computes the solution of 
        c'(t) + c(t) = b N_dde( c(t-d) )
    on the time grid t.
    
    Args:
        c0 (float): initial condition c(0)=c0.
        c_d (numpy.ndarray): values of c(t) on [-d,0].
        t (numpy.ndarray): time grid.
        b (float): connectivity parameter.
        a (float): diffusion parameter.
        V_F (float): value of the firing potential.
    
    Returns:
        numpy.ndarray: approximation of the solution c(t) on the time grid t.
        numpy.ndarray: approximation of firing rate  N_dde( c(t) ) on the time grid t.
    """
    dt = t[1]-t[0]
    j_d = c_d.size

    c = np.zeros(t.size)
    c[0] = c0
    
    N_dde_list = np.zeros(t.size)
    N_dde_list[0] = N_dde(c0, a, V_F)
    for k in range(1,t.size):
        if k<j_d:
            c[k] = (2-dt)/(2+dt) * c[k-1] + 2*dt/(2+dt) * b * N_dde(c_d[k],a,V_F)
            N_dde_list[k] = N_dde(c[k],a,V_F)
        else:
            c[k] = (2-dt)/(2+dt) * c[k-1] + 2*dt/(2+dt) * b * N_dde(c[k-j_d],a,V_F)
            N_dde_list[k] = N_dde(c[k],a,V_F)
    
    return c, N_dde_list


###############      Figures




    ## # # #    Figure 2 and 5A and 6A and 6B   # # # ##

def figure_2_5A_6A_6B():
    """
    Computes and plots the figures 2, 5A, 6A, 6B.
    """    
    # Spatial grid and space step
    V_min = -3
    V_R = 0
    V_F = 1
    v = np.linspace(V_min,V_F,300)
    dv = v[1]-v[0]
    
    # j is the index of v=V_R
    j_r=0
    while v[j_r] < V_R :
        j_r += 1
    
    # Temporal grid and time step
    T_max = 25
    t = np.linspace(0,T_max,400)
    dt = t[1]-t[0]
    
    # Parameters of the equation
    a = 0.2
    b = -45
    d = 1
    
    # Initial condition in space p_0(v)
    
    p_0 = np.zeros(v.size)
    p_0[1:-1] = np.exp(-(v[1:-1]+1)**2 / (2*a) )
    p_0 = p_0 / (sum(p_0)*dv)
    
    p_si = p_0.copy()
    
    # Initial condition N(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    N = np.zeros(j_d)
    
    # Stationary state
    p_inf, N_inf = steady_state(v, b, a, j_r)
    
    # Simulation of the solution to the PDE
    
    N_list  = np.zeros(t.size)
    moments = np.zeros(t.size)
    max_sol = np.zeros(t.size)
    entropy = np.zeros(t.size)
    full_si = np.zeros((t.size,v.size))
    
    for k in range(t.size):
        p_si, N = iteration_semi_implicit_delay(p_si, N, v, dt, b, a, j_r)
        
        N_list[k]    = N[-1]
        moments[k]   = simps( v * p_si , v )
        max_sol[k]   = v[np.argmax( p_si )]
        entropy[k]   = relative_entropy(G_log, p_si, p_inf, v)
        full_si[k,:] = p_si.copy()
    
    
    k1 = 329
    k2 = 341
    k3 = 354
    
    # Figure 2A
    
    fig_1A = [ [v, full_si[k1], 'navy',  '-', "t={:.2f}".format(t[k1])],
               [v, full_si[k2], 'blue',  '--', "t={:.2f}".format(t[k2])],
               [v, full_si[k3], 'darkgoldenrod',  '-.', "t={:.2f}".format(t[k3])] ]
    display_1d_multliple(fig_1A, xlab="v", ylab="$p(v,t)$", title=None)
    
    # Figure 2B
    
    fig_1B = [ [t, N_list, 'navy',  '-', "N(t)"],
               [t, N_inf*np.ones(t.size), 'darkgoldenrod',  '--', "$N_\infty$"] ]
    display_1d_multliple(fig_1B, xlab="t", ylab="N(t)", title=None)
    
    # Figure 2C
    
    fig_1C = [ [t, max_sol, 'navy',  '-', "Maximum point"],
               [t, moments, 'darkgoldenrod',  '--', "First moment"] ]
    display_1d_multliple(fig_1C, xlab="t", ylab="PDE movement", title=None)
    
    # Figure 2D
    
    fig_1D = [ [t, entropy, 'navy',  '-', None]  ]
    display_1d_multliple(fig_1D, xlab="t", ylab="$\mathcal{S}(p|p_\infty)(t)$", title=None)
    
    
    # Figure 5A
    
    c0 = moments[0]
    N_d = np.zeros(j_d)
    
    c = solution_ode(c0, N_d, N_list, t, b)
    
    fig_4A = [ [t, max_sol, 'navy',  '-', "Maximum point"],
               [t, moments, 'darkgoldenrod',  '--', "First moment"],
               [t, c, 'purple',  '-.', "ODE solution"] ]
    display_1d_multliple(fig_4A, xlab="t", ylab="PDE vs ODE", title=None)
    
    # Figure 6A
    
    fig_5A = [ [v, full_si[k1] - soliton(v, c[k1], a), 'navy',  '-', "R(v,t) t={:.2f}".format(t[k1])],
               [v, full_si[k2] - soliton(v, c[k2], a), 'blue',  '--', "R(v,t) t={:.2f}".format(t[k2])],
               [v, full_si[k3] - soliton(v, c[k3], a), 'darkgoldenrod',  '-.', "R(v,t) t={:.2f}".format(t[k3])] ]
    display_1d_multliple(fig_5A, xlab="v", ylab="R(v,t)", title=None)
    
    # Figure 6B
    
    fig_5B = [ [v, full_si[k1], 'navy',  '-', "PDE t={:.2f}".format(t[k1])],
               [v, soliton(v, c[k1], a), 'blue',  '--', "$\phi(v-c(t))$ t={:.2f}".format(t[k1])],
               [v, full_si[k2], 'darkgoldenrod',  '-', "PDE t={:.2f}".format(t[k2])],
               [v, soliton(v, c[k2], a), 'yellow',  '--', "$\phi(v-c(t))$ t={:.2f}".format(t[k2])],
               [v, full_si[k3], 'darkgreen',  '-', "PDE t={:.2f}".format(t[k3])],
               [v, soliton(v, c[k3], a), 'forestgreen',  '--', "$\phi(v-c(t))$ t={:.2f}".format(t[k3])] ]
    display_1d_multliple(fig_5B, xlab="v", ylab="$p(v,t)$ vs $\phi(v-c(t))$", title=None)
    
    return None



    ## # # #    Figure 3   # # # ##


def figure_3():
    """
    Computes and plots the figure 3.
    """
    # Spatial grid and space step
    V_min = -6
    V_R = 0
    V_F = 1
    v = np.linspace(V_min,V_F,300)
    dv = v[1]-v[0]
    
    # j is the index of v=V_R
    j_r=0
    while v[j_r] < V_R :
        j_r += 1
    
    # Temporal grid and time step
    T_max = 40
    t = np.linspace(0,T_max,400)
    dt = t[1]-t[0]
    
    # Parameters of the equation
    a = 0.2
    b = -45
    d = 1
    
    # Initial condition in space p_0(v)
    
    p_0 = np.zeros(v.size)
    p_0[1:-1] = np.exp(-(v[1:-1]+1)**2 / (2*a) )
    p_0 = p_0 / (sum(p_0)*dv)
    
    # Initial condition N(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    N = np.zeros(j_d)
    
    b_list = np.linspace(-1000,-50,20)
    
    N_norm_linf = np.zeros(b_list.size)
    N_norm_l1 = np.zeros(b_list.size)
    for i in range(b_list.size):
    # Simulation of the solution to the PDE
        N_list  = np.zeros(t.size)
        p_si = p_0.copy()
        b = b_list[i]
        for k in range(t.size):
            p_si, N = iteration_semi_implicit_delay(p_si, N, v, dt, b, a, j_r)
            
            N_list[k]    = N[-1]
        
        N_norm_l1[i]   = simps( N_list[100:] / (t[-1]-t[100]) , t[100:])
        N_norm_linf[i] = np.max(N_list[100:])
        
        c0 = -1
        N_d = np.zeros(j_d)
        c = solution_ode(c0, N_d, N_list, t, b)
        
        # Figure 3B
        if i==0:            
            fig_2B = [ [t, N_list, 'navy',  '-', None]  ]
            display_1d_multliple(fig_2B, xlab="t", ylab="N(t)", title=None)
        
    
    # Figure 3A
    fig_2A = [ [b_list, N_norm_l1, 'navy',  '-', "$L^1$ norm"],
               [b_list, N_norm_linf, 'purple',  '--', "$L^\infty$ norm"] ]
    display_1d_multliple(fig_2A, xlab="Parameter b", ylab="Norms of N(t)", title=None)
    
    
    
    return None
    
    ## # # #    Figure 4 and 5B and 6C and 6D   # # # ##


    
def figure_4_5B_6C_6D():
    """
    Computes and plots the figures 4, 5B, 6C, 6D.
    """
    # Spatial grid and space step
    V_min = -9
    V_R = -2
    V_F = 0
    v = np.linspace(V_min,V_F,1000)
    dv = v[1]-v[0]
    
    # j is the index of v=V_R
    j_r=0
    while v[j_r] < V_R :
        j_r += 1
    
    # Temporal grid and time step
    T_max = 15
    t = np.linspace(0,T_max,5000)
    dt = t[1]-t[0]
    
    # Parameters of the equation
    a = 0.2
    b = -35
    d = 1
    
    # Initial condition in space p_0(v)
    
    p_0 = np.zeros(v.size)
    p_0[1:-1] = np.exp(-(v[1:-1]+3)**2 / (2*a) )
    p_0 = p_0 / (sum(p_0)*dv)
    
    p_si = p_0.copy()
    
    # Initial condition N(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    N = np.zeros(j_d)
    
    # Stationary state
    p_inf, N_inf = steady_state(v, b, a, j_r)
    
    # Simulation of the solution to the PDE
    
    k1 = 1790
    k2 = 2000
    k3 = 2166
    
    N_list  = np.zeros(t.size)
    moments = np.zeros(t.size)
    max_sol = np.zeros(t.size)
    entropy = np.zeros(t.size)
    # full_si = np.zeros((t.size,v.size))
    full_si = np.zeros((3,v.size))
    
    for k in range(t.size):
        p_si, N = iteration_semi_implicit_delay(p_si, N, v, dt, b, a, j_r)
        
        N_list[k]    = N[-1]
        moments[k]   = simps( v * p_si , v )
        max_sol[k]   = v[np.argmax( p_si )]
        entropy[k]   = relative_entropy(G_log, p_si, p_inf, v)
        
        if k==k1 :
            full_si[0,:] = p_si.copy()
        elif k==k2 :
            full_si[1,:] = p_si.copy()
        elif k==k3 :
            full_si[2,:] = p_si.copy()
    
    
    # Figure 4A
    
    fig_3A = [ [v, full_si[0], 'firebrick',  '-', "t={:.2f}".format(t[k1])],
               [v, full_si[1], 'red',  '--', "t={:.2f}".format(t[k2])],
               [v, full_si[2], 'orange',  '-.', "t={:.2f}".format(t[k3])] ]
    display_1d_multliple(fig_3A, xlab="v", ylab="$p(v,t)$", title=None)
    
    # Figure 4B
    
    fig_3B = [ [t, N_list, 'firebrick',  '-', "N(t)"],
               [t, N_inf*np.ones(t.size), 'orange',  '--', "$N_\infty$"] ]
    display_1d_multliple(fig_3B, xlab="t", ylab="N(t)", title=None)
    
    # Figure 4C
    
    fig_3C = [ [t, max_sol, 'firebrick',  '-', "Maximum point"],
               [t, moments, 'orange',  '--', "First moment"] ]
    display_1d_multliple(fig_3C, xlab="t", ylab="PDE movement", title=None)
    
    # Figure 4D
    
    fig_3D = [ [t, entropy, 'firebrick',  '-', None]  ]
    display_1d_multliple(fig_3D, xlab="t", ylab="$\mathcal{S}(p|p_\infty)(t)$", title=None)
    
    # Figure 5B
    
    c0 = moments[0]
    N_d = np.zeros(j_d)
    
    c = solution_ode(c0, N_d, N_list, t, b)
    
    fig_4B = [ [t, max_sol, 'firebrick',  '-', "Maximum point"],
               [t, moments, 'orange',  '--', "First moment"],
               [t, c, 'purple',  '-.', "ODE solution"] ]
    display_1d_multliple(fig_4B, xlab="t", ylab="PDE vs ODE", title=None)
    
    
    # Figure 6C
    
    fig_5C = [ [v, full_si[0] - soliton(v, c[k1], a), 'firebrick',  '-', "R(v,t) t={:.2f}".format(t[k1])],
               [v, full_si[1] - soliton(v, c[k2], a), 'red',  '--', "R(v,t) t={:.2f}".format(t[k2])],
               [v, full_si[2] - soliton(v, c[k3], a), 'orange',  '-.', "R(v,t) t={:.2f}".format(t[k3])] ]
    display_1d_multliple(fig_5C, xlab="v", ylab="R(v,t)", title=None)
    
    # Figure 6D
    
    fig_5D = [ [v, full_si[0], 'firebrick',  '-', "PDE t={:.2f}".format(t[k1])],
               [v, soliton(v, c[k1], a), 'red',  '--', "$\phi(v-c(t))$ t={:.2f}".format(t[k1])],
               [v, full_si[1], 'orange',  '-', "PDE t={:.2f}".format(t[k2])],
               [v, soliton(v, c[k2], a), 'yellow',  '--', "$\phi(v-c(t))$ t={:.2f}".format(t[k2])],
               [v, full_si[2], 'magenta',  '-', "PDE t={:.2f}".format(t[k3])],
               [v, soliton(v, c[k3], a), 'purple',  '--', "$\phi(v-c(t))$ t={:.2f}".format(t[k3])] ]
    display_1d_multliple(fig_5D, xlab="v", ylab="$p(v,t)$ vs $\phi(v-c(t))$", title=None)
    
    return None



    ## # # #    Figure 7A and 7B      # # # ##


def figure_7A_7B():
    """
    Computes and plots the figures 7A and 7B.
    """
    # Spatial grid and space step
    V_min = -3
    V_R = 0
    V_F = 1
    v = np.linspace(V_min,V_F,300)
    dv = v[1]-v[0]
    
    # j is the index of v=V_R
    j_r=0
    while v[j_r] < V_R :
        j_r += 1
    
    # Temporal grid and time step
    T_max = 30
    t = np.linspace(0,T_max,400)
    dt = t[1]-t[0]
    
    # Parameters of the equation
    a = 0.2
    b = -45
    d = 1
    
    # Initial condition in space p_0(v)
    
    p_0 = np.zeros(v.size)
    p_0[1:-1] = np.exp(-(v[1:-1]+1)**2 / (2*a) )
    p_0 = p_0 / (sum(p_0)*dv)
    
    p_si = p_0.copy()
    
    # Initial condition N(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    N = np.zeros(j_d)
    
    # Simulation of the solution to the PDE
    
    N_list  = np.zeros(t.size)
    moments = np.zeros(t.size)
    max_sol = np.zeros(t.size)
    
    for k in range(t.size):
        p_si, N = iteration_semi_implicit_delay(p_si, N, v, dt, b, a, j_r)
        
        N_list[k]    = N[-1]
        moments[k]   = simps( v * p_si , v )
        max_sol[k]   = v[np.argmax( p_si )]
    
    # Simulation of the DDE
    
    c_d = np.zeros(j_d)
    c0 = -1
    
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    
    # Figure 7A
    
    fig_6A = [ [t[53:], max_sol[53:], 'navy',  '-', "PDE solution"],
               [t[53:], c[43:-10], 'purple',  '--', "DDE solution"] ]
    display_1d_multliple(fig_6A, xlab="t", ylab="Movement of the PDE and the DDE", title=None)
    
    # Figure 7B
    
    fig_6B = [ [t[53:], N_list[53:], 'navy',  '-', "$N_{pde}$"],
               [t[53:], N_dde_list[43:-10], 'purple',  '--', "$N_{dde}$"] ]
    display_1d_multliple(fig_6B, xlab="t", ylab="Firing rates $N_{pde}(t)$ and $N_{dde}(t)$", title=None)
    
    return None


    
    ## # # #    Figure 7C and 7D     # # # ##


def figure_7C_7D():
    """
    Computes and plots the figures 7C and 7D.
    """
    # Spatial grid and space step
    V_min = -8
    V_R = -2
    V_F = 0
    v = np.linspace(V_min,V_F,500)
    dv = v[1]-v[0]
    
    # j is the index of v=V_R
    j_r=0
    while v[j_r] < V_R :
        j_r += 1
    
    # Temporal grid and time step
    T_max = 25
    t = np.linspace(0,T_max,1000)
    dt = t[1]-t[0]
    
    # Parameters of the equation
    a = 0.2
    b = -35
    d = 1
    
    # Initial condition in space p_0(v)
    
    p_0 = np.zeros(v.size)
    p_0[1:-1] = np.exp(-(v[1:-1]+5)**2 / (2*a) )
    p_0 = p_0 / (sum(p_0)*dv)
    
    p_si = p_0.copy()
    
    # Initial condition N(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    N = np.zeros(j_d)
    
    # Simulation of the solution to the PDE
    
    N_list  = np.zeros(t.size)
    moments = np.zeros(t.size)
    max_sol = np.zeros(t.size)
    
    for k in range(t.size):
        p_si, N = iteration_semi_implicit_delay(p_si, N, v, dt, b, a, j_r)
        
        N_list[k]    = N[-1]
        moments[k]   = simps( v * p_si , v )
        max_sol[k]   = v[np.argmax( p_si )]
    
    # Simulation of the DDE
    
    c_d = np.zeros(j_d)
    c0 = -5
    
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    
    # Figure 7C
    
    fig_6C = [ [t[180:-40], max_sol[180:-40], 'firebrick',  '-', "PDE solution"],
               [t[180:-40], c[220:], 'purple',  '--', "DDE solution"] ]
    display_1d_multliple(fig_6C, xlab="t", ylab="Movement of the PDE and the DDE", title=None)
    
    # Figure 7D
    
    fig_6D = [ [t[110:-40], N_list[110:-40], 'firebrick',  '-', "$N_{pde}$"],
               [t[110:-40], N_dde_list[150:], 'purple',  '--', "$N_{dde}$"] ]
    display_1d_multliple(fig_6D, xlab="t", ylab="Firing rates $N_{pde}(t)$ and $N_{dde}(t)$", title=None)
    
    
    return None


    ## # # #    Figure 8     # # # ##   


def figure_8():
    """
    Computes and plots the figure 8.
    """
    # DDE parameters
    V_F = 1
    a = 0.2
    d = 1
    
    # Temporal grid and time step
    T_max = 30
    t = np.linspace(0,T_max,1000)
    dt = t[1]-t[0]
    
    # Initial condition c_d(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    c_d = np.zeros(j_d)
    c0 = -1
    
    
    # Figure 8A
    
    b = -50
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    fig_7A = [ [t[166:], c[166:], 'navy',  '-', None]  ]
    display_1d_multliple(fig_7A, xlab="t", ylab="c(t)", title=None)

    # Figure 8B
    
    b = -100
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    fig_7B = [ [t[166:], c[166:], 'navy',  '-', None]  ]
    display_1d_multliple(fig_7B, xlab="t", ylab="c(t)", title=None)

    # Figure 8C
    
    b = -1000
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    fig_7C = [ [t[166:], c[166:], 'navy', '-', None]  ]
    display_1d_multliple(fig_7C, xlab="t", ylab="c(t)", title=None)

    # Figure 8D
    
    b = -5000
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    fig_7D = [ [t[166:], c[166:], 'navy',  '-', None]  ]
    display_1d_multliple(fig_7D, xlab="t", ylab="c(t)", title=None)
    
    return None


    ## # # #    Figure 9     # # # ##


def figure_9():
    """
    Computes and plots the figure 9.
    """
    # DDE parameters
    V_F = 0
    a = 0.2
    d = 1
    
    # Temporal grid and time step
    T_max = 30
    t = np.linspace(0,T_max,1000)
    dt = t[1]-t[0]
    
    # Initial condition c_d(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    c_d = np.zeros(j_d)
    c0 = -5 
    
    
    # Figure 9A
    
    b = -50
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    fig_8A = [ [t, c, 'firebrick',  '-', None]  ]
    display_1d_multliple(fig_8A, xlab="t", ylab="c(t)", title=None)
    
    # Figure 9B
    
    b = -100
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    fig_8B = [ [t, c, 'firebrick',  '-', None]  ]
    display_1d_multliple(fig_8B, xlab="t", ylab="c(t)", title=None)
    
    # Figure 9C
    
    b = -1000
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    fig_8C = [ [t, c, 'firebrick',  '-', None]  ]
    display_1d_multliple(fig_8C, xlab="t", ylab="c(t)", title=None)
    
    # Figure 9D
    
    b = -5000
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    fig_8D = [ [t, c, 'firebrick',  '-', None]  ]
    display_1d_multliple(fig_8D, xlab="t", ylab="c(t)", title=None)
    
    return None



    ## # # #    Figure 10     # # # ##

def figure_10():
    """
    Computes and plots the figure 10.
    """
    # DDE parameters
    V_F = 0
    a = 0.2
    d = 1
    
    # Temporal grid and time step
    T_max = 20
    t = np.linspace(0,T_max,10000)
    dt = t[1]-t[0]
    
    # Initial condition c_d(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    c_d = np.zeros(j_d)
    c0 = -5 
    
    C_M = 0.8015
    
    
    # Figure 10A
    
    b = -50
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)

    beta = 1 / np.log( - b/np.sqrt(2*np.pi*a) )
    
    t_beta = t[::int(1/beta)]/int(1/beta)
    u = beta * np.log( - 1/np.sqrt(a)*c[::int(1/beta)] )
    
    fig_9A = [ [t_beta, u, 'firebrick',  '-', "$u_β$"],
               [t_beta, C_M * np.ones(t_beta.size) , 'orange',  '--', "$C_M$"] ]
    display_1d_multliple(fig_9A, xlab="t", ylab="$u_β(t)$", title=None)
    
    # Figure 10B
    
    b = -100
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    beta = 1 / np.log( - b/np.sqrt(2*np.pi*a) )
    
    t_beta = t[::int(1/beta)]/int(1/beta)
    u = beta * np.log( - 1/np.sqrt(a)*c[::int(1/beta)] )
    
    fig_9B = [ [t_beta, u, 'firebrick',  '-', "$u_β$"],
               [t_beta, C_M * np.ones(t_beta.size) , 'orange',  '--', "$C_M$"] ]
    display_1d_multliple(fig_9B, xlab="t", ylab="$u_β(t)$", title=None)
    
    # Figure 10C
    
    b = -1000
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    beta = 1 / np.log( - b/np.sqrt(2*np.pi*a) )
    
    t_beta = t[::int(1/beta)]/int(1/beta)
    u = beta * np.log( - 1/np.sqrt(a)*c[::int(1/beta)] )
    
    fig_9C = [ [t_beta, u, 'firebrick',  '-', "$u_β$"],
               [t_beta, C_M * np.ones(t_beta.size) , 'orange',  '--', "$C_M$"] ]
    display_1d_multliple(fig_9C, xlab="t", ylab="$u_β(t)$", title=None)
    
    
    # Figure 10D
    
    b = -50000
    c, N_dde_list = solution_dde(c0, c_d, t, b, a, V_F)
    
    beta = 1 / np.log( - b/np.sqrt(2*np.pi*a) )
    
    t_beta = t[::int(1/beta)]/int(1/beta)
    u = beta * np.log( - 1/np.sqrt(a)*c[::int(1/beta)] )
    
    fig_9D = [ [t_beta, u, 'firebrick',  '-', "$u_β$"],
               [t_beta, C_M * np.ones(t_beta.size) , 'orange',  '--', "$C_M$"] ]
    display_1d_multliple(fig_9D, xlab="t", ylab="$u_β(t)$", title=None)

    return None



    ## # # #    Figure 11     # # # ##    

def figure_11():
    """
    Computes and plots the figure 11.
    """
    # DDE parameters
    V_F = 0
    a = 0.2
    d = 1
    
    # Temporal grid and time step
    T_max = 20
    t = np.linspace(0,T_max,10000)
    dt = t[1]-t[0]
    
    # Initial condition c_d(t) in time, defined between -d and 0
    j_d = 0
    while j_d*dt < d:
        j_d += 1
    c_d = np.zeros(j_d)
    c0 = -5 
    
    b_list = np.linspace(-50,-5000000,100)
    c_range = np.zeros(b_list.size)
    for i in range(b_list.size) :
        # print(i)
        c, N_dde_list = solution_dde(c0, c_d, t, b_list[i], a, V_F)
        c_range[i] = np.max(c[int(t.size/2):])-np.min(c[int(t.size/2):])
    
    C_M = 0.8015
    
    fig_10 = [ [b_list, c_range, 'firebrick',  '-', "Range of c"],
               [b_list, np.sqrt(a)*(-b_list/np.sqrt(2*np.pi*a))**C_M , 'orange',  '--', "$\sqrt{a} (-b/\sqrt{2\pi a})^{C_M}$"] ]
    display_1d_multliple(fig_10, xlab="Parameter b", ylab="Range of c(t)", title=None)
    
    return None


## Production of the figures


if __name__ == "__main__" :
    
    """ Total time to compute all the figures at once on a processor Intel Core i7-10510U CPU @ 1.80GHz 2.30 GHz and a 16,0 Go RAM: around 500 seconds """
    
    
    figure_2_5A_6A_6B()      # Computation time: 10 seconds
    figure_3()               # Computation time: 60 seconds
    figure_4_5B_6C_6D()      # Computation time: 310 seconds
    figure_7A_7B()           # Computation time: 5 seconds
    figure_7C_7D()           # Computation time: 15 seconds
    figure_8()               # Computation time: 1 seconds
    figure_9()               # Computation time: 1 seconds
    figure_10()              # Computation time: 1 seconds
    figure_11()              # Computation time: 10 seconds
    
    print("done")



## Computation time

end_time = time.time()

if __name__ == "__main__" :
    print("Execution time : ",end_time-start_time)
