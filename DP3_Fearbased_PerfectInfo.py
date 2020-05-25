from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw


##Total fitness obtained through a given time step from performing a in environment E
#This function is used in Main to compare the performances of the different heuristics.
def F(a,E):
    if E=='S':
        gamma = gamma_S
    if E=='R':
        gamma = gamma_R
    return( G(1-a) * ( gamma * Psur(a) + (1-gamma)))


##Long-term reproductive value of an animal performing intensity a in environment E
def H1(a,V,E):
    #adjust parameters depending on the value of E
    if E=='S':
        h = ((1-transition_S)*F(a,'S')*V[0])+(transition_S*F(a,'R')*V[1])
    else:
        h = ((1-transition_R)*F(a,'R')*V[1])+(transition_R*F(a,'S')*V[0])
    return(h)


## Dynamic programming operator (finds the a that maximizes V for each E, and fills up V and A with the corresponding max)
def T1(V,E):
    maxi=0 #maximum H
    maxa=0 #optimal a
    for a in alist: #for many a's
        t = H1(a,V,E) #calculate the reproductive value if foraging at a in E
        if t>maxi: #if a better t is found, update maxima
            maxi=t
            maxa=a
    return(maxi,maxa)


## Find optimal strategy for Perfect Information, as the limit of a sequence of functions
def PerfectInfo():
    #V = relative probability of survival for any long period under a optimal strategy
    #newVe are going to build a function sequence in order to obtain the optimal V(E) [ 0 = SAFE, 1 = RISKY ]
    exec(open("param.txt").read(),globals()) #executing parameter file
    V = [1,1]
    newV = [1,1]
    a_max = [0,0] #maximizing levels of vigilance
    maxdiff=100 #convergence criterion
    #main loop
    while maxdiff>=0.00001: #until the sequence converges
        V=deepcopy(newV) #previous newV is stored in V
        newV = np.zeros(2) #current newV is initialized
        #main calculation
        newV[0], a_max[0] = T1(V,'S')
        newV[1], a_max[1] = T1(V,'R')
        #Normalization
        maxi = np.amax(newV)
        if maxi != 0:
            newV = newV/maxi
        #recompute maximum difference btw V and newV for convergence
        maxdiff= np.amax(np.abs(newV - V))
    return(a_max)



