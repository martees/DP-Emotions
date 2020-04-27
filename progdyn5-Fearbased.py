from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw
import progdyn5Annex as anx

##Parameters
def Param(): #in a function so that it is synchronized with progdyn5Annex
    #General
    true=0.1 #density of true predation cues left by predators
    noise=0.01 #density of false predation cues in any environment
    c=8

    #Safe environment
    gamma_S=0 #density of predators
    transition_S=0.01

    #Risky environment
    gamma_R=0.5 #density of predators
    transition_R=0.01

    #Survival function
    m=0.5 #inflexion point
    b=10 #steepness of growth
    def Psur(a):
        return( 1/(1+exp(-b*(a-m))) )

    #Possible values for a
    alist=np.arange(0,1,0.01)

Param()

##Numerical solution of the corresponding Nonacs model
#Function that returns the fitness-maximizing a* for E=S and E=R
#It is numerically computable by finding the maximum of the fitness function (reached at the derivative's root)
#Note: for now we consider that either the safe envt has a 0 predator density (in which case the optimal vigilance level is 0), either both predator densities are non-zero.

A=[0,0]
def NumericOptimal():
    if gamma_S == 0:
        A[0]=0
        A[1]= (-gamma_R*lambertw(exp(-m*b+b-(1/gamma_R))/gamma_R)+b*gamma_R-1)/(b*gamma_R)
    else:
        A[0]= (-gamma_S*lambertw(exp(-m*b+b-(1/gamma_S))/gamma_S)+b*gamma_S-1)/(b*gamma_S)
        A[1]= (-gamma_R*lambertw(exp(-m*b+b-(1/gamma_R))/gamma_R)+b*gamma_R-1)/(b*gamma_R)
    print(A)


##Perfect Information
def G(x):
    return( 1/(1+exp(-20*(x-0.2))) )

#fitness obtained through a given time step from performing a in environment E
def F(a,E):

    if E=='S':
        gamma = gamma_S

    if E=='R':
        gamma = gamma_R

    return( G(1-a) * ( gamma * Psur(a) + (1-gamma)))


# Reproductive value of an animal performing intensity a in environment E
def H(a,V,E):
    #adjust parameters depending on the value of E
    if E=='S':
        h = ((1-transition_S)*F(a,'S')*V[0])+(transition_S*F(a,'R')*V[1])
    else:
        h = ((1-transition_R)*F(a,'R')*V[1])+(transition_R*F(a,'S')*V[0])
    return(h)

def newH(x,a,V,E):
    #adjust parameters depending on the value of E
    if E=='S':
        h = (1-a)*((1-transition_S)* V[1] + transition_S *V[1])**10
    else:
        h = (1-a)*((1-transition_R)* V[1] + transition_R*V[0])**10
    return(h)


# Dynamic programming operator (finds the a that maximizes V for each E, and fills up V and A with the corresponding max)
def T(V,E):
    maxi=0 #maximum H
    maxa=0 #optimal a
    for a in alist: #for many a's
        t = H(a,V,E) #calculate the reproductive value if foraging at a in E
        if t>maxi: #if a better t is found, update maxima
            maxi=t
            maxa=a
    return(maxi,maxa)


# Find optimal strategy for Perfect Information, as the limit of a sequence of functions
def PerfectInfo():
    #V = relative probability of survival for any long period under a optimal strategy
    #newVe are going to build a function sequence in order to obtain the optimal V(E) [ 0 = SAFE, 1 = RISKY ]
    V = [1,1]
    newV = [1,1]
    a_max = [0,0] #maximizing levels of vigilance

    j=0
    maxdiff=100
    #main loop
    while maxdiff>=0.00001: #until the sequence converges
        V=deepcopy(newV) #previous newV is stored in V
        newV = np.zeros(2) #current newV is initialized

        #Process tracking (1)
    #    j += 1 #for nice printing purposes
    #    t =  process_time()
    #    print("Iteration ", j, ", start time : ", t, sep='', end='')

        #main calculation
        newV[0], a_max[0] = T(V,'S')
        newV[1], a_max[1] = T(V,'R')

        #Normalization
        maxi = np.amax(newV)
        if maxi != 0:
            newV = newV/maxi

        #recompute maximum difference btw V and U for convergence
        maxdiff= np.amax(np.abs(newV - V))

    #    #Process tracking (2)
    #    print(", iteration took ", process_time()-t, "s, maxdiff is : ", maxdiff, sep = '')

    print(newV,a_max)


#    plt.imshow(a_max) #plotting foraging intensity matrix F
 #   #plt.imshow(F, interpolation='gaussian') #gaussian smoothing
 #   plt.gca().invert_yaxis()
  #  plt.pause(0.1)
#
 #   plt.colorbar(aspect='auto')
  #  plt.show()
    #


##Bayesian
#p is the probability that E=G according to the animal



#Find optimal strategy as the limit of a sequence of functions
#Here, functions are represented as tables associating a tuple (x,p) to the reproductive value associated with the optimal foraging intensity (relative probability of survival for any long period under a optimal strategy).

def Bayesian():
    #Initial function = associates a foraging intensity of 1 with all possible (x,p) except for (0,p)=0
    #x=LINE, p=COL, environment=TUPLE (0=good, 1=bad)
    U = np.zeros( (s+1,N+1, 2) ) +1
    U[0] = 0

    #Convergence parameter (to compare to maximum acceptable difference between V and U)
    maxdiff=100

    j=0
    #MAIN LOOP
    while maxdiff>=0.001: #until the sequence converges
        V=deepcopy(U) #previous U is stored in V
        U = np.zeros( (s+1,N+1, 2) )

        #Process tracking (1)
        j += 1 #for nice printing purposes
        t =  process_time()
        print("Iteration ", j, ", start time : ", t, sep='', end='')

        #Application of the updated prior permutation to the p columns
        rows, column_indices = np.ogrid[:V.shape[0], :V.shape[1]]
        V = V[rows, np.reshape(updated_prior, (1, N+1))]

        #MAIN CALCULATION: T
        U, F = T() #we recompute F each time (useful only on last iteration, could be worth some work)

        #Normalization
        maxi = np.amax(U)
        if maxi != 0:
            U = U/maxi

        #recompute maximum difference btw V and U for convergence
        maxdiff= np.amax(np.abs(U - V))

        #Process tracking (2)
        print(", iteration took ", process_time()-t, "s, maxdiff is : ", maxdiff, sep = '')

    plt.imshow(F) #plotting foraging intensity matrix F
    #plt.imshow(F, interpolation='gaussian') #gaussian smoothing
    plt.gca().invert_yaxis()
    plt.pause(0.1)

    plt.colorbar(aspect='auto')
    plt.show()



##Imperative commands

NumericOptimal()
PerfectInfo()
Plot()






