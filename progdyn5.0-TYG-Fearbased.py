from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from time import process_time
from scipy.special import lambertw

##Parameters
#General
true=0.1 #density of true predation cues left by predators
noise=0.01 #density of false predation cues in any environment

#Safe environment
gamma_S=0.5 #density of predators
transition_S=0.01

#Risky environment
gamma_R=0.9 #density of predators
transition_R=0.01

#Survival function
m=0.5 #inflexion point
b=20 #steepness of growth
def Psur(a):
    return( 1/(1+math.exp(-b*(a-m))) )

#Fitness funtion
def W(a,E):
    if E=='S':
        return((1-a)* (Psur(a)**gamma_S/transition_S))
    if E=='R':
        return((1-a)* (Psur(a)**gamma_R/transition_R))

#Possible values for a
alist=np.arange(0,1,0.01)

##Graphical check: return plot of P and W curves with current parameters
def Plot():
    max=1 #upper limit
    N=100 #number of values
    P=np.arange(N).astype(float)
    WR=np.arange(N).astype(float)
    WS=np.arange(N).astype(float)
    V=[1,1]
    FR=np.arange(N).astype(float)
    FS=np.arange(N).astype(float)
    for n in range(N):
        i=(n*max)/N
        P[n] = Psur(i)
        WS[n] = W(i,'S')/100
        WR[n] = W(i,'R')/100
        FR[n]= F(i,'S')
        FS[n]= F(i,'R')
    plt.plot(P,label='P')
    plt.plot(WS,label='WS')
    plt.plot(WR,label='WR')
    plt.plot(FR, label='FR')
    plt.plot(FS, label='FS')
    plt.legend()
    plt.show()

##Numerical solution
#Function that returns the fitness-maximizing a* for E=S and E=R
#It is numerically computable by finding the maximum of the fitness function (reached at the derivative's root)
#Note: for now we consider that either the safe envt has a 0 predator density (in which case the optimal vigilance level is 0), either both predator densities are non-zero.

A=[0,0]
def NumericOptimal():
    if gamma_S == 0:
        A[0]=0
        A[1]= (-gamma_R*lambertw(math.exp(-m*b+b-(1/gamma_R))/gamma_R)+b*gamma_R-1)/(b*gamma_R)
    else:
        A[0]= (-gamma_S*lambertw(math.exp(-m*b+b-(1/gamma_S))/gamma_S)+b*gamma_S-1)/(b*gamma_S)
        A[1]= (-gamma_R*lambertw(math.exp(-m*b+b-(1/gamma_R))/gamma_R)+b*gamma_R-1)/(b*gamma_R)
    print(A)


##Perfect Information
#fitness obtained through a given time step from performing a in environment E
def F(a,E):
    if E=='S':
        return((1-a) * (gamma_S*Psur(a) + (1-gamma_S)))
    if E=='R':
        return((1-a) * (gamma_R*Psur(a) + (1-gamma_R)))


# Reproductive value of an animal performing intensity a in environment E
def H(a,V,E):
    #adjust parameters depending on the value of E
    if E=='S':
        h = ((1-transition_S)*F(a,'S')*V[0])+(transition_S*F(a,'R')*V[1])
    else:
        h = ((1-transition_R)*F(a,'R')*V[1])+(transition_R*F(a,'S')*V[0])
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


# Find optimal strategy for Perfect Information
def PerfectInfo():
    #V = relative probability of survival for any long period under a optimal strategy
    #newVe are going to build a function sequence in order to obtain the optimal V(E) [ 0 = SAFE, 1 = RISKY ]
    V = [1,1]
    newV = [1,1]
    a_max = [0,0] #maximizing levels of vigilance

    j=0
    maxdiff=100
    #main loop
    while maxdiff>=0.001: #until the sequence converges
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

NumericOptimal()
PerfectInfo()




























