from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw

##Parameters

#General
true=0.1 #density of true predation cues left by predators
noise=0.01 #density of false predation cues in any environment
c=8

#Safe environment
gamma_S=0.01 #density of predators
transition_S=0.01

#Risky environment
gamma_R=0.5 #density of predators
transition_R=0.01


#Survival function (probability of surviving a predator attack with level of antipredator behavior = a)
m1=0.5 #inflexion point
b1=10 #steepness of growth
def Psur(a):
    return( 1/(1+exp(-b1*(a-m1))) )

#Payoff function (expected immediate fitness gain from level of antipredator behavior = a)
m2=0.2
b2=20
def G(a):
    return( 1/(1+exp(-b2*(a-m2))) )

#Possible values for a
Na =100 #number of possible values
alist=np.arange(0,1,1/Na) #corresponding interval
#Number of possible values for estimate p that E=S
N = 1000

##Graphical check: return plot of P, G and W curves with current parameters
def Plot(V):
    max=1 #upper limit
    N=1000 #number of values

    P=np.arange(Na).astype(float)
    Gain=np.arange(Na).astype(float)
    W=np.arange(Na).astype(float)

    for n in range(Na):
        i=(n*max)/Na

        P[n] = Psur(i)
        Gain[n] = G(1-i)
        W[n]= f3(0.5,i,V)[0]

    plt.plot(P,label='Psurvival')
    plt.plot(Gain,label='G')
    plt.plot(W,label='W, with p=0.5 and E=S')
    plt.legend()
    plt.show()

##Numerical solution of the corresponding Nonacs model
#Function that returns the fitness-maximizing a* for E=S and E=R
#It is numerically computable by finding the maximum of the fitness function (reached at the derivative's root)
#Note: for now we consider that either the safe envt has a 0 predator density (in which case the optimal vigilance level is 0), either both predator densities are non-zero.

def NumericOptimal():
    A=[0,0]
    if gamma_S == 0:
        A[0]=0
        A[1]= (-gamma_R*lambertw(exp(-m1*b1+b1-(1/gamma_R))/gamma_R)+b1*gamma_R-1)/(b*gamma_R)
    else:
        A[0]= (-gamma_S*lambertw(exp(-m1*b1+b1-(1/gamma_S))/gamma_S)+b1*gamma_S-1)/(b1*gamma_S)
        A[1]= (-gamma_R*lambertw(exp(-m1*b1+b1-(1/gamma_R))/gamma_R)+b1*gamma_R-1)/(b1*gamma_R)
    print(A)


