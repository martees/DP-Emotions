from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw

#Executing the parameter file
exec(open("param.txt").read())

##Posterior estimate permutation
#(depends only on p, so we could also compute it only once, could be worth some work)
def nextg(g,d,c,result):
    if g == 0:
        return(0)
    if result==1: #if encounter
        newg = max(0,min(100,g-d+c))
    if result==0: #if no encounter
        newg = max(0,min(100,g-d))
    return(newg)


##Long-term reproductive value of an animal performing level a in environment E
#W returns a tuple with the reproductive value associated with level a for the given estimate p that conditions are safe, for both safe (H[0]) and risky (H[1]) environments
def W3(g,a,V,d,c):
    #Gauge updates
    g_encounter=nextg(g,d,c,1)
    g_no_encounter=nextg(g,d,c,0)

    safe_encounter1 = gamma_S * Psur(a) * V[max(0,g_encounter-1)][0]
    safe_encounter2 = gamma_S * Psur(a) * V[min(L, g_encounter+1)][0]
    safe_encounter = safe_encounter1+safe_encounter2

    safe_no_encounter1 = (1-gamma_S) * V[max(0,g_encounter-1)][0]
    safe_no_encounter2 = (1-gamma_S) * V[min(L, g_no_encounter+1)][0]
    safe_no_encounter = safe_no_encounter1 + safe_no_encounter2

    risky_encounter1 = gamma_R * Psur(a) * V[max(0,g_encounter-1)][1]
    risky_encounter2 = gamma_R * Psur(a) * V[min(L, g_encounter+1)][1]
    risky_encounter = risky_encounter1 + risky_encounter2

    risky_no_encounter1 = (1-gamma_R) * V[max(0,g_encounter-1)][1]
    risky_no_encounter2 = (1-gamma_R) * V[min(L, g_no_encounter+1)][1]
    risky_no_encounter = risky_no_encounter1 + risky_no_encounter2


    H_Safe = G(1-a)* ( (1-transition_S)*(safe_encounter + safe_no_encounter) + transition_S*(risky_encounter + risky_no_encounter) )

    H_Risky = G(1-a)* ( (1-transition_R)* (risky_encounter + risky_no_encounter) + transition_R * (safe_encounter + safe_no_encounter))

    return(np.array([H_Safe,H_Risky]))


## Dynamic programming operator
def T3(V,p_g,d,c):
    H=np.zeros((L+1, 2)) #table of H for all g's and E's
    t=np.zeros((L+1)) #table of t for all g's
    tmaxi=np.zeros((L+1)) #table that keeps track of the max t encountered for each cell
    Hmaxi=np.zeros((L+1, 2))  #table that keeps track of the correspnding H
    amaxi=np.zeros((L+1)) #table that keeps track of the corresponding a

    #Loop that looks for the argmax (the maximum t and the associated a)
    for a in alist:
        #MAIN CALCULATION: put the reproductive value we want to maximize in each cell of t
        for g in range(L+1):
            H[g] = W3(g,a,V,d,c)
            #p_g gives the probability that E=S knowing that fear is at level g
            t[g]=p_g[g]*H[g][0]+(1-p_g[g])*H[g][0]
            if t[g] > tmaxi[g]:#if the reproductive value associated with (p,a) is > tmax
                tmaxi[g] = t[g]
                Hmaxi[g, 0] = H[g, 0]
                Hmaxi[g, 1] = H[g, 1]
                amaxi[g] = a

    return(Hmaxi, amaxi)


##Find optimal strategy as the limit of a sequence of functions
#Here, functions are represented as tables associating a tuple (x,p) to the reproductive value associated with the optimal level of antipredator behavior (relative probability of survival for any long period under a optimal strategy).
def Gauge(p_g,d,c):
    #Initial function = associates a level of antipredator behavior a of 1 with all possible p's and E's
    #a=LINE, environment=TUPLE (0=safe, 1=risky)
    newV = np.ones((L+1,2))

    #Convergence parameter (to compare to maximum acceptable difference between newV and newV)
    maxdiff=100

    j=0
    #MAIN LOOP
    while maxdiff>=0.001: #until the sequence converges
        V=deepcopy(newV) #previous newV is stored in V
        newV = np.zeros( (L+1, 2) )

        #Process tracking (1)
        j += 1 #for nice printing purposes
        t =  process_time()
        print("Iteration ", j, ", start time : ", t, sep='', end='')

        make.gauge() #<<< for testing purposes, NOT FINAL

        #MAIN CALCULATION: T2
        newV, A = T3(V,p_g,d,c) #we recompute A each time (useful only on last iteration, could be worth some work)

        #Normalization
        maxi = np.amax(newV)
        if maxi != 0:
            newV = newV/maxi

        #recompute maximum difference btw V and newV for convergence
        maxdiff= np.amax(np.abs(newV - V))

        #Process tracking (2)
        print(", iteration took ", process_time()-t, "s, maxdiff is : ", maxdiff, sep = '')
#        if j%10==0:
#            print(V)
    return(A, V)


##Convergence of the fear gauge
#g is the level of the fear gauge.
#It can take integer values between 1 and L. At each time step, it naturally decreases from d units. When the animal encounters a predator, it increases from c units.
#using g, we build a probability table p_g, which gives the animal the estimated probability that the environment is safe given that their fear is at level g.

def makeGauge():
    p_g=np.ones((L+1,2))
    p_g=p_g/(2*L)
    return(p_g)














