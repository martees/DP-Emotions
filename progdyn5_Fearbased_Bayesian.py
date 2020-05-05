from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw

#Executing the parameter file
exec(open("param.txt").read())

#p is the probability that E=S according to the animal. We want it to be updated in a bayesian way depending on whether the agent encounters a predator. There are two types of updates: the prior updates, which takes into account potential transitions happening during the actionm and the posterior updates, which take into account the encounters of the agent to increase or decrease p.

##Posterior estimate permutation
#(depends only on p, so we could also compute it only once, could be worth some work)
def nextp2(p,result):
    if result==1: #if encounter
        newp = (gamma_S*p) / ( gamma_S*p + gamma_R*(N-p) )
    if result==0: #if no encounter
        newp = ((1-gamma_S)*p)/( (1-gamma_S)*p + ((1-gamma_R)*(N-p)))
    return(np.floor(N*newp).astype(int))


##Long-term reproductive value of an animal performing level a in environment E
#W returns a tuple with the reproductive value associated with level a for the given estimate p that conditions are safe, for both safe (H[0]) and risky (H[1]) environments
def W2(p,a,V):
    #Prior update
    p = transition_R*(1-p) + (1-transition_S)*p

    #Posterior estimates
    p_encounter=nextp2(p,1)
    p_no_encounter=nextp2(p,0)

    safe_encounter1 = gamma_S * Psur(a) * V[max(0,p_encounter-1)][0]
    safe_encounter2 = gamma_S * Psur(a) * V[min(100, p_encounter+1)][0]
    safe_encounter = safe_encounter1+safe_encounter2

    safe_no_encounter1 = (1-gamma_S) * V[max(0,p_encounter-1)][0]
    safe_no_encounter2 = (1-gamma_S) * V[min(100, p_no_encounter+1)][0]
    safe_no_encounter = safe_no_encounter1 + safe_no_encounter2

    risky_encounter1 = gamma_R * Psur(a) * V[max(0,p_encounter-1)][1]
    risky_encounter2 = gamma_R * Psur(a) * V[min(100, p_encounter+1)][1]
    risky_encounter = risky_encounter1 + risky_encounter2

    risky_no_encounter1 = (1-gamma_R) * V[max(0,p_encounter-1)][1]
    risky_no_encounter2 = (1-gamma_R) * V[min(100, p_no_encounter+1)][1]
    risky_no_encounter = risky_no_encounter1 + risky_no_encounter2


    H_Safe = G(1-a)* ( (1-transition_S)*(safe_encounter + safe_no_encounter) + transition_S*(risky_encounter + risky_no_encounter) )

    H_Risky = G(1-a)* ( (1-transition_R)* (risky_encounter + risky_no_encounter) + transition_R * (safe_encounter + safe_no_encounter))

    return(np.array([H_Safe,H_Risky]))


## Dynamic programming operator
def T2(V):
    H=np.zeros((N+1, 2)) #table of H for all p's and E's
    t=np.zeros((N+1)) #table of t for all p's
    tmaxi=np.zeros((N+1)) #table that keeps track of the max t encountered for each cell
    Hmaxi=np.zeros((N+1, 2))  #table that keeps track of the correspnding H
    amaxi=np.zeros((N+1)) #table that keeps track of the corresponding a

    #Loop that looks for the argmax (the maximum t and the associated a)
    for a in alist:
        #MAIN CALCULATION: put the reproductive value we want to maximize in each cell of t
        for p in range(N+1):
            H[p] = W2(p,a,V)
            t[p] = (p/N)*H[p][0]+(1-(p/N))*H[p][1] #0 for good, 1 for bad
            if t[p] > tmaxi[p]:#if the reproductive value associated with (p,a) is > tmax
                tmaxi[p] = t[p]
                Hmaxi[p, 0] = H[p, 0]
                Hmaxi[p, 1] = H[p, 1]
                amaxi[p] = a

    return(Hmaxi, amaxi)


##Find optimal strategy as the limit of a sequence of functions
#Here, functions are represented as tables associating a tuple (x,p) to the reproductive value associated with the optimal level of antipredator behavior (relative probability of survival for any long period under a optimal strategy).
def Bayesian():
    #Initial function = associates a level of antipredator behavior a of 1 with all possible p's and E's
    #a=LINE, environment=TUPLE (0=safe, 1=risky)
    newV = np.ones((N+1,2))

    #Convergence parameter (to compare to maximum acceptable difference between newV and newV)
    maxdiff=100

    j=0
    #MAIN LOOP
    while maxdiff>=0.001: #until the sequence converges
        V=deepcopy(newV) #previous newV is stored in V
        newV = np.zeros( (N+1, 2) )

        #Process tracking (1)
        j += 1 #for nice printing purposes
        t =  process_time()
        print("Iteration ", j, ", start time : ", t, sep='', end='')

#        #Application of the updated prior permutation to the p columns
#        rows, column_indices = np.ogrid[:V.shape[0], :V.shape[1]]
#        V = V[np.reshape(updated_prior, (N+1,1)), column_indices]

        #MAIN CALCULATION: T2
        newV, A = T2(V) #we recompute A each time (useful only on last iteration, could be worth some work)

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
