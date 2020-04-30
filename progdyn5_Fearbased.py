from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw
from progdyn5_Annex import *

##Perfect Information

#Total fitness obtained through a given time step from performing a in environment E
def F(a,E):
    if E=='S':
        gamma = gamma_S
    if E=='R':
        gamma = gamma_R
    return( G(1-a) * ( gamma * Psur(a) + (1-gamma)))


#Long-term reproductive value of an animal performing intensity a in environment E
def H1(a,V,E):
    #adjust parameters depending on the value of E
    if E=='S':
        h = ((1-transition_S)*F(a,'S')*V[0])+(transition_S*F(a,'R')*V[1])
    else:
        h = ((1-transition_R)*F(a,'R')*V[1])+(transition_R*F(a,'S')*V[0])
    return(h)


# Dynamic programming operator (finds the a that maximizes V for each E, and fills up V and A with the corresponding max)
def T1(V,E):
    maxi=0 #maximum H
    maxa=0 #optimal a
    for a in alist: #for many a's
        t = H1(a,V,E) #calculate the reproductive value if foraging at a in E
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
    print(newV,a_max)


#   plt.imshow(a_max) #plotting level of antipredator behavior matrix F
#   plt.gca().invert_yaxis()
#   plt.pause(0.1)
#
#   plt.colorbar(aspect='auto')
#   plt.show()


##Bayesian
#p is the probability that E=S according to the animal. We want it to be updated in a bayesian way depending on whether the agent encounters a predator. There are two types of updates: the prior updates, which takes into account potential transitions happening during the actionm and the posterior updates, which take into account the encounters of the agent to increase or decrease p.
#note: for now no prior update

#Posterior estimate permutation (depends only on p, so we could also compute it only once, could be worth some work)
def nextp(p,result):
    if result==1: #if encounter
        newp = (gamma_S*p) / ( gamma_S*p + gamma_R*(N-p) )
    if result==0: #if no encounter
        newp = ((1-gamma_S)*p)/( (1-gamma_S)*p + ((1-gamma_R)*(N-p)))
    return(np.floor(N*newp).astype(int))


#Long-term reproductive value of an animal performing level a in environment E
#W returns a tuple with the reproductive value associated with level a for the given estimate p that conditions are safe, for both safe (H[0]) and risky (H[1]) environments
def W(p,a,V):
    #Posterior estimates
    p_encounter=nextp(p,1)
    p_no_encounter=nextp(p,0)

    safe_encounter = gamma_S * Psur(a)* V[p_encounter][0]
    safe_no_encounter = (1-gamma_S)*V[p_no_encounter][0]
    risky_encounter = gamma_R * Psur(a)* V[p_encounter][1]
    risky_no_encounter = (1-gamma_R)*V[p_no_encounter][1]

    H_Safe = G(a)* ( (1-transition_S)*(safe_encounter + safe_no_encounter) + transition_S*(risky_encounter + risky_no_encounter) )

    H_Risky = G(a)* ( (1-transition_R)* (risky_encounter + risky_no_encounter) + transition_R * (safe_encounter + safe_no_encounter))

    return(np.array([H_Safe,H_Risky]))


# Dynamic programming operator
def T2(V):
    H=np.zeros((N+1, 2)) #table of H for all p's and E's
    t=np.zeros((N+1)) #table of t for all p's
    tmaxi=np.zeros((N+1)) #table that keeps track of the max t encountered for each cell
    Hmaxi=np.zeros((N+1, 2))  #table that keeps track of the correspnding H
    amaxi=np.zeros((N+1)) #table that keeps track of the corresponding a

    #Loop that looks for the argmax (the maximum t and the associated a)
    for a in alist:

        #MAIN CALCULATION: put the reproductive value we want to maximize in each cell
        for p in range(N+1):
#            p = np.floor((transition_R*(N-p) + (1-transition_S)*p)).astype(int) #updated prior
            H[p] = W(p,a,V)
            t[p] = (p/N)*H[p][0]+(1-(p/N))*H[p][1] #0 for good, 1 for bad
            if t[p] > tmaxi[p]:
                tmaxi[p] = t[p]
                Hmaxi[p, 0] = H[p, 0]
                Hmaxi[p, 1] = H[p, 1]
                amaxi[p] = a

    return(Hmaxi, amaxi)


#Find optimal strategy as the limit of a sequence of functions
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


##Imperative commands

#Plot(F1) #graphical check: return plot of P and W curves with current parameters
#NumericOptimal() #numerical solution of the corresponding Nonacs model

PerfectInfo()
A, V = Bayesian()




























