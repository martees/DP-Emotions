from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw

#Executing the parameter file
exec(open("param.txt").read())

#p is the probability that E=S according to the animal. We want it to be updated in a bayesian way depending on whether the agent encounters a predator. There are two types of updates: the prior updates, which takes into account potential transitions happening during the actionm and the posterior updates, which take into account the encounters of the agent to increase or decrease p."

##Prior update permutation
p_all = np.arange(0, N+1, 1).astype(int) #a list of all possible p's
#we apply the prior to all possible p's, giving us the first step of the permutation to apply to the V matrix when the p will have to be updated
updated_prior = np.floor(transition_R*(N-p_all) + (1-transition_S)*p_all).astype(int)

##Posterior estimate permutation
#(depends only on p, so we could also compute it only once, could be worth some work)
def nextp(p,encounter):
    "Takes in a list of p's and an event and returns the updated p list."
    if encounter==1: #if encounter
        newp = (gamma_S*p) / ( gamma_S*p + gamma_R*(N-p) )
    if encounter==0: #if no encounter
        newp = ((1-gamma_S)*p)/( (1-gamma_S)*p + ((1-gamma_R)*(N-p)))
    return(np.ceil((N*newp)).astype(int))


##Long-term reproductive value of an animal performing level a in environment E
#W returns a tuple with the reproductive value associated with level a for the given estimate p that conditions are safe, for both safe (H[0]) and risky (H[1]) environments
def W2(a,V):
    #Prior update
    p = updated_prior

    #Posterior estimates
    p_encounter=np.reshape(nextp(p,1), (-1,1))
    p_no_encounter=np.reshape(nextp(p,0), (-1,1))

    #Column permutations associated to these estimates
    #In order to switch from a time t's estimate to the time t+1, we operate on the whole (a,E) table, by swapping the rowss according to the nextp vector - **a row p's survival expectations are now a row nextp's**.

    #getting the indices
    row_indices, column_indices = np.ogrid[:V.shape[0], :V.shape[1]]

    #What we would be doing without stochasticity
    #V_Enc = V[row_indices, p_encounter] #changing the rows to those given by p_encounter
    #V_NoEnc = V[row_indices, p_no_encounter]  #same for p_no_encounter

    #What we do with stochasticity (for convergence purposes)
        #changing the rows to those given by p_encounter
    V_Enc_Neg = V[np.clip(p_encounter-1, 0, N), column_indices]
    V_Enc_Pos = V[np.clip(p_encounter+1, 0, N), column_indices]
        #changing the rows to those given by p_no_encounter
    V_NoEnc_Neg = V[np.clip(p_no_encounter-1, 0, N), column_indices]
    V_NoEnc_Pos = V[np.clip(p_no_encounter+1, 0, N), column_indices]


    safe_encounter1 = gamma_S * Psur(a) * V_Enc_Pos[:,0]
    safe_encounter2 = gamma_S * Psur(a) * V_Enc_Neg[:,0]
    safe_encounter = 1/2*safe_encounter1 + 1/2*safe_encounter2

    safe_no_encounter1 = (1-gamma_S) * V_NoEnc_Pos[:,0]
    safe_no_encounter2 = (1-gamma_S) * V_NoEnc_Neg[:,0]
    safe_no_encounter = 1/2*safe_no_encounter1 + 1/2*safe_no_encounter2

    risky_encounter1 = gamma_R * Psur(a) * V_Enc_Pos[:,1]
    risky_encounter2 = gamma_R * Psur(a) * V_Enc_Neg[:,1]
    risky_encounter = 1/2*risky_encounter1 + 1/2*risky_encounter2

    risky_no_encounter1 = (1-gamma_R) * V_NoEnc_Pos[:,1]
    risky_no_encounter2 = (1-gamma_R) * V_NoEnc_Pos[:,1]
    risky_no_encounter = 1/2*risky_no_encounter1 + 1/2*risky_no_encounter2


    H_Safe = G(1-a)* ( (1-transition_S)*(safe_encounter + safe_no_encounter) + transition_S*(risky_encounter + risky_no_encounter) )
    #H_Safe = np.reshape(H_Safe, (-1, 1)) #reshape it into a one column vector

    H_Risky = G(1-a)* ( (1-transition_R)* (risky_encounter + risky_no_encounter) + transition_R * (safe_encounter + safe_no_encounter))
    #H_Risky = np.reshape(H_Risky, (-1, 1)) #reshape it into a one column vector

    return(H_Safe, H_Risky)


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
        H = W2(a,V)
        t = (p_all/N)*H[0]+(1-(p_all/N))*H[1] #we divide the probability p by N because it contains the probability*N for easier indexation reasons

        #Simple version [DOES NOT WORK NEEDS CORRECTION]
        #for p in range(N+1):
        #    if t[0][p] > tmaxi[p]:#if the reproductive value associated with (p,a) is > tmax
        #        tmaxi[p] = t[0][p]
        #        Hmaxi[:, 0] = np.reshape(H[0],(1,-1)) #we switch back to rows
        #        Hmaxi[:, 1] = np.reshape(H[1],(1,-1))
        #        amaxi[p] = a

        #Fast and furious version
        #We create two tables: old and new. 'new' multiplied by a table will "select" (keep them, while all other values are put to 0) all the values we want to replace, and 'old' multiplied by a table will select all the old values that we want to keep.
        new = (t>tmaxi) #in each cell, if t > maxi (ie if we should update the max) we have a 1, otherwise we have a 0
        old = 1-new #old is just the complementary of new
        tmaxi = tmaxi*old + t*new  #tmaxi*old = the values that should be kept, and t*new = the values that should be updated. The cells that shouldn't be updated are now 0, and the others were just multiplied by one
        #same operation for both E=R and E=S on H
        Hmaxi[:,0] = Hmaxi[:,0]*old + H[0]*new
        Hmaxi[:,1] = Hmaxi[:,1]*old + H[1]*new
        amaxi = amaxi*old+a*new  #fmaxi*old keeps the old values of f that we do not want to update, and f*new updates the new values

    return(Hmaxi, amaxi)


##Find optimal strategy as the limit of a sequence of functions
#Here, functions are represented as tables associating a tuple (x,p) to the reproductive value associated with the optimal level of antipredator behavior (relative probability of survival for any long period under a optimal strategy).
def Bayesian():

    exec(open("param.txt").read(),globals()) #executing parameter file

    #Initial function = associates a level of antipredator behavior a of 1 with all possible p's and E's
    #a=LINE, environment=TUPLE (0=safe, 1=risky)
    newV = np.ones((N+1,2))

    #Convergence parameter (to compare to maximum acceptable difference between newV and newV)
    maxdiff=100

    j=0
    #MAIN LOOP
    while maxdiff>=0.001: #until the sequence converges
        V=deepcopy(newV) #previous newV is stored in V
        newV = np.zeros((N+1, 2))

        #Process tracking (1)
        #j += 1 #for nice printing purposes
        #t =  process_time()
        #print("Iteration ", j, ", start time : ", t, sep='', end='')

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
        #print(", iteration took ", process_time()-t, "s, maxdiff is : ", maxdiff, sep = '')
#        if j%10==0:
#            print(V)
    return(A)

#A = Bayesian()
#plt.plot(A)
#plt.show()