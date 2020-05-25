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
def nextg(P_g,d,c,result):
    #idea for a bias
    #one = np.ones(int(N/2))
    #more = 2*np.ones(int(N/2)+1)
    #c = np.concatenate((one,more),axis=None)

    if result==1: #if encounter
        newPg = np.clip(P_g-d+c, 0,100).astype(int)
    if result==0: #if no encounter
        newPg = np.clip(P_g-d, 0,100).astype(int)
    return(newPg)


##Long-term reproductive value of an animal performing level a in environment E
#W returns a tuple with the reproductive value associated with level a for the given estimate p that conditions are safe, for both safe (H[0]) and risky (H[1]) environments
def W(P_g,a,V,d,c):
    #Gauge updates
    g_encounter=np.reshape(nextg(P_g,d,c,1), (-1,1))
    g_no_encounter=np.reshape(nextg(P_g,d,c,0), (-1,1))

    #Column permutations associated to these estimates
    #In order to switch from a time t's estimate to the time t+1, we operate on the whole (a,E) table, by swapping the rowss according to the nextp vector - **a row p's survival expectations are now a row nextp's**.

    #Getting the indices
    row_indices, column_indices = np.ogrid[:V.shape[0], :V.shape[1]]

    #What we would be doing without stochasticity
    #V_Enc = V[row_indices, g_encounter] #changing the rows to those given by g_encounter
    #V_NoEnc = V[row_indices, g_no_encounter]  #same for g_no_encounter

    #What we do with stochasticity (for convergence purposes)
        #changing the rows to those given by p_encounter
    V_Enc_Neg = V[np.clip(g_encounter-1, 0, L), column_indices]
    V_Enc_Pos = V[np.clip(g_encounter+1, 0, L), column_indices]
        #changing the rows to those given by p_no_encounter
    V_NoEnc_Neg = V[np.clip(g_no_encounter-1, 0, L), column_indices]
    V_NoEnc_Pos = V[np.clip(g_no_encounter+1, 0, L), column_indices]

    #The different equation terms, weighted by the predation probabilities
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

    #MAIN CALCULATION
    #integrates fitness pay-off and transition probabilities. H_Safe = the environment is effectively safe.
    H_Safe = G(1-a)* ( (1-transition_S)*(safe_encounter + safe_no_encounter) + transition_S*(risky_encounter + risky_no_encounter) )
    H_Risky = G(1-a)* ( (1-transition_R)* (risky_encounter + risky_no_encounter) + transition_R * (safe_encounter + safe_no_encounter))

    return(np.array([H_Safe,H_Risky]))


## Dynamic programming operator
def T(V,P_g,d,c):
    H=np.zeros((L+1, 2)) #table of H for all g's and E's
    t=np.zeros((L+1)) #table of t for all g's
    tmaxi=np.zeros((L+1)) #table that keeps track of the max t encountered for each cell
    Hmaxi=np.zeros((L+1, 2))  #table that keeps track of the correspnding H
    amaxi=np.zeros((L+1)) #table that keeps track of the corresponding a

    #Loop that looks for the argmax (the maximum t and the associated a)
    for a in alist:
        #MAIN CALCULATION: put the reproductive value we want to maximize in each cell of t
        H = W(P_g,a,V,d,c)
        t = (P_g/L)*H[0]+(1-(P_g/L))*H[1] #we divide the gauge by L because it contains the probability*L for easier indexation reasons

        #Simple version idea [probably doesn't work due to matrix shape inconsistencies]
        #for p in range(N+1):
        #    if t[0][p] > tmaxi[p]:#if the reproductive value associated with (p,a) is > tmax
        #        tmaxi[p] = t[0][p]
        #        Hmaxi[:, 0] = H[0] #we switch back to rows
        #        Hmaxi[:, 1] = H[1]
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
def Gauge(P_g,d,c):
    #Initial function = associates a level of antipredator behavior a of 1 with all possible p's and E's
    #a=LINE, environment=TUPLE (0=safe, 1=risky)
    newV = np.ones((L+1,2))

    #Convergence parameter (to compare to maximum acceptable difference between newV and newV)
    maxdiff=100

    #j=0
    #MAIN LOOP
    while maxdiff>=0.001: #until the sequence converges
        V=deepcopy(newV) #previous newV is stored in V
        newV = np.zeros( (L+1, 2) )

        #Process tracking (1)
        #j += 1 #for nice printing purposes
        #t =  process_time()
        #print("Iteration ", j, ", start time : ", t, sep='', end='')

        #MAIN CALCULATION: T2
        newV, A = T(V,P_g,d,c) #we recompute A each time (useful only on last iteration, could be worth some work)

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


##Computing the next step's Pop for the population convergence below
#Takes in the Pop matrix and operates the whole survival calculation on it
def nextPop(Pop,Opt):
    #Slow, cell by cell version
    #nextPop[g,E] = G(1-Opt[g])* ( (1-transition_E)*(gamma_E * Psur(Opt[g]) * Pop[g+d-c,E] + (1-gamma_E) * Pop[g+d,E] ) + transition_E*(gamma_notE * Psur(Opt[g]) * Pop[g+d-c,notE] + (1-gamma_E) * Pop[g+d,notE]))

    #Fast and furious version, rolls the whole table at once to align the appropriate cells
    #Pop_enc = [ Pop[g+d-c,0] , Pop[g+d-c,1] ], which is to say the Pop matrix rolled downwards from c-d, with all values Pop[i,:] with i superior to L-d+c being replaced by Pop[L-d+c]
    Pop_enc = np.roll(Pop,c-d,axis=0)
    Pop_enc[:c-d, :] = np.resize( np.tile(Pop_enc[c-d], c-d) , (c-d, 2))
    #Pop_no_enc = [ Pop[g+d,0] , Pop[g+d,1] ], which is to say the Pop matrix rolled upwards from d, with all values Pop[i,:] with i inferior to d-c being replaced by Pop[L-d]
    Pop_no_enc = np.roll(Pop,-d,axis=0)
    Pop_no_enc[-d:, :] = np.resize( np.tile(Pop_no_enc[L-d], d) , (d, 2))
    #Convert the functions so that they can be applied to the Opt vectors
    vec_G = np.vectorize(G)
    vec_P = np.vectorize(Psur)

    #Updated frequencies for safe (0) and risky (1) environments. Basically the same expression as H, since this is backward optimization again
    Pop[:,0] = vec_G(1-Opt) * ( (1-transition_S)*(gamma_S * vec_P(Opt) * Pop_enc[:,0] + (1-gamma_S) * Pop_no_enc[:,0] ) + transition_S*(gamma_R * vec_P(Opt) * Pop_enc[:,1] + (1-gamma_R) * Pop_no_enc[:,1] ))
    Pop[:,1] = vec_G(1-Opt) * ( (1-transition_R)*(gamma_R * vec_P(Opt) * Pop_enc[:,0] + (1-gamma_R) * Pop_no_enc[:,0] ) + transition_R*(gamma_S * vec_P(Opt) * Pop_enc[:,1] + (1-gamma_S) * Pop_no_enc[:,1] ))

    return(Pop)


##Convergence of the fear gauge
#g is the level of the fear gauge. It can take integer values between 1 and L. At each time step, it naturally decreases from d units. When the animal encounters a predator, it increases from c units.
#Using g, we build a probability table PG, which gives the animal the estimated probability that the environment is safe given that their fear is at level g. To find this PG table, we study, again, the convergence of a series of tables PG. Note that PG depends on the optimal strategy.
#To evaluate PG we thus perform forward iterations, to evaluate the survival probability of agents using a given PG table.

def FindGauge():
    #Initialization of the PG table with an equal chance of being in a safe environment whatever the gauge value
    newPG = np.ones(L+1)

    #We reinitialize PG in order to refill it with the new proportions
    #l = np.arange(0,L+1)
    #newPG = np.concatenate((l[:, np.newaxis], l[:, np.newaxis]), axis=1)
    #newPG = (l / np.sum(l))*(L+1)

    #Convergence parameter for PG
    maxdiff1=100

    j = 0
    #PG CONVERGENCE LOOP
    while maxdiff1>=0.000000001: #until the PG sequence converges
        #Process tracking (1)
        j += 1 #for nice printing purposes
        t =  process_time()
        print("Iteration ", j, ", start time : ", t, sep='', end='')

        PG=deepcopy(newPG) #previous newPG is stored in PG

        Opt = Gauge(PG,d,c) #we recompute the optimal strategy based on previous PG
        #Opt = np.reshape(Opt, (L+1,1))
        newPop = np.ones((L+1, 2))/(2*L) #we initialize the population

        #Convergence parameter for Pop
        maxdiff2=100

        print(Opt)
        print(newPop)
        while maxdiff2>=0.000001: #until the Pop sequence converges
            Pop=deepcopy(newPop) #previous newPop is stored in Pop
            newPop = np.ones((L+1, 2))/(2*L) #initialization with equal frequencies in every state/environment
            newPop = nextPop(newPop,Opt) #nextPop operates on newPop to compute the next time step
            newPop = newPop/np.sum(newPop) #normalization
            maxdiff2 = np.amax(np.abs(newPop-Pop)) #new maximum difference
        print(newPop)

        #We recompute PG accordingly. The probability of being in a safe environment knowing that g = k is taken to be the proportion of individuals in state (S,k).
        newPG = newPop/np.reshape(np.sum(newPop,axis=1),(L+1,1))
        newPG = newPG[:,0]
        newPG = newPG/np.sum(newPG) #normalization
        maxdiff1 = np.amax(np.abs(newPG-PG)) #new maximum difference

        #Process tracking (2)
        print(", iteration took ", process_time()-t, "s, maxdiff1 is : ", maxdiff1, sep = '')
        if j%10==0:
            print(V)

    return(newPG)



GAUGE = FindGauge()












