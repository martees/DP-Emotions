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
    if result==1: #if encounter
        newg = np.clip(P_g-d+c, 0,100).astype(int)
    if result==0: #if no encounter
        newg = np.clip(P_g-d, 0,100).astype(int)
    return(newg)


##Long-term reproductive value of an animal performing level a in environment E
#W returns a tuple with the reproductive value associated with level a for the given estimate p that conditions are safe, for both safe (H[0]) and risky (H[1]) environments
def W3(P_g,a,V,d,c):
    #Gauge updates
    g_encounter=np.reshape(nextg(P_g,d,c,1), (-1,1))
    g_no_encounter=np.reshape(nextg(P_g,d,c,0), (-1,1))

    #Column permutations associated to these estimates
    #In order to switch from a time t's estimate to the time t+1, we operate on the whole (a,E) table, by swapping the rowss according to the nextp vector - **a row p's survival expectations are now a row nextp's**.

    #getting the indices
    row_indices, column_indices = np.ogrid[:V.shape[0], :V.shape[1]]

    #What we would be doing without stochasticity
    #V_Enc = V[row_indices, p_encounter] #changing the rows to those given by p_encounter
    #V_NoEnc = V[row_indices, p_no_encounter]  #same for p_no_encounter

    #What we do with stochasticity (for convergence purposes)
        #changing the rows to those given by p_encounter
    V_Enc_Neg = V[np.clip(g_encounter-1, 0, L), column_indices]
    V_Enc_Pos = V[np.clip(g_encounter+1, 0, L), column_indices]
        #changing the rows to those given by p_no_encounter
    V_NoEnc_Neg = V[np.clip(g_no_encounter-1, 0, L), column_indices]
    V_NoEnc_Pos = V[np.clip(g_no_encounter+1, 0, L), column_indices]


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

    H_Risky = G(1-a)* ( (1-transition_R)* (risky_encounter + risky_no_encounter) + transition_R * (safe_encounter + safe_no_encounter))

    return(np.array([H_Safe,H_Risky]))


## Dynamic programming operator
def T3(V,P_g,d,c):
    H=np.zeros((L+1, 2)) #table of H for all g's and E's
    t=np.zeros((L+1)) #table of t for all g's
    tmaxi=np.zeros((L+1)) #table that keeps track of the max t encountered for each cell
    Hmaxi=np.zeros((L+1, 2))  #table that keeps track of the correspnding H
    amaxi=np.zeros((L+1)) #table that keeps track of the corresponding a

    #Loop that looks for the argmax (the maximum t and the associated a)
    for a in alist:
        #MAIN CALCULATION: put the reproductive value we want to maximize in each cell of t
        H = W3(P_g,a,V,d,c)
        t = (P_g/L)*H[0]+(1-(P_g/L))*H[1] #we divide the gauge by L because it contains the probability*L for easier indexation reasons

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
        newV, A = T3(V,P_g,d,c) #we recompute A each time (useful only on last iteration, could be worth some work)

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


##Computing the next step's N for the population convergence below
#Takes in the N matrix and operates the whole survival calculation on it
def nextN(N,Opt):
    #Non-matrix version
    #nextN[g,E] = G(1-Opt[g])* ( (1-transition_E)*(gamma_E * Psur(Opt[g]) * N[g+d-c,E] + (1-gamma_E) * N[g+d,E] ) + transition_E*(gamma_notE * Psur(Opt[g]) * N[g+d-c,notE] + (1-gamma_E) * N[g+d,notE]))

    #N_enc = [ N[g+d-c,0] , N[g+d-c,1] ], which is to say the N matrix rolled downwards from c-d, with all values N[i,:] with i superior to L-d+c being replaced by N[L-d+c]
    N_enc = np.roll(N,c-d,axis=0)
    N_enc[:c-d, :] = np.resize( np.tile(N_enc[c-d], c-d) , (c-d, 2))

    #N_no_enc = [ N[g+d,0] , N[g+d,1] ], which is to say the N matrix rolled upwards from d, with all values N[i,:] with i inferior to d-c being replaced by N[L-d]
    N_no_enc = np.roll(N,-d,axis=0)
    N_no_enc[-d:, :] = np.resize( np.tile(N_no_enc[L-d], d) , (d, 2))

    vec_G = np.vectorize(G)
    vec_P = np.vectorize(Psur)

    N[:,0] = vec_G(1-Opt) * ( (1-transition_S)*(gamma_S * vec_P(Opt) * N_enc[:,0] + (1-gamma_S) * N_no_enc[:,0] ) + transition_S*(gamma_R * vec_P(Opt) * N_enc[:,1] + (1-gamma_R) * N_no_enc[:,1] ))
    N[:,1] = vec_G(1-Opt) * ( (1-transition_R)*(gamma_R * vec_P(Opt) * N_enc[:,0] + (1-gamma_R) * N_no_enc[:,0] ) + transition_R*(gamma_S * vec_P(Opt) * N_enc[:,1] + (1-gamma_S) * N_no_enc[:,1] ))

    return(N)


##Convergence of the fear gauge
#g is the level of the fear gauge. It can take integer values between 1 and L. At each time step, it naturally decreases from d units. When the animal encounters a predator, it increases from c units.
#Using g, we build a probability table P_g, which gives the animal the estimated probability that the environment is safe given that their fear is at level g. To find this P_g table, we study, again, the convergence of a series of tables PG. Note that PG depends on the optimal strategy.
#To evaluate PG we use population simulation.

def FindGauge():
    #Initialization of the PG table with an equal chance of being in a safe environment whatever the gauge value
    newPG = np.ones(L+1)/L

    #Convergence parameter for PG
    maxdiff1=100

    j = 0
    #PG CONVERGENCE LOOP
    while maxdiff1>=0.000001: #until the PG sequence converges
        #Process tracking (1)
        j += 1 #for nice printing purposes
        t =  process_time()
        print("Iteration ", j, ", start time : ", t, sep='', end='')

        PG=deepcopy(newPG) #previous newPG is stored in PG
        #We reinitialize PG in order to refill it with the new proportions
        l = np.arange(0,L)
        newPG = np.concatenate((l[:, np.newaxis], l[:, np.newaxis]), axis=1)
        newPG = newPG / np.sum(newPG)

        Opt = Gauge(PG,d,c) #we recompute the optimal strategy based on previous PG
        newN = np.ones((L+1, 2))/(2*L) #we initialize the population

        #Convergence parameter for N
        maxdiff2=100

        while maxdiff2>=0.000001: #until the N sequence converges
            N=deepcopy(newN) #previous newN is stored in N
            newN = np.ones((L+1, 2))/(2*L) #initialization with equal frequencies in every state/environment
            newN = nextN(newN,Opt) #nextN operates on newN to compute the next time step
            newN = newN/np.sum(newN) #normalization
            maxdiff2 = np.amax(np.abs(newN-N)) #new maximum difference

        #We recompute PG accordingly. The probability of being in a safe environment knowing that g = k is taken to be the proportion of individuals in state (S,k).
        newPG = newN/np.reshape(np.sum(newN,axis=1),(L+1,1))
        newPG = newPG[:,0]
        newPG = newPG/np.sum(newPG) #normalization
        maxdiff1 = np.amax(np.abs(newPG-PG)) #new maximum difference


        #Process tracking (2)
        print(", iteration took ", process_time()-t, "s, maxdiff1 is : ", maxdiff1, sep = '')
        if j%10==0:
            print(V)

    return(newPG)



#GAUGE = FindGauge()












