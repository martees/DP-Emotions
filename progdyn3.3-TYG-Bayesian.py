from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from time import process_time


##Parameters
s=100 #maximum reserve level
m=1 #metabolic cost at each step
d=0.002 #maximum probability of a predator attack
food1=5 #energy in food type 1 (both equiprobable)
food2=6 #energy in food type 2
c=2 #power of relationship btw f and mu(x,f)

#good conditions
food_G=0.7 #amount of food
transition_G=0.01 #proba of transitioning to bad

#bad conditions (same)
food_B=0.3
transition_B=0.01

#Number of possible values for estimate p that E=G
N = 100
#Number and range of possible values for foraging intensity f
M=1000
flist = np.arange(0,1,1/M)


##Risk of predation of an animal foraging at level f in state x
def mu(f):
    X = np.arange(0, s+1, 1) #x lies between 0 and s
    return(f*d*(0.5+(X/s)))

## Reproductive value of an animal performing intensity f in state (x,E)
#W returns a s+1 * N+1 * 2 table, having applied the H calculation on all the table at once
def W(P,f):

    #Posterior estimates
    psuccess=nextp[P,f,1]
    pfailure=nextp[P,f,0]
    #Column permutations associated to these estimates
    rows, column_indices = np.ogrid[:V.shape[0], :V.shape[1]] #we start with the V[.][psuccess/failure] part
    VSuccess = V[rows, np.reshape(psuccess, (1, N+1))] #changes the colu;ns to those given by psuccess
    VFailure = V[rows, np.reshape(pfailure, (1, N+1))]

    #T1=V[max(x-m,0)][p]   (for both E=good/0 and E=bad/1)
    T1 = np.roll(VFailure, m, axis=0) #we shift all the lines m times downwards
    T1[:m, :, :] = np.resize( np.tile(T1[m], m) , (m, N+1, 2)) #we fill in the gap on the top with m times the same line

    #T2=V[min(x+food1-m,s)][psuccess] (for both E=good/0 and E=bad/1)
    #x+food1-m =x - (m-food1)
    T2 = np.roll(VSuccess, m-food1, axis=0) #we shift all the lines food1-m times upwards
    T2[-(food1 - m):, :, :] = np.resize( np.tile(T2[-(food1 - m)-1], food1 - m) , (food1 - m, N+1, 2)) #we fill in the gap on down with food1-m times the same line

    #T3=V[min(x+food2-m,s)][psuccess] (for both E=good/0 and E=bad/1)
    T3 = np.roll(VSuccess, m-food2, axis=0)#we shift all the lines food2-m times upwards
    T3[-(food2 - m):, :, :] = np.resize( np.tile(T3[-(food2 - m)-1], food2 - m) , (food2 - m, N+1, 2)) #we fill in the gap on down with food2-m times the same line

    #MAIN CALCULATION
    H = np.zeros((s+1,N+1, 2))
    #case where E = good (E = 0, not E = 1)
    H[:,:,0] = (1-mu(f))*((1-food_G*f)*((1-transition_G)*T1[:,:,0] + transition_G*T1[:,:,1])+(food_G*f)*((1-transition_G)*(0.5*T2[:,:,0]+0.5*T3[:,:,0])+transition_G*(0.5*T2[:,:,1]+0.5*T3[:,:,1])))
    #case where E = bad (E = 1, not E = 0)
    H[:,:,1] = (1-mu(f))*((1-food_B*f)*((1-transition_B)*T1[:,:,1] + transition_B*T1[:,:,0])+(food_B*f)*((1-transition_B)*(0.5*T2[:,:,1]+0.5*T3[:,:,1])+transition_B*(0.5*T2[:,:,0]+0.5*T3[:,:,0])))

    return(H)

## Dynamic programming operator
def T():
    tmaxi=np.zeros((s+1, N+1)) #table that will keep track of the max t encountered for each cell
    Hmaxi=np.zeros((s+1, N+1, 2))  #table that will keep track of the correspnding H
    fmaxi=np.zeros((s+1, N+1)) #table that will keep track of the corresponding f

    #the loop that looks for the argmax (the maximum t and the associated f)
    for f in flist:

        #MAIN CALCULATION: reproductive value we want to maximize in each cell
        H = W(p,f)
        t = (p/N)*H[:,:,0]+(1-(p/N))*H[:,:,1] #0 for good, 1 for bad

        #updating the maximum tables
        comp = (t>tmaxi) #in each cell, if t > maxi (ie if we should update the max) we have a 1, otherwise we have a 0
        tmaxi = np.maximum(tmaxi, comp*t)  #comp*t : the cells that shouldn't be updated are 0, the others were just multtiplied by one
        Hmaxi[:,:,0] = np.maximum(Hmaxi[:,:,0], comp*H[:,:,0]) #slicing, might take long
        Hmaxi[:,:,1] = np.maximum(Hmaxi[:,:,1], comp*H[:,:,1])
        fmaxi = np.maximum(fmaxi, comp * f)  #comp*f : les cases qui ne doivent pas être update sont à zéro les autres valent f

    #if reserves are empty, death
    #on met à zéro la première ligne de fmaxi (cas x = 0)
    Hmaxi[0] = 0

    return(Hmaxi, fmaxi)


## Next estimate p: Bayesian.

#Updated prior permutation (depends only on p, so we only have to compute it once)
p_temp = np.arange(1, N, 1).astype(int)
p_temp = np.floor((transition_B/(N-p_temp) + (1-transition_G)/p_temp)).astype(int)
updated_prior = np.zeros(N+1)
updated_prior[0] = transition_B*N
updated_prior[1: -1] = p_temp
updated_prior = updated_prior.astype(int)

#Posterior estimate permutation (depends only on p and f, so we also compute it only once)
#create a table giving next estimate for each (p,f,result): p=LINE, f=COL, result=TUPLE
#function that fills in the table
def nextpfill(p,f,result):
    S1 = food_G*f
    S2 = food_B*f
    R1 = 1-S1
    R2 = 1-S2
    if result==1: #if success
        if f==0: #if division by zero occurs, no update (no info)
            newp = p
        else:
            newp = (S1*p)/( S2*(N-p)+S1*p )
    if result==0: #if failure
        newp = (R1*p)/( R2*(N-p)+R1*p )
    return(np.floor(N*newp).astype(int))

nextp = np.zeros((N,M,2))
for p in range(N):
    for f in flist:
        nextp[p,int(M*f),1]=nextpfill(p,f,1)
        nextp[p,int(M*f),0]=nextpfill(p,f,0)

nextp = nextp.astype(int)

## Find optimal strategy as the limit of a sequence of functions
#Here, functions are represented as tables associating a tuple (x,p) to the reproductive value associated with the optimal foraging intensity (relative probability of survival for any long period under a optimal strategy).

#p is the initial probability that E=G according to the animal
#p=transition_B/(transition_B+transition_G)

#Initial function = associates a foraging intensity of 1 with all possible (x,p) except for (0,p)=0
#x=LINE, p=COL, environment=TUPLE (0=good, 1=bad)
U = np.zeros( (s+1,N+1, 2) ) +1
U[0] = 0

#Convergence parameter (to compare to maximum acceptable difference between V and U)
maxdiff=100

#MAIN LOOP
for j in range(100):
#while maxdiff>=0.001: #until the sequence converges
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