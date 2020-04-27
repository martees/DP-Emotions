from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math


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

#List of possible values for foraging intensity f
flist = np.arange(0,1,0.01)

##OptF Matrix: optimal foraging intensity for every given x,E : x lines, |E| columns (here = 2)
OptF=[]
for food in range(s+1):
    OptF.append([-1,-1])

##Risk of predation of an animal foraging at level f in state x
def mu(x,f):
    return((f**c)*d*(0.5+(x/s)))

## Reproductive value of an animal performing intensity f in state (x,E)
def H(x,E,f):
    #adjust parameters depending on the value of E
    if E=='G':
        F=food_G
        TR=transition_G
        E=0 #good=0 in the V matrix
        notE=1
    else:
        F=food_B
        TR=transition_B
        E=1
        notE=0

    #compute the
    UM=1-mu(x,f)
    T1=V[max(x-m,0)][E]
    T2=V[max(x-m,0)][notE]
    T3=V[min(x+food1-m,s)][E]
    T4=V[min(x+food2-m,s)][E]
    T5=V[min(x+food1-m,s)][notE]
    T6=V[min(x+food2-m,s)][notE]

    h = (UM)*((1-F*f)*((1-TR)*T1 + TR*T2)+(F*f)*((1-TR)*(0.5*T3+0.5*T4)+TR*(0.5*T5+0.5*T6)))
    return(h)

## Dynamic programming operator (finds optimal foraging strategy for each x,E and fills up OptF)
def T(x,E):
    if x==0: #if reserves are empty, death
        return(0)
    else: #else look for the best strategy
        maxi=0 #maximum H
        maxf=0 #optimal f
        for f in flist: #for many f's
            t = H(x,E,f) #we calculate the reproductive value if foraging at f in (x,E)
            if t>=maxi: #if a better t is found, update maxima
                maxi=t
                maxf=f
        if E=='G': #0=good in the OptF matrix
            E=0
        else:
            E=1
        OptF[x][E]=maxf
        return(maxi)


## Find optimal strategy
#V = relative probability of survival for any long period under a optimal strategy
#We are going to build a function sequence in order to obtain the optimal V(x,E)

V = [[0,0]]
W = [[0,0]]

#First function
for f in range(s+1):
    W.append([1,1])

maxdiff=100

while maxdiff>=0.0001: #until the sequence converges
    V=deepcopy(W) #n is stored in V
    W=[]
    maxi=0 #max value of T, that will be used to normalize W
    for food in range(s+1):
        W.append([T(food,'G'),T(food,'B')])
        if T(food,'G') > maxi:
            maxi = T(food,'G')
        if T(food,'B') > maxi:
            maxi = T(food,'B')
    for food in range(s+1):
        for i in range(2):
            W[food][i]=W[food][i]/maxi
    maxdiff=W[0][0]-V[0][0]
    for food in range(s+1):
        for i in range(2):
            if abs(W[food][i]-V[food][i])>maxdiff:
                maxdiff=abs(W[food][i]-V[food][i])

del(OptF[0])
plt.plot(OptF)
plt.show()









