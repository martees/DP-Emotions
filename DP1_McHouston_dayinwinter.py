from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


#Variables:
#   t the current time
#   T final time
#   x the current state (at time t)
#   L maximum state
#   Actions matrix with at line i:
#       first column: the probability to die performing action u_i
#       secund column: the probability of finding 0 food performing u_i
#       third column: the probability of finding 1 food performing u_i
#       ...
#       fth column: the probability of finding f food performing u_i

Actions=[[0,0.5,0.125,0.25,0.125],[0.01,0.4,0.15,0.3,0.15]]
N=len(Actions)
L=20
T=200
Matrix=[]

## Matrix L lignes, T colonnes, N-uplets de -1
for l in range(L+1):
    Matrix.append([])
    for c in range(T):
        Matrix[l].append([])
        for i in range(N):
            Matrix[l][c].append(-1)

## Reproductive value of an animal that survived until time T with resources x
def R1(x):
    if x>=10:
        return(1)
    else:
        return(0)

def R2(x):
    if x>=1:
        return(1)
    else:
        return(0)

## Reproductive value of an animal performing action u at time t while in state x
def H(x,t,act):
    h=0
    u=Actions[act]
    for food in range(1,len(u)):
        newstate=min(x+food-2,L)
        h += (1-u[0])*(u[food]*V(newstate,t+1))
    Matrix[x][t][act]=h
    return(h)

## Reproductive value of an animal following the optimal strategy from t (where they are in state x) to T
def V(x,t):
    if x==0: #if reserves are empty, death
        return(0)
    elif t==T: #if time is over, see Reward function
        return(R1(x))
    else:
        max=0
        bestAction=-1
        for act in range(N):
            if Matrix[x][t][act]!=-1:
                h=Matrix[x][t][act]
            else:
                h = H(x,t,act)
            if h>max:
                max=h
        return(max)

## Takes Matrix as an argument and returns the best action for each state/time + color plots it
def BestAction(Matrix):
    OptAct=deepcopy(Matrix)
    for l in range(L+1):
        for c in range(T):
            maxi=max(OptAct[l][c])
            for a in range(N):
                if Matrix[l][c][a]==maxi:
                    OptAct[l][c]=a
    plt.imshow(OptAct)
    plt.colorbar(aspect='auto')
    plt.gca().invert_yaxis()
    plt.show()
    return(OptAct)



















