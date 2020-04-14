from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from time import process_time
from scipy.special import lambertw

##Parameters
#General
true=0.1 #density of true predation cues left by predators
noise=0.01 #density of false predation cues in any environment

#Safe environment
d_pred_S=0
d_cues_S=noise

#Risky environment
d_pred_R=100
d_cues_R=noise+true*d_pred_R

#Survival function
m=0.5 #inflexion point
b=20 #steepness of growth
def P_s(V):
    return( 1/(1+math.exp(-b*(V-m))) )

#Fitness funtion
def W(V,E):
    if E=='S':
        return((1-V)* (P_s(V)**d_pred_S) )
    if E=='R':
        return((1-V)* (P_s(V)**d_pred_R) )

#Possible values for V
Vlist=np


##Perfect info
#Function that returns the fitness-maximizing V* for E=S and E=R
A=[0,0]

def PerfectInfo():
    if d_pred_S == 0:
        A[0]=0
        A[1]= (-d_pred_R*lambertw(math.exp(-m*b+b-(1/d_pred_R))/d_pred_R)+b*d_pred_R-1)/(b*d_pred_R)
    else:
        A[0]= (-d_pred_S*lambertw(math.exp(-m*b+b-(1/d_pred_S))/d_pred_S)+b*d_pred_S-1)/(b*d_pred_S)
        A[1]= (-d_pred_R*lambertw(math.exp(-m*b+b-(1/d_pred_R))/d_pred_R)+b*d_pred_R-1)/(b*d_pred_R)
    return(A)


def GraphicalCheck():
    max = 1
    N=100
    P=np.arange(N).astype(float)
    WR=np.arange(N).astype(float)
    WS=np.arange(N).astype(float)
    for n in range(N):
        i=(n*max)/N
        P[n] = P_s(i)
        WS[n] = W(i,'S')
        WR[n] = W(i,'R')
    plt.plot(P,label='P')
    plt.plot(WS,label='S')
    plt.plot(WR,label='R')
    plt.legend()
    plt.show()













