from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw
import

##Graphical check: return plot of P and W curves with current parameters
def Plot():
    max=1 #upper limit
    N=1000 #number of values
    P=np.arange(N).astype(float)
    HR=np.arange(N).astype(float)
    HS=np.arange(N).astype(float)
    V=[1,1]
    FR=np.arange(N).astype(float)
    FS=np.arange(N).astype(float)
    GG=np.arange(N).astype(float)
    max1=-1
    maxarg1=-1
    max2=-1
    maxarg2=-1
    for n in range(N):
        i=(n*max)/N
        P[n] = Psur(i)
        V=[1,0.01]
        HS[n] = H(i,V,'S')
        HR[n] = H(i,V,'R')
        FR[n]= F(i,'R')
        if FR[n]>max1:
            max1=FR[n]
            maxarg1=n
        FS[n]= F(i,'S')
        if FS[n]>max2:
            max2=FS[n]
            maxarg2=n
        GG[n] = G(1-i)
    plt.plot(P,label='P')
    plt.plot(FS,label='FS')
    plt.plot(FR,label='FR')
    #plt.plot(HR, label='HR')
    #plt.plot(HS, label='HS')
    plt.plot(GG,label='G')
    plt.axvline(x=maxarg1, label='riskymax')
    plt.axvline(x=maxarg2, label='safemax')
    plt.legend()
    plt.show()