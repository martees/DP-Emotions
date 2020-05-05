from progdyn5_Parameters import *
Create("param.txt")

from progdyn5_Fearbased_PerfectInfo import *
from progdyn5_Fearbased_Bayesian import *
from progdyn5_Fearbased_Gauge import *

#Executing the parameter file
exec(open("param.txt").read())



##Numerical solution based on function F
#Function that returns the fitness-maximizing a* for E=S and E=R
#It is numerically computable by finding the maximum of the fitness function F (reached at the derivative's root)
def NumericOptimal():
    SAFE =  (b1*gamma_S*exp(-b1*(a - m1)))/((exp(-b1*(a - m1)) + 1)**2 (exp(-b2*(a - m2)) + 1)) + (b2*exp(-b2*(a - m2))*(gamma_S/(exp(-b1*(a - m1)) + 1) - gamma_S + 1))/(exp(-b2*(a - m2)) + 1)**2
    RISKY = (b1*gamma_R*exp(-b1*(a - m1)))/((exp(-b1*(a - m1)) + 1)**2 (exp(-b2*(a - m2)) + 1)) + (b2*exp(-b2*(a - m2))*(gamma_R/(exp(-b1*(a - m1)) + 1) - gamma_R + 1))/(exp(-b2*(a - m2)) + 1)**2
    return([SAFE,RISKY])


## Effect of average residence time in each environment on optimal perfectly informed strategy
def EffectOfTime():
    T=1000 #maximum mean residence time tested
    stability = np.zeros((T,2))
    for t in range(1,T):
        Set("param.txt","transition_S",str((t-1)/T))
        Set("param.txt","transition_R",str((t-1)/T))
        stability[t][0],stability[t][1] = PerfectInfo()
    plt.plot(stability[1:])
    plt.xscale('log')
    plt.legend(['Good','Bad'])
    plt.show()


## Effect of ratio of riskiness between the two envts on optimal perfectly informed strategy
def EffectOfRatio():
    R=10 #maximum ratio tested
    step=0.1 #ratio granularity
    stability = np.zeros((int(R/step),2))
    Set("param.txt","gamma_S",str(0.1000))
    for r in np.arange(1,R/step).astype(int):
        Set("param.txt","gamma_R",str(0.1*int(r/R)))
        stability[r][0],stability[r][1] = PerfectInfo()
    plt.plot(stability[1:])
    plt.legend(['Good','Bad'])
    plt.show()



## Simulation
#First, we run the different methods until an optimal strategy is found by convergence


#Then, we create a random sequence of environments, for L time steps, and depending on the chosen transition parameters


#We simulate the different populations' optimal a's in this sequence, based on the known evolution of p curves and the known optimal strategies they have been selected to follow.


#We assess the cumulated fitness over the time period L of each strategy, using the core fitness function F(a) (see F1(a) in PerfectInfo), which allows us to compare the different heuristics.
