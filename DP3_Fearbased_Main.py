from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import *
from time import process_time
from scipy.special import lambertw
from random import *
#cosmetics
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


from DP3_Parameters import *
Create("param.txt")

import DP3_Fearbased_PerfectInfo as pi
import DP3_Fearbased_Bayesian as bay
import DP3_Fearbased_Gauge as gau

#Executing the parameter file
exec(open("param.txt").read())



##Numerical solution based on function F
#Function that returns the fitness-maximizing a* for E=S and E=R
#It is numerically computable by finding the maximum of the fitness function F (reached at the derivative's root)
def NumericOptimal():
    exec(open("param.txt").read(),globals())

    SAFE =  (b1*gamma_S*exp(-b1*(a - m1)))/((exp(-b1*(a - m1)) + 1)**2 (exp(-b2*(a - m2)) + 1)) + (b2*exp(-b2*(a - m2))*(gamma_S/(exp(-b1*(a - m1)) + 1) - gamma_S + 1))/(exp(-b2*(a - m2)) + 1)**2
    RISKY = (b1*gamma_R*exp(-b1*(a - m1)))/((exp(-b1*(a - m1)) + 1)**2 (exp(-b2*(a - m2)) + 1)) + (b2*exp(-b2*(a - m2))*(gamma_R/(exp(-b1*(a - m1)) + 1) - gamma_R + 1))/(exp(-b2*(a - m2)) + 1)**2

    return([SAFE,RISKY])


## Effect of average residence time in each environment on optimal perfectly informed strategy
def TimeEffect():
    step=0.01 #step at which the transition rate varies
    n = int(1/step) #number of transition rates to test for
    stability = np.zeros((n,2))
    axis=np.zeros((n,2))
    for t in range(1,n):
        #both transition times are set to be the same
        Set("param.txt","transition_S",str(t*step))
        Set("param.txt","transition_R",str(t*step))
        #fill the table
        stability[t][0],stability[t][1] = pi.PerfectInfo()
        #for a nicer plot
        axis[t]=[t*step,t*step]
    plt.plot(axis[1:],stability[1:])
    plt.xlabel('transition probability')
    plt.ylabel('optimal a with perfect info')
    #plt.xscale('log')
    plt.gca().invert_xaxis()
    #plt.gca().set_ylim([0,1])
    plt.legend(['Good','Bad'])
    plt.show()
#TimeEffect()

#Don't show trans probabilities > 0.4-0.5

##Matrix version of previous function
def TimeEffectMatrix():
    step=0.01 #step with which the transition rate varies
    n = int(1/step) #number of transition rates to test for
    stability = np.zeros((n,n,2))
    axis=np.zeros((n,2))
    for t1 in range(1,n):
        Set("param.txt","transition_S",str(t1*step)) #setting the safe transition
        for t2 in range(1,n):
            Set("param.txt","transition_R",str(t2*step)) #setting the risky transition
            stability[t1,t2,0],stability[t1,t2,1] = pi.PerfectInfo() #filling the matrix
    plt.imshow(stability[:,:,0] / stability[:,:,1])
    plt.colorbar(aspect='auto')
    plt.xlabel('transition probability from Safe')
    plt.ylabel('transition probability from Risky')
    plt.legend(['Good','Bad'])
    plt.show()
    return(stability)
#stability=TimeEffectMatrix()

## Effect of ratio of riskiness between the two envts on optimal perfectly informed strategy
def RatioEffect():
    R=100 #maximum ratio tested
    stability = np.zeros((R,2))
    axis=np.zeros((R,2))
    Set("param.txt","gamma_S",str(0.01000))
    for r in range(1,R):
        Set("param.txt","gamma_R",str(gamma_S*r))
        print(gamma_S)
        print(gamma_S*r)
        stability[r][0],stability[r][1] = pi.PerfectInfo()
        axis[r]=[r,r]
    plt.plot(axis[1:],stability[1:])
    plt.xlabel('ratio R/S')
    plt.ylabel('optimal a with perfect info')
    #plt.gca().set_ylim([0,1])
    plt.legend(['Good','Bad'])
    plt.show()
#RatioEffect()

#one day maybe test for the sensibility to sygmoidal parameters

## Effect of gamma matrix
def GammaEffectMatrix():
    step=0.01 #step with which the gammas vary until 1
    n = int(1/step) #number of transition rates to test for
    stability = np.zeros((n,n,2))
    axis=np.zeros((n,2))
    for g1 in range(1,n):
        Set("param.txt","gamma_S",str(g1*step))
        for g2 in range(g1,n):
            Set("param.txt","gamma_R",str(g2*step))
            stability[g1,g2,0],stability[g1,g2,1] = pi.PerfectInfo()
            print('g1=',g1,'g2=',g2, 'good=', stability[g1,g2,0], 'bad=', stability[g1,g2,1])
    ratio = stability[:,:,0] / stability[:,:,1]
    g = sns.heatmap(ratio, cmap="YlGnBu")
    plt.xlabel('gamma_R')
    plt.ylabel('gamma_S')
    # We take all ticks
    g.set_xticks(np.arange(len(ratio)))
    g.set_yticks(np.arange(len(ratio[0])))
    # We set half of them invisible
    plt.setp(g.get_xticklabels()[::2], visible=False)
    plt.setp(g.get_yticklabels()[::2], visible=False)
    # ... and label them with the respective list entries
    g.set_xticklabels(np.around(np.arange(0,1,0.01),2))
    g.set_yticklabels(np.around(np.arange(0,1,0.01),2))
    plt.gca().invert_yaxis()
    plt.show()
    return(stability)
#stability=GammaEffectMatrix()

##Effect of c and d on gauge results [USELESS FOR NOW - does not work with Gauge() as it is]
def c_d_Matrix():
    step=10 #step with which the gammas vary until 1
    n = int(L/step) #number of transition rates to test for
    stability = np.zeros((n,n,L+1))
    axis=np.zeros((n,2))
    for inc in range(1,n):
        Set("param.txt","c",str(inc*step))
        for dec in range(inc):
            Set("param.txt","d",str(dec*step))
            stability[inc,dec] = gau.Dummy()
            print('c=',inc*step,'d=',dec*step, 'a=', stability[inc,dec,int(L/2)])

    x = y = np.arange(0, L, step)
    # here are the x,y and respective z values
    X, Y = np.meshgrid(x, y)
    Z = np.array(np.amin(stability,axis = -1))
    # this is the value to use for the color
    V = np.amax(stability,axis = -1)

    # create the figure, add a 3d axis, set the viewing angle
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(45,60)

    # here we create the surface plot, but pass V through a colormap
    # to create a different color for each patch
    ax.plot_surface(X, Y, Z, facecolors=cm.Oranges(V))

    plt.show()
    return(stability)

stab = c_d_Matrix()

##Immediate fitness
def F(a,pred):
    if pred==0:
        return(G(1-a))
    if pred==1:
        return(Psur(a)*G(1-a))

## Simulation for perfectly informed agents
def Sim_PerfectInfo(T,IsEnv=False,environment=0):
    'Returns the accumulated fitness of an agent following a perfectly informed strategy during a period of length T generated using current paremeters (stochastically).'
    exec(open("param.txt").read(),globals()) #executing parameter file

    #First, we run the method until an optimal strategy is found by convergence
    OptA = pi.PerfectInfo()
    #Storage tables
    resulting_a = np.zeros(T) #to store the a chosen by the animal at each time step
    local_fitness = np.zeros(T) #to store the resulting fitness of the animal's behavior
    global_fitness = 0 #to store the global resulting fitness
    pred_encounter = np.zeros(T) #store predation encounters
    if IsEnv == False:
        environment = np.zeros(T) #to store the sequence of environments

    rand = 1000 #max random number that can be chosen later on. /!\ will not allow to model any lambda inferior to 1/rand
    threshold1_S = int(rand*transition_S) #threshold below which the environment switches => risky
    threshold1_R = int(rand*transition_R) #threshold below which the environment switches => safe
    threshold2_S = int(rand*gamma_S) #threshold below which there's a pred encounter in safe envt
    threshold2_R = int(rand*gamma_R) #threshold below which there's a pred encounter in risky envt
    for t in range(T):
    #We create a random sequence of environments, for T time steps, and depending on the chosen transition parameters
        r = randint(1,1000)
        if IsEnv == False:
            if t==0:
                if r<500:
                    environment[t]=0
                else:
                    environment[t]=1
            elif environment[t-1] == 0:
                if r<threshold1_S:
                    environment[t]=1
                else:
                    environment[t]=0
            elif environment[t-1] == 1:
                if r<threshold1_R:
                    environment[t]=0
                else:
                    environment[t]=1
    #We simulate the event of meeting a predator, based on the current environment's gamma
        r = randint(1,1000)

        if environment[t] == 0: #envt is safe
            if r<threshold2_S:
                pred_encounter[t]=1 #predation events storage
            else:
                pred_encounter[t]=0 #no encounter
        if environment[t] == 1: #envt is risky
            if r<threshold2_R:
                pred_encounter[t]=1 #predation events storage
            else:
                pred_encounter[t]=0 #no encounter

        #We simulate the population's optimal a's in this sequence, based on the known optimal strategies they have been selected to follow.
        opt = OptA[int(environment[t])] #0=Safe, 1=Risky
        resulting_a[t]=opt
        #We assess the cumulated fitness over the time period T of the strategy, using the core fitness function F(a) (see F1(a) in PerfectInfo), which allows us to compare the different heuristics.
        if pred_encounter[t] == 0:
            fit = F(opt,0)
        if pred_encounter[t] == 1:
            fit = F(opt,1)
        local_fitness[t]=fit
        global_fitness+=fit

    return(environment,resulting_a,local_fitness,global_fitness)

def Plot_Sim_PerfectInfo(T):
    S_PI=Sim_PerfectInfo(T)
    plt.plot(L[0], label='environment')
    plt.plot(L[1], label='resulting_a')
    plt.plot(L[2], label='local_fitness')
    plt.legend()
    plt.show()

## Simulation for bayesian agents
def Sim_Bayesian(T,IsEnv=False,environment=0):
    'Returns the accumulated fitness of an agent following a bayesian strategy during a period of length T generated using current paremeters (stochastically).'
    exec(open("param.txt").read(),globals()) #executing parameter file

    #First, we run the method until an optimal strategy is found by convergence
    OptA = bay.Bayesian()
    #Storage tables
    pred_encounter = np.zeros(T) #store predation encounters
    if IsEnv==False:
        environment = np.zeros(T) #to store the sequence of environments

    resulting_a = np.zeros(T) #to store the a chosen by the animal at each time step
    resulting_p = np.zeros(T) #to store the resulting p at each time step
    resulting_p[-1]= N/2 #initialization (see below, called at t=0)

    local_fitness = np.zeros(T) #to store the resulting fitness of the animal's behavior
    global_fitness = 0 #to store the global resulting fitness


    rand = 1000 #max random number that can be chosen later on. /!\ will not allow to model any lambda/gamma inferior to 1/rand
    threshold1_S = int(rand*transition_S) #threshold below which the environment switches => risky
    threshold1_R = int(rand*transition_R) #threshold below which the environment switches => safe
    threshold2_S = int(rand*gamma_S) #threshold below which there's a pred encounter in safe envt
    threshold2_R = int(rand*gamma_R) #threshold below which there's a pred encounter in risky envt
    for t in range(T):
    #We create a random sequence of environments, for T time steps, and depending on the chosen transition parameters
        if IsEnv==False:
            r = randint(1,1000)
            if t==0: #first step: equal chances of starting with safe and risky
                if r<500:
                    environment[t]=0
                else:
                    environment[t]=1
            elif environment[t-1] == 0: #envt was safe
                if r<threshold1_S:
                    environment[t]=1 #switched
                else:
                    environment[t]=0
            elif environment[t-1] == 1: #envt was risky
                if r<threshold1_R:
                    environment[t]=0 #switched
                else:
                    environment[t]=1

    #We simulate the event of meeting a predator, based on the current environment's gamma
    #Due to approximations made in the nextp function, without stochasticity, unwanted clippings on p=0 and p=N occur. We solve this by forcing p out of 0 if no predator encounter happens, and by forcing it down N when an encounter happens.
        r = randint(1,1000)
        if environment[t] == 0: #envt is safe
            if r<threshold2_S:
                pred_encounter[t]=1 #predation events storage
                #Correcting for the absorption by max p = N
                if resulting_p[t-1] == N :
                    resulting_p[t] = bay.nextp(resulting_p[t-1]-1,1)
                else:
                    resulting_p[t] = bay.nextp(resulting_p[t-1],1)
            else:
                pred_encounter[t]=0 #no encounter
                #Correcting for the absorption by min p = 0
                if resulting_p[t-1] == 0 :
                    resulting_p[t] = bay.nextp(resulting_p[t-1]+1,0)
                else:
                    resulting_p[t] = bay.nextp(resulting_p[t-1],0)
        if environment[t] == 1: #envt is risky
            if r<threshold2_R:
                pred_encounter[t]=1 #predation events storage
                #Correcting for the absorption by 100
                if resulting_p[t-1] == N:
                    resulting_p[t] = bay.nextp(resulting_p[t-1]-1,1)
                else:
                    resulting_p[t] = bay.nextp(resulting_p[t-1],1)
            else:
                pred_encounter[t]=0 #no encounter
                #Correcting for the absorption by min p = 0
                if resulting_p[t-1] == 0 :
                    resulting_p[t] = bay.nextp(resulting_p[t-1]+1,0)
                else:
                    resulting_p[t] = bay.nextp(resulting_p[t-1],0)

        #We simulate the population's optimal a's in this sequence, based on the known optimal strategies they have been selected to follow.
        opt = OptA[int(min(resulting_p[t],100))]
        resulting_a[t]=opt
        #We assess the cumulated fitness over the time period T of the strategy, using the core fitness function F(a) (see F1(a) in PerfectInfo), which allows us to compare the different heuristics.
        if pred_encounter[t] == 0:
            fit = F(opt,0)
        if pred_encounter[t] == 1:
            fit = F(opt,1)
        local_fitness[t]=fit
        global_fitness+=fit

    return(environment,resulting_a,local_fitness,global_fitness, resulting_p, pred_encounter)


def Plot_Sim_Bayesian(T):
    'Function that runs a simulation of length T and then plots it adequately'

    #Time table
    t = np.around(np.arange(0,T,1),1)
    #Simulation results
    environment,resulting_a,local_fitness,global_fitness, resulting_p, pred_encounter = Sim_Bayesian(T)

    #Plotting
    plt.title('Bayesian Simulation')

    #Plot environment
    env = plt.subplot(511)
    #Make colors out of environment list
    colors = list(map(lambda x: "yellow" if x else "green", environment)) #green if safe, yellow is risky
    plt.bar(t,np.ones(T),width=1.0, color=colors) #barplot without inter-bar spaces
    env.set_ylabel('Environment')
    #Legend (built manually)
    legend_elements = [Patch(facecolor='green',edgecolor='white', label='Safe'), Patch(facecolor='yellow', edgecolor='black',label='Risky')]
    env.legend(handles=legend_elements, loc='upper right')

    #Plotting chosen antipredator behavior a
    a = plt.subplot(512,sharex=env)
    plt.plot(resulting_a)
    plt.eventplot(t*pred_encounter,lineoffsets = 0.5, linelengths = 1, color = 'gray', linewidth=0.5)
    plt.setp(a.get_xticklabels(), visible=False)
    plt.ylim(0.5, 0.6)
    a.set_ylabel('Resulting a')

    #Plotting the resulting estimate p
    p = plt.subplot(513,sharex=env)
    plt.plot(resulting_p/N)
    plt.eventplot(t*pred_encounter,lineoffsets = 0.5, linelengths = 1, color = 'gray', linewidth=0.5)
    plt.setp(p.get_xticklabels(), visible=False)
    plt.ylim(0, 1)
    p.set_ylabel('Resulting estimate p')
    p.fill_between(t, resulting_p/N, color='green')
    p.fill_between(t, resulting_p/N, np.ones(T), color='yellow')

    localf= plt.subplot(514,sharex=env)
    plt.plot(local_fitness, label='local_fitness')
    plt.eventplot(t*pred_encounter,lineoffsets = 0.5, linelengths = 1, color = 'gray', linewidth=0.5)
    plt.setp(localf.get_xticklabels(), visible=False)
    plt.ylim(0.95, 1)
    localf.set_ylabel('Local Fitness')

    plt.show()


#Plot_Sim_Bayesian(1000)

## Simulation for gauge-using agents
def Sim_Gauge(T,IsEnv=False,environment=0):
    'Returns simulation of an agent following a gauge-based strategy during a period of length T generated using current paremeters (stochastically).'

    exec(open("param.txt").read(),globals()) #executing parameter file

    #First, we run the method until an optimal strategy is found by convergence
    OptA = gau.Dummy()
    #Storage tables
    pred_encounter = np.zeros(T) #store predation encounters
    if IsEnv==False:
        environment = np.zeros(T) #to store the sequence of environments


    resulting_a = np.zeros(T) #to store the a chosen by the animal at each time step
    resulting_g = np.zeros(T) #to store the resulting p at each time step
    resulting_g[-1]= L/2 #initialization (see below, called at t=0)

    local_fitness = np.zeros(T) #to store the resulting fitness of the animal's behavior
    global_fitness = 0 #to store the global resulting fitness

    rand = 1000 #max random number that can be chosen later on. /!\ will not allow to model any lambda/gamma inferior to 1/rand
    threshold1_S = int(rand*transition_S) #threshold below which the environment switches => risky
    threshold1_R = int(rand*transition_R) #threshold below which the environment switches => safe
    threshold2_S = int(rand*gamma_S) #threshold below which there's a pred encounter in safe envt
    threshold2_R = int(rand*gamma_R) #threshold below which there's a pred encounter in risky envt
    for t in range(T):
    #We create a random sequence of environments, for T time steps, and depending on the chosen transition parameters
        if IsEnv==False:
            r = randint(1,1000)
            if t==0: #first step: equal chances of starting with safe and risky
                if r<500:
                    environment[t]=0
                else:
                    environment[t]=1
            elif environment[t-1] == 0: #envt was safe
                if r<threshold1_S:
                    environment[t]=1 #switched
                else:
                    environment[t]=0
            elif environment[t-1] == 1: #envt was risky
                if r<threshold1_R:
                    environment[t]=0 #switched
                else:
                    environment[t]=1

    #We simulate the event of meeting a predator, based on the current environment's gamma
        r = randint(1,1000)

        if environment[t] == 0: #envt is safe
            if r<threshold2_S:
                pred_encounter[t]=1 #predation events storage
                resulting_g[t] = gau.nextg(resulting_g[t-1],1)
            else:
                pred_encounter[t]=0 #no encounter
                resulting_g[t] = gau.nextg(resulting_g[t-1],0)

        if environment[t] == 1: #envt is risky
            if r<threshold2_R:
                pred_encounter[t]=1 #predation events storage
                resulting_g[t] = gau.nextg(resulting_g[t-1],1)
            else:
                pred_encounter[t]=0 #no encounter
                resulting_g[t] = gau.nextg(resulting_g[t-1],0)

        #We simulate the population's optimal a's in this sequence, based on the known optimal strategies they have been selected to follow.
        opt = OptA[int(min(resulting_g[t],100))]
        resulting_a[t]=opt
        #We assess the cumulated fitness over the time period T of the strategy, using the core fitness function F(a) (see F1(a) in PerfectInfo), which allows us to compare the different heuristics.
        if pred_encounter[t] == 0:
            fit = F(opt,0)
        if pred_encounter[t] == 1:
            fit = F(opt,1)
        local_fitness[t]=fit
        global_fitness+=fit

    return(environment,resulting_a,local_fitness,global_fitness, resulting_g, pred_encounter)

def Plot_Sim_Gauge(T):
    'Function that runs a simulation of length T and then plots it adequately'

    #Time table
    t = np.around(np.arange(0,T,1),1)
    #Simulation results
    environment,resulting_a,local_fitness,global_fitness, resulting_g, pred_encounter = Sim_Gauge(T)

    #Plotting
    plt.title('Gauge Simulation')

    #Plot environment
    env = plt.subplot(511)
    #Make colors out of environment list
    colors = list(map(lambda x: "yellow" if x else "green", environment)) #green if safe, yellow is risky
    plt.bar(t,np.ones(T),width=1.0, color=colors) #barplot without inter-bar spaces
    env.set_ylabel('Environment')
    #Legend (built manually)
    legend_elements = [Patch(facecolor='green',edgecolor='white', label='Safe'), Patch(facecolor='yellow', edgecolor='black',label='Risky')]
    env.legend(handles=legend_elements, loc='upper right')

    #Plotting chosen antipredator behavior a
    a = plt.subplot(512,sharex=env)
    plt.plot(resulting_a)
    plt.eventplot(t*pred_encounter,lineoffsets = 0.5, linelengths = 1, color = 'gray', linewidth=0.5)
    plt.setp(a.get_xticklabels(), visible=False)
    plt.ylim(0.5, 0.6)
    a.set_ylabel('Resulting a')

    #Plotting the resulting gauge level g
    g = plt.subplot(513,sharex=env)
    plt.plot(resulting_g/L)
    plt.eventplot(t*pred_encounter,lineoffsets = 0.5, linelengths = 1, color = 'gray', linewidth=0.5)
    plt.setp(g.get_xticklabels(), visible=False)
    plt.ylim(0, 1)
    g.set_ylabel('Resulting gauge level g')
    g.fill_between(t, resulting_g/L, color='yellow')
    g.fill_between(t, resulting_g/L, np.ones(T), color='green')

    localf= plt.subplot(514,sharex=env)
    plt.plot(local_fitness, label='local_fitness')
    plt.eventplot(t*pred_encounter,lineoffsets = 0.5, linelengths = 1, color = 'gray', linewidth=0.5)
    plt.setp(localf.get_xticklabels(), visible=False)
    plt.ylim(0.95, 1)
    localf.set_ylabel('Local Fitness')

    plt.show()

Plot_Sim_Gauge(500)

## Comparison

#Reminder
#Sim_PerfectInfo: returns (environment,resulting_a,local_fitness,global_fitness)
#Sim_Bayesian: returns (environment,resulting_a,local_fitness,global_fitness, resulting_p, pred_encounter)
#Sim_Gauge: returns (environment,resulting_a,local_fitness,global_fitness, resulting_g, pred_encounter)

#def GlobalFitComp():






