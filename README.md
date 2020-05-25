# DP-Emotions

This is a project about applying dynamic programming techniques to animal behavior, the final goal being to incorporate emotional state in models. Dynamic programmig is useful when trying to solve optimal dynamic and state-dependent decision making. 

## Prerequisites

-  Python 3.0 or higher (was developped with Python 3.7.3)
-  Python packages and versions:
      - numpy              1.16.4 
      - matplotlib         3.1.0   
      - scipy              1.3.0   

They can be simply installed with a pip command in Python.
```
pip install numpy
```

## Global Structure
The project contains, to this date, five independent programs that are based on different paradigms.  
**DP1** is based on a small model examplified in the book *Models of Animal Behaviour* by A. Houston and J.M. Mcnamara. 
**DP2** algorithms are based on the 2018 article *Trust your gut: using physiological states as a source of information is almost as effective as optimal Bayesian learning* by Higginson et al., and correspond to the different approaches presented in this article.  
**DP3** is a personnal development based on these works. It incorporates fear as an estimation parameter of the safeness of an environment. 


## Theoretical elements
### Dynamic Programming
The goal of this project is to model an agent's optimal behavior in a dynamic environment. Dynamic programming is an algorithmic method that allows us to numerically solve such behavior in a polynomial time, to the one condition that each expression at a given timestep can be expressed based on the previous timestep's values.  
In order to achieve this, a function V is introduced: V is a function of the animal's state and its environment, and represents a relative probability of survival from this state on, provided that through the next steps the agent's strategy is optimal. In all of the algorithms, one of the main loops' purpose is to perform backward optimization to compute the optimal strategy based on a given V, until V converges.    

### DP1: Day in Winter
The animal has to choose between a safe option u1 and a risky one u2, and optimal one-day strategies are solved.   

### DP2: Trust Your Guts
The environment can be either good or bad (differing in food finding probability), transitionning with a certain probabiity from one to the other at each time step.  
The agent's state is described by its food reserve levels. It can decide how much to invest in foraging.  
- In the PerfectInfo paradigm, the agents know every environment change, acting accordingly.  
- In the Bayesian paradigm, the agents estimate the probability for the conditions being good, at each timestep, and update it based on the food items they find, in a Bayesian way (note: this estimate is, for now, called "p" instead of the rho used in the article).  
- *in development* In the ReserveBased paradigm, the agents estimate the probability for the conditions being good based off of their reserve levels. Due to temporal autocorrelation, we indeed expect reserve levels to be correlated to external conditions (being higher when the environment is good).  

### DP3: Fearbased
The environment can be either safe or risky (differing in predator encounter probability), transitionning with a certain probabiity from one to the other at each time step.  
The agent can choose its level of antipredator behavior. This level yields a certain amount of protection against predators, and a certain amount of accumulated fitness (both of these pay-offs are sygmoidal).
- PerfectInfo is the same as the Trust Your Guts one.
- Bayesian estimates are based on predator encounters instead of food items found, since the agents' reserve levels are not taken into account in this model.
- *in development* 

## In Practice

### DP3 files
Description of progdyn5 files: 
- a _Parameters_ script, that should be ran before anything else, in order to create the param.txt file used in other modules. It contains a Set function, that allows changing any parameter in the file, at any time.
- three scripts corresponding to the 
- Contains a _Main_ script, that runs tests using the previous functions.  

```
Give an example
```


## Built With

Pyzo 4.7.3 (https://pyzo.org/)


## Authors

* **Alice Al-Asmar** - *Most of the coding*
* **François-Xavier Dechaume-Moncharmont** - *Theoretical development*
* **Orégane Desrentes** - *Fast-and-furious-ing of the codes that required it*


## References 

* Alasdair Houston and John M. McNamara 
* TYG
