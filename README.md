# DP-Emotions

This is a project about applying dynamic programming techniques to animal behavior, the final goal being to incorporate emotional state in models. Dynamic programmig is useful when trying to solve optimal dynamic and state-dependent decision making. 

## Structure
The project contains, to this date, five independent programs that are based on different paradigms.  
**progdyn1** is based on a small model examplified in the book *Models of Animal Behaviour* by A. Houston and J.M. Mcnamara. The animal has to choose between a safe option u1 and a risky one u2, and optimal one-day strategies are solved.  
**progdyn2, 3 and 4** are based on the 2018 article *Trust your gut: using physiological states as a source of information is almost as effective as optimal Bayesian learning* by Higginson et al., and correspond to the different approaches presented in this article. The agent is either in a good or a bad environment, and can decide how much to invest in foraging.  
**progdyn5** is a personnal development based on these works. It incorporates fear as an estimation parameter of the safeness of an environment.  
  
  
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

## Using the codes
### Progdyn5
Contains a _Parameters_ script, that should be ran before anything else, in order to create the param.txt file used in other modules.  
Contains three scripts with different functions depending on what kind of heuristic the agent uses to assess its environment: _PerfectInfo_ where the agents are perfectly informed of their surroundings, _Bayesian_ where it learns by applying the Bayes theorem, and _Gauge_ where it takes its decisions based on the values of a fear gauge.  
Contains a _Main_ script, that runs tests using the previous functions.  

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
