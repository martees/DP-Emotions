#For code structure reasons, parameters are stored in a separate txt file.
#This part of the code should be run

#Can be used to both reset the parameters file, or create a new parameters file.
#Takes the name of the file as a string in argument.
#WARNING: for length-keeping purposes, all parameters should be set with four digits, or "0." + four digits when relevant.
def Create(filename):
    "Creates parameter file, eg Create('param.txt')"
    File = open(filename,'w')
    #Safe environment
    File.write('gamma_S=0.0100 \n') #density of predators
    File.write('transition_S=0.0100\n')
    #Risky environment
    File.write('gamma_R=0.5000 \n') #density of predators
    File.write('transition_R=0.0100\n')
    #Survival function (probability of surviving a predator attack with level of antipredator behavior = a)
    File.write('m1=0.5000 \n') #inflexion point
    File.write('b1=100 \n') #steepness of growth
    File.write('def Psur(a):\n')
    File.write('    return( 1/(1+exp(-b1*(a-m1))) )\n')
    #Payoff function (expected immediate fitness gain from level of antipredator behavior = a)
    File.write('m2=0.2000 \n')
    File.write('b2=20 \n')
    File.write('def G(a): \n')
    File.write('    return( 1/(1+exp(-b2*(a-m2))) ) \n')
    #Possible values for a
    File.write('Na=1000 \n') #number of possible values
    File.write('alist=np.arange(0,1,1/Na) \n') #corresponding interval
    #Number of possible values for estimate p that E=S\n'
    File.write('N=100 \n')
    #Gauge parameters
    File.write('L=100 \n') #size of the gauge
    File.write('d=10 \n') #fear decrease at each time step (base value)
    File.write('c=30 \n') #fear increase when encountering a predator (base value)
    File.close()

#Function that allows global parameter editing.
#Takes three str
def Set(filename,par,value):
    "All args are strings. Set parameter par to a (4-digit or '0.'+4-digit) value."
    par += "="
    f = open(filename, 'r')
    lines = f.readlines()
    for l in range(len(lines)):
        line = lines[l]
        if line.find(par) != -1:
            lines[l] = par + value + '\n'
            f.close()
            f = open(filename, 'w')
            f.writelines(lines)
            f.close()



Create("param.txt")






















