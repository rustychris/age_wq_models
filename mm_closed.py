import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw
##

t=np.linspace(0,40,100)

# numerical solution for 1st order kinetics
Cinit=1.0 # [mg/l] = 70uM
knit=0.15 # [d-1]

def dCdt1(t,C):
    return -knit*C

soln1=solve_ivp(dCdt1,
                [t[0],t[-1]],
                [Cinit],
                t_eval=t,
                dense_output=True)

# MM kinetics
Csat=0.5 # mgN/l
kmm=2*Csat*knit # rough magnitude for matching rate

def dCdtMM(t,C):
    return -kmm*C/(Csat+C)

solnMM=solve_ivp(dCdtMM,
                [t[0],t[-1]],
                [Cinit],
                t_eval=t,
                dense_output=True)

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
ax.plot(soln1.t,soln1.y[0,:],label='1st order-num')
ax.plot(soln1.t,Cinit*np.exp(-soln1.t*knit),label='1st order-ana',ls='--')

ax.plot(solnMM.t,solnMM.y[0,:],label='MM-num')

# Analytical MM:
F=Cinit/Csat*np.exp( Cinit/Csat - kmm/Csat*solnMM.t)
CMM_ana = Csat*lambertw(F)
ax.plot(solnMM.t,np.real(CMM_ana),label='MM-ana',ls='--')

ax.legend()
ax.axis(xmin=0,ymin=0)

##

