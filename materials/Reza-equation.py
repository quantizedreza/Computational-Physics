#(c) Reza Rahemi 
import numpy as np
import matplotlib.pyplot as plt
# evenly sampled temperature at 5k intervals
t = np.arange(295., 1000., 5)
phi0= 3. # in eV constant for metal 
gamma= 583. # Rahemi, R., & Li, D. (2015). Variation in electron work function with temperature and its effect on the Young’s modulus of metals. Scripta Materialia, 99, 41-44.
K= 8.6173303*10.**(-5.) #Boltzman constant eV/k
phiT= phi0 - gamma*((K*t)**2)/(phi0)
plt.title("Electron Work Function Vs Temperature")
plt.xlabel('Temperature/K')
plt.ylabel('Work Function/eV')
plt.plot(t,phiT)
plt.show()
