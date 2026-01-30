# Simple Monte Carlo Simulation to calculate the value for pi=3.14...
import numpy as np
area=1
area_out_circle=0
n=1000000 # choose 
# a random number between 0 and 1:
def random():
    return np.random.rand()
random() #this is a number between 0 and 1 by random.
for i in range(n):
    x=random()
    y=random()
    distance=x**2+y**2
    if distance<=1:
        area_out_circle+=1

pi=4*area_out_circle/n

print(pi)
