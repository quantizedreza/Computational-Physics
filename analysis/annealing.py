
# now, let's flip a random spin and calculate the new energy

def flip_random_spin(state):
    """Flip a random spin in the state."""
    new_state = state.copy()
    index = random.randint(0, len(state) - 1)
    new_state[index] *= -1
    return new_state
    

new_a_k = flip_random_spin(a_k)
new_E = round(energy(new_a_k, J), 2)
print("energy of new state after flipping a random spin:", new_E)

# determine if we accept the new state
def accept_new_state(E_old, E_new, T):
    """Decide whether to accept the new state based on energy difference and temperature."""
    if E_new < E_old:
        return True
    else:
        delta_E = E_new - E_old
        acceptance_probability = np.exp(-delta_E / T)
        return random.random() < acceptance_probability
T = 1.0  # example temperature
if accept_new_state(E, new_E, T):
    a_k = new_a_k
    E = new_E
    print("New state accepted.")
else:
    print("New state rejected.")   
print("Current energy:", E) 

print("Current state:", a_k)

# plot as a function of kT, the energy and magnetization
import matplotlib.pyplot as plt
temperatures = np.linspace(0.1, 5.0, 50)
energies = []
magnetizations = [] 

for T in temperatures:
    a_k = np.array(random.choices([-1, 1], k=100))  # reset to random state
    E = energy(a_k, J)
    for _ in range(1000):  # number of Monte Carlo steps
        new_a_k = flip_random_spin(a_k)
        new_E = energy(new_a_k, J)
        if accept_new_state(E, new_E, T):
            a_k = new_a_k
            E = new_E
    energies.append(E / len(a_k))  # average energy per spin
    magnetizations.append(np.abs(np.sum(a_k)) / len(a_k))  # average magnetization per spin
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(temperatures, energies, label='Energy per spin')
plt.xlabel('Temperature (kT)')
plt.ylabel('Energy')
plt.title('Energy vs Temperature')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(temperatures, magnetizations, label='Magnetization per spin', color='orange')
plt.xlabel('Temperature (kT)')
plt.ylabel('Magnetization')
plt.title('Magnetization vs Temperature')
plt.legend()
plt.tight_layout()
plt.show()
