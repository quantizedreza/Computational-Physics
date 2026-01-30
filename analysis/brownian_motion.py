import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb
    
# Simulation parameters
n = 10 #Number of particles 
alpha = 5.0
dt = 0.1
T = 500

def brownian_motion_simulation(n, alpha, dt, T, sigma=1):
    # Initialize positions and velocities
    x = np.zeros((n, T+1, 3)) #3 scalar equations for positions 
    v = np.zeros((n, T+1, 3)) #3 scalar equations for velocities.
    # Set initial positions with Gaussian distribution
    x[:, 0] = np.random.normal(0, 1, size=(n, 3))
    for k in range(T):
        # Random force for all particles
        delta_W = np.random.normal(0, sigma * np.sqrt(dt), size=(n, 3))
        # Update velocities
        v[:, k+1] = v[:, k] - alpha * v[:, k] * dt + delta_W
        # Update positions
        x[:, k+1] = x[:, k] + v[:, k] * dt
    return x, v


# Perform simulation
positions, velocities = brownian_motion_simulation(n, alpha, dt, T)
# Animation setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Brownian Motion of {n} Particles, Suspended in a Fluid')

# Define colors for each particle
colors = plt.cm.Set1(np.linspace(0, 1, n))

# Create lines and points for each particle with the same color
lines = []
points = []
for i in range(n):
    color = to_rgb(colors[i])
    line, = ax.plot([], [], [], '-', color=color, label=f'Particle {i}')
    point, = ax.plot([], [], [], 'o', color=color, markersize=5)
    lines.append(line)
    points.append(point)

def animate(i):
    for p in range(n):
        lines[p].set_data(positions[p, :i+1, 0], positions[p, :i+1, 1])
        lines[p].set_3d_properties(positions[p, :i+1, 2])
        points[p].set_data(positions[p, i, 0], positions[p, i, 1])
        points[p].set_3d_properties(positions[p, i, 2])
    return points + lines

anim = FuncAnimation(fig, animate, frames=len(positions[0]), interval=20, blit=True)
 
# Create a legend
#ax.legend(loc='upper right')


anim.save('multi_brownian_motion.mp4', writer='ffmpeg', fps=30)