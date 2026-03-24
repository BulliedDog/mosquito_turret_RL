import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from simulations.mosquito_env import MosquitoTurretEnv

# 1. Load the Environment and the Brain
env = MosquitoTurretEnv()
model = PPO.load("models/mosquito_pilot_v1")

# 2. Setup the Plotting Window
plt.ion() # Interactive mode ON
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_title("AI Mosquito Turret: Live Tracking")

# Create visual markers
turret_marker, = ax.plot([], [], 'ro', markersize=10, label="Turret (Laser)")
mosq_marker, = ax.plot([], [], 'g*', markersize=8, label="Mosquito")
ax.legend()

# 3. The Visualization Loop
obs, _ = env.reset()
for _ in range(500): # Watch 500 frames
    # Tell the brain to predict the next move
    action, _states = model.predict(obs, deterministic=True)
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract positions from the observation [T_x, T_y, M_x, M_y, V_x, V_y]
    t_x, t_y, m_x, m_y = obs[0], obs[1], obs[2], obs[3]
    
    # Update the dots on the screen
    turret_marker.set_data([t_x], [t_y])
    mosq_marker.set_data([m_x], [m_y])
    
    plt.draw()
    plt.pause(0.01) # Controls the "speed" of the simulation
    
    if terminated:
        obs, _ = env.reset()

plt.ioff()
plt.show()