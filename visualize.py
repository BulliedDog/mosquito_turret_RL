import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from simulations.mosquito_env import MosquitoTurretEnv

# 1. Load and Wrap the Environment (Must match train.py exactly)
base_env = MosquitoTurretEnv()
env = DummyVecEnv([lambda: base_env])
env = VecFrameStack(env, n_stack=4)

filename = "mosquito_pilot_v4"
model = PPO.load(f"models/{filename}", env=env)

# 2. Setup the Plotting Window
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_title("AI Mosquito Turret v3: Motion-Aware Tracking")

turret_marker, = ax.plot([], [], 'ro', markersize=10, label="Turret (Laser)")
mosq_marker, = ax.plot([], [], 'g*', markersize=8, label="Mosquito")
ax.legend()

# 3. The Visualization Loop
obs = env.reset() # VecEnv reset returns just the obs
for _ in range(1000): 
    action, _states = model.predict(obs, deterministic=True)
    
    # VecEnv returns 4 values: obs, rewards, dones, infos
    obs, rewards, dones, infos = env.step(action)
    
    # --- DATA EXTRACTION ---
    # obs[0] is our one environment. It contains 24 numbers.
    # The LATEST frame is the last 6 numbers in that stack.
    latest_frame = obs[0][-6:] 
    t_x, t_y, m_x, m_y = latest_frame[0], latest_frame[1], latest_frame[2], latest_frame[3]
    
    # Update markers
    turret_marker.set_data([t_x], [t_y])
    mosq_marker.set_data([m_x], [m_y])
    
    plt.draw()
    plt.pause(0.01)
    
    # VecEnv automatically resets when 'dones' is True
    if dones[0]:
        print("Target Hit or Reset!")

plt.ioff()
plt.show()