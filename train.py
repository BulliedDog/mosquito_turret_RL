from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from simulations.mosquito_env import MosquitoTurretEnv
import os

# Create folders
os.makedirs("models", exist_ok=True)

# 1. Instantiate the World
env = MosquitoTurretEnv()

# 2. DEBUG CHECK (Don't skip this!)
# This ensures your math is compatible with the AI library
check_env(env)
print("Environment check passed! Starting training...")

# 3. Build the Brain
# We use MlpPolicy (standard neural network)
model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./logs/")

# 4. Training (Start with 100k steps to test)
model.learn(total_timesteps=100000)

# 5. Save the result
model.save("models/mosquito_pilot_v1")
print("Brain saved to models/mosquito_pilot_v1.zip")