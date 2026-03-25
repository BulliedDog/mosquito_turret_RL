from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from simulations.mosquito_env import MosquitoTurretEnv
import os

# Create folders
os.makedirs("models", exist_ok=True)

# 1. Instantiate the "Naked" World
base_env = MosquitoTurretEnv()

# 2. DEBUG CHECK (Must be done on the base env)
check_env(base_env)
print("Environment check passed! Wrapping for motion history...")

# 3. Wrap for Motion History (The "Secret Sauce")
# DummyVecEnv turns it into a vectorized environment (required for stacking)
# VecFrameStack(n_stack=4) lets the AI see the current frame + 3 previous frames
env = DummyVecEnv([lambda: base_env])
env = VecFrameStack(env, n_stack=4)

# 4. Build the Brain
# Note: The input layer will now automatically handle 4x the data (6 obs * 4 frames = 24 inputs)
model = PPO("MlpPolicy", env, verbose=1,
            ent_coef=0.01, 
            learning_rate=0.0003, 
            device="cpu", 
            tensorboard_log="./logs/")

# 5. Training
print("Starting training v3 with Motion Blur memory...")
model.learn(total_timesteps=1500000)

# 6. Save the result
filename = "mosquito_pilot_v4"
model.save(f"models/{filename}")
print(f"Brain saved to models/{filename}")