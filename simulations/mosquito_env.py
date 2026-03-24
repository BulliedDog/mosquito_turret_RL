import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections

class MosquitoTurretEnv(gym.Env):
    def __init__(self):
        super(MosquitoTurretEnv, self).__init__()
        # Action: [Change in Pan Angle, Change in Tilt Angle]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        # Observation: [Turret_X, Turret_Y, Mosq_X, Mosq_Y, Mosq_VX, Mosq_VY]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Real-world Lag (3 frames of delay = ~100ms on Arduino)
        self.latency = 3
        self.action_buffer = collections.deque([np.array([0,0])] * self.latency)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turret_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.mosq_pos = np.random.uniform(-0.6, 0.6, 2).astype(np.float32)
        self.mosq_vel = np.random.uniform(-0.03, 0.03, 2).astype(np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.turret_pos, self.mosq_pos, self.mosq_vel]).astype(np.float32)

    def step(self, action):
        # 1. Handle Latency
        self.action_buffer.append(action)
        exec_action = self.action_buffer.popleft()

        # 2. Update Turret
        self.turret_pos = np.clip(self.turret_pos + exec_action, -1, 1)

        # 3. Update Mosquito (Physics)
        self.mosq_pos += self.mosq_vel
        self.mosq_vel += np.random.uniform(-0.005, 0.005, 2) # Wind/Zags
        self.mosq_vel = np.clip(self.mosq_vel, -0.05, 0.05)

        # 4. Calculate Reward
        dist = np.linalg.norm(self.turret_pos - self.mosq_pos)
        reward = -dist # Distance penalty
        if dist < 0.05: reward += 5.0 # Hit bonus

        # 5. End conditions
        terminated = bool(dist < 0.02 or np.any(np.abs(self.mosq_pos) > 1.1))
        return self._get_obs(), reward, terminated, False, {}