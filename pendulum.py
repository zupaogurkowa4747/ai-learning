import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the Pendulum environment (no extra C++ installation needed!)
env = gym.make("Pendulum-v1", render_mode="rgb_array")

# 2. Train the PPO model using standard MlpPolicy
model = PPO("MlpPolicy", env, verbose=1)

print(" pendulum AI training started...")
model.learn(total_timesteps=50000)
print(" Training finished!")

# 3. Save the trained model
model.save("pendulum_agent")
print(" Model saved successfully!")

# 4. Watch the AI swing and balance the pendulum live
env_vis = gym.make("Pendulum-v1", render_mode="human")
obs, info = env_vis.reset()

print(" Watching our agent balance the pendulum:")
for _ in range(1000):
  action, _states = model.predict(obs, deterministic=True)
  obs, reward, terminated, truncated, info = env_vis.step(action)

  if terminated or truncated:
    obs, info = env_vis.reset()

env_vis.close()