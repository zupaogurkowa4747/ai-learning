import gymnasium as gym
from stable_baselines3 import PPO

# 1. Tworzymy środowisko do treningu (CartPole: wózek i kijek)
env = gym.make("CartPole-v1")

# 2. Tworzymy model PPO (Proximal Policy Optimization - ten sam algorytm co u Yosha!)
# "MlpPolicy" oznacza, że używamy standardowej sieci neuronowej do podejmowania decyzji.
model = PPO("MlpPolicy", env, verbose=1)

print("🚀 Rozpoczynam trening AI... (potrwa to dosłownie kilka sekund)")
model.learn(total_timesteps=20000)  # AI uczy się przez 20 tysięcy kroków
print("✅ Trening zakończony!")

# 3. Tworzymy nowe środowisko, ale tym razem z podglądem graficznym (render_mode="human")
env_vis = gym.make("CartPole-v1", render_mode="human")
obs, info = env_vis.reset()

print("🎮 Oglądamy jak wytrenowany bot radzi sobie na żywo:")
for _ in range(1000):
  # Model przewiduje najlepszy ruch na podstawie obecnego stanu ekranu/fizyki
  action, _states = model.predict(obs, deterministic=True)
  obs, reward, terminated, truncated, info = env_vis.step(action)

  if terminated or truncated:
    obs, info = env_vis.reset()

env_vis.close()