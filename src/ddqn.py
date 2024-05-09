from matplotlib import pyplot as plt
from stable_baselines3 import DoubleDQN

from single_env import SingleNBIoT

env = SingleNBIoT()

model = DoubleDQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("ddqn")

del model # remove to demonstrate saving and loading

model = DoubleDQN.load("ddqn")

obs, info = env.reset()

total_reward = []
no_iterations = 0

while True:
  print("Testing, iteration: ", no_iterations)
  action, _states = model.predict(obs)
  obs, reward, terminated, truncated, info = env.step(action)
  total_reward.append(reward)
  no_iterations += 1
  if terminated or truncated:
    obs, info = env.reset()
    break

fig, ax = plt.subplots()
ax.plot([i for i in range(no_iterations)], total_reward)
plt.show()