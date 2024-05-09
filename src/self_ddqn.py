import gym
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Nadam
from keras.losses import mean_squared_error
from collections import deque

env = gym.make("CartPole-v1", render_mode="rgb_array")

input_shape = [4]
n_outputs = 2
batch_size = 4
discount_factor = 0.95
optimizer = Nadam(learning_rate=1e-2)
loss_fn = mean_squared_error

model = Sequential([
   Dense(32, activation="elu", input_shape=input_shape),
   Dense(32, activation="elu"),
   Dense(n_outputs)
])

target = tf.keras.models.clone_model(model)
target.set_weights(model.get_weights())

replay_buffer = deque(maxlen=2000)

def epsilon_greedy_policy(state, epsilon=0):
   if np.random.rand() < epsilon:
      return np.random.randint(n_outputs)
   else:
      Q_values = model.predict(state[np.newaxis], verbose=0)[0]
      print(Q_values)
   m = Q_values.argmax()
   print(m)
   return m

def sample_experiences(batch_size):
   indices = np.random.randint(len(replay_buffer), size=batch_size)
   batch = [replay_buffer[index] for index in indices]
   sample_exp = [
      np.array([experience[field_index] for experience in batch]) for field_index in range(6)
   ]
   return sample_exp

def play_one_step(env, state, epsilon):
   action = epsilon_greedy_policy(state, epsilon)
   next_state, reward, done, truncated, info = env.step(action)
   replay_buffer.append((state, action, reward, next_state, done, truncated))
   return next_state, reward, done, truncated, info


def training_step(batch_size):
   experiences = sample_experiences(batch_size)
   states, actions, rewards, next_states, dones, truncateds = experiences
   next_Q_values = model.predict(next_states, verbose=0)
   best_next_actions = next_Q_values.argmax(axis=1)
   next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
   max_next_Q_values = (target.predict(next_states, verbose=0) * next_mask).sum(axis=1)
   runs = 1.0 - (dones | truncateds)
   target_Q_values = rewards + runs * discount_factor * max_next_Q_values
   target_Q_values = target_Q_values.reshape(-1, 1)
   mask = tf.one_hot(actions, n_outputs)

   with tf.GradientTape() as tape:
      all_Q_values = model(states)
      Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
      loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
   grads = tape.gradient(loss, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))

rewards = []
no_episodes = 600

for episode in range(no_episodes):
   obs, info = env.reset()
   print("Episode: " + str(episode))
   sum_of_rewards = 0
   for step in range(200):
      epsilon = max(1 - episode / 500, 0.01)
      obs, reward, done, truncated, info = play_one_step(env, obs, epsilon)
      sum_of_rewards += reward
      if done or truncated:
         break
   rewards.append(sum_of_rewards)
   if episode % 50 == 0:
      target.set_weights(model.get_weights())
   if episode > 50:
      training_step(batch_size)

env.close()
