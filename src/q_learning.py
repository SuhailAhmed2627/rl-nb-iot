import gym
import numpy
import random
from os import system
from time import sleep

# Define function to clear console window.
def clear():
    system('clear')

clear()

# Create the environment.
env = gym.make("Taxi-v3",render_mode='ansi').env

# Create the Q-table and initialise it with zeros.
q_table = numpy.zeros([env.observation_space.n, env.action_space.n])

train_ep = 20_000
display_ep = 5
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.1 # Exploration Rate
all_epochs = []
all_penalties = []

def update_q_table(state, action, reward, next_state):
    old_value = q_table[state, action] # Retrieve old value from the q-table.
    next_max = numpy.max(q_table[next_state])
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state, action] = new_value

for i in range(train_ep):
    # Reset the environment and get the initial state.
    state = env.reset()[0]
    done = False # Set done to False to indicate the episode is not finished.
    penalties, reward, = 0, 0 # Set penalties and reward to 0.
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space.
        else:
            action = numpy.argmax(q_table[state]) # Exploit learned values.
        next_state, reward, done, truncated, info = env.step(action) 
        update_q_table(state, action, reward, next_state)
        
        # Check for illegal pickup/dropoff locations.
        if reward == -10: 
            penalties += 1
        state = next_state
    
    print(f"EP: {i}")


total_epochs, total_penalties = 0, 0

for _ in range(display_ep):
    state = env.reset()[0]
    steps, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        print(state)
        action = numpy.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)

        if reward == -10:
            penalties += 1

        steps += 1
        clear()
        print(env.render())
        print(f"Episode: {_}")
        print(f"Steps: {steps}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        sleep(0.10) 

    total_penalties += penalties
    total_epochs += steps

print(f"Results after {display_ep} episodes:")
print(f"Average steps per episode: {total_epochs / display_ep}")
print(f"Average penalties per episode: {total_penalties / display_ep}")