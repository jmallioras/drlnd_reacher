from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from agent import DDPGAgent
import time

# Initialize the Reacher Unity Environment
env = UnityEnvironment(file_name="Reacher.app", no_graphics=True)
# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# Number of agents
n_agents = len(env_info.agents)
# Size of each action
action_size = brain.vector_action_space_size

# Get state space
states = env_info.vector_observations
state_size = states.shape[1]

# Initialize agent
agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=89)

# Load trained weights
agent.critic_regular.load_state_dict(torch.load('critic_checkpoint.pth'))
agent.actor_regular.load_state_dict(torch.load('actor_checkpoint.pth'))

states = env_info.vector_observations
scores = np.zeros(n_agents)
while True:
    actions = agent.act(states)
    env_info = env.step(actions)[brain_name]
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    scores += env_info.rewards
    states = next_states
    if np.any(dones):
        break
env.close()
