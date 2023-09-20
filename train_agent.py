from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from agent import DDPGAgent
import time

# Initialize the Unity Environment
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

# Environment Goal
TARGET_SCORE = 30.0
# Averaged score
SCORE_AVERAGED = 100

# Setting the maximum amount of episodes to train
N_EPISODES = 160
# Max Timesteps
MAX_TIMESTEPS = 1000

# Initialize agent
agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=0)

# Train the agent
def train(n_episodes=N_EPISODES):
    scores_deque = deque(maxlen=SCORE_AVERAGED)
    real_scores = []
    averaged_scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    for episode in range(1, N_EPISODES + 1):
        # Get the current states for each agent
        states = env.reset(train_mode=True)[brain_name].vector_observations
        # Initialize 20x scores
        scores = np.zeros(n_agents)

        for t in range(MAX_TIMESTEPS):

            # Act according to our policy
            actions = agent.act(states)
            # Send the decided actions to all the agents
            env_info = env.step(actions)[brain_name]
            # Get next state for each agent
            next_states = env_info.vector_observations
            # Get rewards obtained from each agent
            rewards = env_info.rewards
            # Info about if an env is done
            dones = env_info.local_done
            # Learn from the collected experience
            agent.step(states, actions, rewards, next_states, dones, t)
            # Update current states
            states = next_states
            # Add the rewards recieved
            scores += rewards

            # Stop the loop if an agent is done
            if np.any(dones):
                break

        # Calculate scores and averages
        score = np.mean(scores)
        scores_deque.append(score)
        avg_score = np.mean(scores_deque)

        real_scores.append(score)
        averaged_scores.append(avg_score)
        scores_window.append(avg_score)

        # Print every 10 episodes
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

        if np.mean(scores_window) >= TARGET_SCORE:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, avg_score))
            torch.save(agent.actor_regular.state_dict(), 'actor_checkpoint.pth')
            torch.save(agent.critic_regular.state_dict(), 'critic_checkpoint.pth')
            break

    return real_scores, averaged_scores


start = time.time()
# Train the agent and get results
scores, averages = train()
print(f"Time Elapsed:{(time.time()-start)/60} mins")

# Plot training progression

plt.plot(np.arange(1, len(scores) + 1), averages)
plt.ylabel('Reacher Environment Average Score')
plt.xlabel('Episode Number')
plt.title("Score progression during training")
plt.savefig("training_progression.png")
