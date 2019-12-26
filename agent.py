import numpy as np
import random
from collections import namedtuple, deque

from model import ANetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.9            # discount factor
LR = 3e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

SAVE_MODEL = "model"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, model_file=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # A-Network
        self.anetwork = ANetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.anetwork.parameters(), lr=LR)

        # storing reward and log_probs
        self.rewards = []
        self.log_probs = []
    
    def step(self, reward, log_prob):
        # This function is changed to just append th reward and log prob. Can be done in main function also
        self.rewards.append(reward)
        self.log_probs.append(log_prob)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_probs = self.anetwork.forward(state)
        actions = Categorical(action_probs)
        action = actions.sample()
        return action.item(), actions.log_prob(action)


    def learn(self, gamma = GAMMA):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + GAMMA*R
            returns.append(R)
        returns = returns[::-1]
        returns = (np.array(returns) - np.mean(returns)) / (np.std(returns) + np.finfo(np.float32).eps.item())
        
        policy_loss = []

        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        
        policy_loss.backward()
        self.optimizer.step()
        self.rewards.clear()
        self.log_probs.clear()
