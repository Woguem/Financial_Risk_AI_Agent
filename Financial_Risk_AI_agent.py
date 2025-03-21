"""
@author: Dr Yen Fred WOGUEM 

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Definition of the Q-learning model
class QLearningAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(QLearningAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, action_size)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  

# Initialisation
state_size = 4  # Customer financial data
action_size = 2  # 0 = Refuse the loan, 1 = Accepting the loan
gamma = 0.99  
lr = 0.001  
epsilon = 0.1  
episodes = 1000  

# Agent and optimizer initialization
agent = QLearningAgent(state_size, action_size)
optimizer = optim.Adam(agent.parameters(), lr=lr)
criterion = nn.MSELoss()

# Customer solvency simulation function
def evaluer_solvabilite(state):
    score = np.mean(state)  # Customer data is averaged
    return score > 0  # If the average > 0, the customer is considered solvent

# Agent training
for episode in range(episodes):
    state = np.random.randn(state_size)  # Simulate a customer with random financial data
    state = torch.FloatTensor(state)
    
    done = False
    total_reward = 0

    while not done:
        # Choose an action (exploration or exploitation)
        if random.random() < epsilon:
            action = random.choice(range(action_size))  # Exploration
        else:
            with torch.no_grad():
                action = torch.argmax(agent(state)).item()  # Exploitation

        # Assessing customer creditworthiness
        client_solvable = evaluer_solvabilite(state.numpy())

        # Determining the reward based on the decision
        if client_solvable and action == 1:  # Loan granted to a good customer
            reward = 10  
        elif not client_solvable and action == 0:  # Loan refused to a bad customer
            reward = 5  
        elif client_solvable and action == 0:  # Loan refused to a good customer (error)
            reward = -5  
        else:  # Loan granted to wrong customer (big mistake)
            reward = -10  

        next_state = np.random.randn(state_size)  # Simulate the customer's new financial situation
        next_state = torch.FloatTensor(next_state)

        # Target value calculation 
        with torch.no_grad():
            target = reward + gamma * torch.max(agent(next_state))  

        # Model update
        prediction = agent(state)[action]  
        loss = criterion(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        state = next_state  

        # We finish after a single decision
        done = True  

    if episode % 100 == 0:
        print(f"Episode {episode}: Total reward = {total_reward}")

print("Training completed.")















