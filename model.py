"""
This file represents the neural network and trainer for the DQN.

This file was created and designed by Christopher du Toit.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 48)
        self.linear3 = nn.Linear(48, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, model_target, done):
        self.model.train()
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float32)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = torch.unsqueeze(done, 0)

        next_state_Q = torch.max(model_target(next_state), dim=-1)[0]
        target = reward + (1. - done) * self.gamma * next_state_Q

        pred = self.model(state)

        if action.shape[-1] > 1 and action.shape[0] != action.shape[-1]:
            action_mask = action
        else:
            action_mask = F.one_hot(action, num_classes=3)

        masked_pred = torch.sum(action_mask * pred, dim=-1)
        self.optimizer.zero_grad()
        loss = self.criterion(masked_pred, target)
        loss.backward()

        self.optimizer.step()








