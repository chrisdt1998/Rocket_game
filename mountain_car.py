"""
This file represents the mountain car agent which can be trained via DQN to learn the environment.

This file was created and designed by Christopher du Toit.
"""

import copy
import torch
import random
from collections import deque
from model import Linear_QNet, QTrainer
import gym
game = gym.make('MountainCar-v0')

MAX_MEMORY = 20000
BATCH_SIZE = 32
# LR = 0.0001

LR = 0.001


class Agent:
    def __init__(self, model_path=None):
        self.n_games = 0
        self.epsilon = 1
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.01
        self.gamma = 0.99 # Discount rate, should be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)
        if model_path is None:
            self.model_main = Linear_QNet(2, 24, 3) # State size, hidden size and output action size
            self.model_target = copy.deepcopy(self.model_main)
            self.trainer = QTrainer(self.model_main, lr=LR, gamma=self.gamma)
        else:
            self.model_main = Linear_QNet(2, 24, 3)
            self.model_main.load_state_dict(torch.load(model_path))
            self.model_main.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self, model_target):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, model_target, dones)

    def train_short_memory(self, state, action, reward, next_state, model_target, done):
        self.trainer.train_step(state, action, reward, next_state, model_target, done)

    def get_action(self, state, test=False):
        # random moves: tradeoff exploration / exploitation
        if random.uniform(0, 1) < self.epsilon and not test:
            move = random.randint(0, 2)
        else:
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model_main(state)
            move = torch.argmax(prediction).item()

        return move

def train(show_visuals=True):
    record = -200
    agent = Agent()
    state_old = game.reset(return_info=False)
    current_iter = 0
    score = 0
    while True:
        if show_visuals and current_iter % 20 == 0:
            game.render()
        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        state_new, reward, done, info = game.step(final_move)

        # Adjust reward for task completion
        if state_new[0] >= 0.5:
            reward += 10

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, agent.model_target, done)

        # Store state, action and reward
        agent.remember(state_old, final_move, reward, state_new, done)

        state_old = state_new

        score += reward

        if done:
            # Train long memory, plot result
            state_old = game.reset()
            agent.n_games += 1
            agent.train_long_memory(agent.model_target)
            agent.epsilon = agent.epsilon - agent.epsilon_decay if agent.epsilon > agent.epsilon_min else agent.epsilon_min

            if score > record:
                record = score
                print("Saving model")
                agent.model_main.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            score = 0

        current_iter += 1
        if current_iter % 2000 == 0:
            print("Updating model target")
            agent.model_target = copy.deepcopy(agent.model_main)


def test(show_visuals=True):
    model_path = 'model/model.pth'
    agent = Agent(model_path=model_path)
    done = False
    state = game.reset(return_info=False)
    score = 0
    while not done:
        action = agent.get_action(state, test=True)
        state, reward, done, info = game.step(action)
        score += reward

        if show_visuals:
            game.render()

    print(score)
if __name__ == '__main__':
    train(show_visuals=True)
    # test()