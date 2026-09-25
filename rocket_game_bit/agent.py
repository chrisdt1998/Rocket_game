"""
This file represents the custom rocket game agent which can be trained via DQN to learn the environment.

This file was created and designed by Christopher du Toit.
"""

import copy
import torch
import random
import numpy as np
from helper import plot
from collections import deque
from rocket_game_bit_AI import Game
from model import Linear_QNet, QTrainer

MAX_MEMORY = 20000
BATCH_SIZE = 32
# LR = 0.0001

LR = 0.00001
class Agent:
    def __init__(self, mode='train', model_path=None):
        self.n_games = 0
        # self.epsilon = 1
        self.epsilon = 0.25
        self.epsilon_decay = 0.00005
        self.epsilon_min = 0.01
        self.gamma = 0.99 # Discount rate, should be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)
        self.num_non_random_moves = 0
        self.num_random_moves = 0
        if mode == 'train':
            if model_path is None:
                self.model_main = Linear_QNet(20 * 30, 480, 3) # State size, hidden size and output action size
                self.model_target = copy.deepcopy(self.model_main)
                self.trainer = QTrainer(self.model_main, lr=LR, gamma=self.gamma)
            else:
                # Load the model from the model path
                self.model_main = Linear_QNet(20 * 30, 480, 3)
                self.model_main.load_state_dict(torch.load(model_path))
                self.model_target = copy.deepcopy(self.model_main)
                self.trainer = QTrainer(self.model_main, lr=LR, gamma=self.gamma)
        elif mode == 'test':
            if model_path is None:
                raise Exception("Model path cannot be None when mode is 'test'")
            self.model_main = Linear_QNet(20 * 30, 480, 3)
            self.model_main.load_state_dict(torch.load(model_path))
            self.model_main.eval()
        else:
            raise Exception("Mode must be either 'train' or 'test'")

    def get_state(self, game):
        # state = game.board.flatten()
        # print(game.board)
        return game.board.flatten()

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
        final_move = [0, 0, 0]
        # random moves: tradeoff exploration / exploitation
        if random.uniform(0, 1) < self.epsilon and not test:
            self.num_random_moves += 1
            move = random.randint(0, 2)
        else:
            self.num_non_random_moves += 1
            state = torch.tensor(state, dtype=torch.float)
            # print(state)
            prediction = self.model_main(state)
            # print(prediction)
            move = torch.argmax(prediction).item()
            # print(move)

        final_move[move] = 1
        # print(final_move)

        return final_move

def train(show_visuals=True, model_path=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(mode='train', model_path=model_path)
    game = Game(show_visuals=show_visuals)
    current_iter = 0
    while agent.n_games < 20000:
        # Get previous state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, agent.model_target, done)

        # Store state, action and reward
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.epsilon = agent.epsilon - agent.epsilon_decay if agent.epsilon > agent.epsilon_min else agent.epsilon_min
            agent.n_games += 1
            agent.train_long_memory(agent.model_target)

            if score > record:
                record = score
                print("Saving top scoring model")
                agent.model_main.save(file_name='top_scoring_model.pth')

            if agent.n_games % 1000 == 0:
                print(f"Saving model at {agent.n_games} games")
                agent.model_main.save(file_name=f'{agent.n_games}_model.pth')

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', agent.epsilon)
            print('Non-random moves:', agent.num_non_random_moves, 'Random moves:', agent.num_random_moves, '% Random:', agent.num_random_moves / (agent.num_non_random_moves + agent.num_random_moves) * 100, '% Non-random:', agent.num_non_random_moves / (agent.num_non_random_moves + agent.num_random_moves) * 100)
            agent.num_non_random_moves = 0
            agent.num_random_moves = 0

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

        current_iter += 1
        if current_iter % 2000 == 0:
            print("Updating model target")
            agent.model_target = copy.deepcopy(agent.model_main)

def test(show_visuals=True, model_path=None):
    model_path = model_path
    agent = Agent(mode='test', model_path=model_path)
    game = Game(show_visuals=show_visuals)
    done = False

    while not done:
        state = agent.get_state(game)
        action = agent.get_action(state, test=True)
        print('Action:', action)
        reward, done, score = game.play_step(action)
        print('Reward:', reward, 'Done:', done, 'Score:', score)


if __name__ == '__main__':
    # train(show_visuals=False, model_path='model/top_scoring_model.pth')
    test(model_path='model/top_scoring_model.pth')
    # test(model_path='checkpoints/bestrun_second_train.pth')