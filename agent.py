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
from rocket_game_AI import Game
from model import Linear_QNet, QTrainer

MAX_MEMORY = 20000
BATCH_SIZE = 32
LR = 0.0001

class Agent:
    def __init__(self, model_path=None):
        self.n_games = 0
        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.01
        self.gamma = 0.99 # Discount rate, should be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)
        if model_path is None:
            self.model_main = Linear_QNet(22, 24, 3) # State size, hidden size and output action size
            self.model_target = copy.deepcopy(self.model_main)
            self.trainer = QTrainer(self.model_main, lr=LR, gamma=self.gamma)
        else:
            self.model_main = Linear_QNet(22, 24, 3)
            self.model_main.load_state_dict(torch.load(model_path))
            self.model_main.eval()


    def get_state(self, game):
        # State contains the position of the player, nearby rocks positions, nearby rocks sizes, nearby rocks speeds.
        # The nearby rocks have been rearrange from nearest to furthest.
        nearby_rocks = game.danger_rocks(5)
        nearby_rock_positions = []
        nearby_rock_sizes = []
        nearby_rock_speeds = []
        for pos, size, speed in zip(nearby_rocks['positions'], nearby_rocks['sizes'], nearby_rocks['speeds']):
            nearby_rock_positions += (pos/game.window_width).tolist()
            nearby_rock_sizes.append(size/game.rock_size_upr_bnd)
            nearby_rock_speeds.append(speed/game.speed_upr_bnd)
        state = [game.player.position[0]/game.window_height, game.player.position[1]/game.window_width] + nearby_rock_positions + nearby_rock_sizes + nearby_rock_speeds
        return np.array(state, dtype=float)

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
            move = random.randint(0, 2)
        else:
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model_main(state)
            move = torch.argmax(prediction).item()

        final_move[move] = 1

        return final_move

def train(show_visuals=True):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game(show_visuals=show_visuals)
    current_iter = 0
    while True:
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
                print("Saving model")
                agent.model_main.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

        current_iter += 1
        if current_iter % 2000 == 0:
            print("Updating model target")
            agent.model_target = copy.deepcopy(agent.model_main)

def test(show_visuals=True):
    model_path = 'model/model.pth'
    agent = Agent(model_path=model_path)
    game = Game(show_visuals=show_visuals)
    done = False

    while not done:
        state = agent.get_state(game)
        action = agent.get_action(state, test=True)
        reward, done, score = game.play_step(action)

    print(score)




if __name__ == '__main__':
    train(show_visuals=False)
    # test()