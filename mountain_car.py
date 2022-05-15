import torch
import random
import numpy as np
from helper import plot
from collections import deque
from model import Linear_QNet, QTrainer
import gym
game = gym.make('MountainCar-v0')

MAX_MEMORY = 100_000
BATCH_SIZE = 100
LR = 0.0001

class Agent:
    def __init__(self, model_path=None):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9 # Discount rate, should be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)
        if model_path is None:
            self.model = Linear_QNet(22, 256, 3) # State size, hidden size and output action size
            self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        else:
            self.model = Linear_QNet(22, 256, 3)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, test=False):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon and not test:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1

        return final_move

def train(show_visuals=True):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    state_old = game.reset()
    while True:
        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        state_new, reward, done, info = game.step(final_move)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store state, action and reward
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                print("Saving model")
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

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
    # train(show_visuals=False)
    test()