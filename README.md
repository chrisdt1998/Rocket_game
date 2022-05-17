# **Rocket game with DQN**

## **Introduction:**

This repository contains a project where the a Deep Q Network is used to learn how to play a simple game, the Rocket 
game. This project also includes an implementation of the Mountain Car problem from OpenAI gym.

## **Learning goals:**

The learning goals for this project was to get a better understanding of how to make a game in pygame that could output 
the state of the game, run an iteration and also take an action as an input. 
The other main goal of this project was to learn about reinforcement learning, specifically, DQN. The DQN consists of 2 
neural networks, the main network and the target network. The idea is that the main network takes in the state as an 
input and outputs Q-values for each of the possible actions. The max Q-value is the chosen action for that state. The 
target network plays the role of stabilizing the outputs. If there is no target network, then there can be fluctuations 
or spikes in the results. 

## **Code structure:**

- rocket_game.py contains the rocket game code to be played manually by a human.
- rocket_game_AI.py contains the rocket game code for the DQN such that it can take the action as an input and output 
the resulting state.
- agent.py contains the code for the agent where one can fine-tune parameters and train the agent or test the agent on 
the best performing model.
- model.py contains the neural networks and Q learning algorithm, written in PyTorch.
- mountain_car.py is the OpenAI gym mountain car application. The idea for this file is to run it on model to debug the 
model and agent code.


## **Current Progress:**

- The rocket game has been completed. Currently, is has been simplified to only move left, right or stay where it is - 
further advancements may include shooting and moving up and down.
- The agent and model are completed. 
- The mountain car implementation is also completed. 
- The only (known) issue now is fine-tuning parameters as the model barely learns the mountain car method well and does 
not learn the rocket game properly.