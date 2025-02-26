# snakeRL

A ECE C147 project centered on Deep-Q Networks (DQNs). 

## Introduction

In value-based reinforcement learning, the goal is to find the optimal value function that can measure how good it is for an agent to be in a given state, or take a given action. Q-learning is an RL algorithm that is used to learn the quality (Q-value) of actions in various states. Finally, deep Q-networks (DQNs) replace the Q table (which stores state, action pairs in Q learning) with a neural network, in order to approximate the Q-values. 
    
    
In our project, we aim to train DQNs that can predict the best course of action in a 1v1 snake game. In particular, we hope to explore the complex interactions between the set hyperparameters, reward function, game environment, and the agents’ state and actions.

## Game Overview

We will train an agent to play a two-player snake game using Reinforcement Learning. This game will be similar to the classic snake game, set on a 2D grid, with each agent controlling a snake. Although the game runs quickly enough to appear “real-time,” it can be viewed more accurately turn-by-turn. On each turn, each player decides whether to turn their snake to the left, to the right, or to continue going straight. If a snake collides with itself or the other snake, the game will end. In this case, whichever snake is longer wins the game, regardless of which snake ended the game by collision.

Snakes move one square on each turn, with their head either turning or continuing straight, depending on their decision. Snakes start at a length of 3, and grow 1 for each tomato that they eat. When a snake moves, their “tail” will follow along their path. If the snake consumes a tomato, their tail will not move on that turn, since they are simultaneously growing as they move.
Snakes grow longer by collecting tomatoes that appear on the map. There will be two tomatoes on the map at any point in time, and snakes can collect them by moving their head onto a tomato. Once a tomato is “eaten” on collision, it will respawn in a different location in the map. 

## Goals

* As explained in the game overview, our main goal is to train an agent that can play a two-player snake game using principles from reinforcement learning. 
* In particular, we would like a clearly defined game environment (represented by a 2D grid), and agents that reach some form of convergence – we can define this as agents whose snakes grow to a length of 10. 
* If time/resources permit, we would like to train agents that are skilled at the game, potentially even beating human opponents.
* We would also like to determine exactly how well our model performs after some training. If time permits, we can assign our model an ELO rating by saving various iterations of the model at different points in training or using different hyperparameters, then evaluate them against each other based on a small sample of games. 

