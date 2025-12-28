# HW3: Artificial Intelligence — Pacman Projects

Author: Chung-Yu Chang (張仲瑜)  
Student ID: 110612117

## Project Overview
This project implements classic AI algorithms in the UC Berkeley Pacman AI framework. Tasks cover Adversarial Search (Minimax, Alpha-Beta, Expectimax) and Reinforcement Learning (Value Iteration, Q-Learning).

## Methodology & Implementation

### Part 1: Adversarial Search
- Minimax Search: depth-limited minimax considering optimal moves for Pacman (Max) and ghosts (Min).
- Alpha-Beta Pruning: prunes branches that cannot affect the outcome to improve efficiency.
- Expectimax: handles stochastic ghost moves by maximizing expected value instead of worst case.

### Part 2: Reinforcement Learning
- Value Iteration: solves MDPs by iteratively updating state values from transitions and rewards.
- Q-Learning: online learning of Q-values via trial and error without an environment model.
- Approximate Q-Learning: feature-based Q-agent using state-action features (e.g., distance to closest food, ghosts 1-step away) with learned weights.

## Attribution
The Pacman AI projects were developed at UC Berkeley. Core projects and autograders were primarily created by John DeNero and Dan Klein. Reference: UC Berkeley CS188 Intro to AI.