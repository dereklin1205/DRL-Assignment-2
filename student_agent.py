import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import gdown
from collections import defaultdict
from Approximator import NTupleApproximator
from UCTMCTS import TreeSearch, DecisionNode
from Game2048Env import Game2048Env

# Global variable to keep track of whether the model has been loaded
approximator = None

def get_action(state, score):
    global approximator
    
    # Load model only if it hasn't been loaded yet
    if approximator is None:
        print("Downloading model...")
        approximator = NTupleApproximator.load_model("converted_model.pkl")
    
    env = Game2048Env()
    env.board = state
    env.score = score
    
    def value_function(a):
        return approximator.value(a)
    
    # Create the MCTS agent
    mcts = TreeSearch(env, approximator)
    root_node = DecisionNode(state, score, env=env)
    for _ in range(mcts.iterations):
        mcts.run_simulation(root_node)

    best_action, distribution = mcts.best_action_distribution(root_node)
    return best_action
    
# main function for testing the agent
if __name__ == "__main__":
    env = Game2048Env()
    state = env.reset()
    done = False
    score = 0
    while not done:
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        if not legal_moves:
            break
        # Use the MCTS agent to get the best action
        action = get_action(state, score)
        
        # Apply the selected action
        state, score, done, _ = env.step(action)  
        print(state)