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
from UCTMCTS import MCTS, ActionNode, ChanceNode
from Game2048Env import Game2048Env
def get_action(state, score):
    # Check if the function has the approximator attribute

    if not hasattr(get_action, "approximator"):
        print("Downloading model...")
        get_action.approximator = NTupleApproximator.load_model("converted_model04_14.pkl")
    
    env = Game2048Env()
    env.board = state
    env.score = score
    
    def value_function(a):
        return get_action.approximator.value(a)
    mcts = MCTS(env, get_action.approximator, iterations = 10, explore_weight=0.0)
    root = ActionNode(env.board, score, env=env)
    for _ in range(mcts.iterations):
            mcts.simulate(root)
    # Create the MCTS agent

    best_action, distribution = mcts.best_action_distribution(root)
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