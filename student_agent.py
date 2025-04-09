# Remember to adjust your student ID in meta.xml
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
from UCTMCTS import *
from Game2048Env import Game2048Env
# 方法 1：使用模糊匹配（自動轉換連結）
url ="https://drive.google.com/file/d/1ieYr5Av-OvMGNCR3myiFhhG-y1XvSP7_/view?usp=sharing"
output = "converted_model_for_submission.pkl"
gdown.download(url, output, quiet=False, fuzzy=True)
approximator = NTupleApproximator.load_model("converted_model_for_submission.pkl")
def get_action(state, score):
    global approximator
    env = Game2048Env()
    env.board = state
    env.score = score
    # print("hello")
    def value_function(a):
        return approximator.value(a)
    # Create the MCTS agent
    iterations = 100  # Number of MCTS iterations
    exploration_constant = 0.001  # Exploration constant for UCT
    mcts = UCTMCTS(env, approximator)
    

    root_node = PlayerNode(state, score, env=env)
    for _ in range(mcts.iterations):
        mcts.run_simulation(root_node)

    best_action, distribution = mcts.best_action_distribution(root_node)
    state, score, done, _ = env.step(best_action)  # Apply the selected action
    # Play the game

        # Run MCTS to get the best action        
        # Take the best action
    return best_action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.
## main function for testing the agent
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

