import copy
import random
import math
import numpy as np
from collections import defaultdict
import gym
from gym import spaces
import matplotlib.pyplot as plt
import os
from Game2048Env import Game2048Env

# Rotate Utility
def rot90(pattern,board_size):
    new_pattern = []
    for (x,y) in pattern:
        new_pattern.append((y, board_size -x -1))
    return new_pattern
def rot180(pattern,board_size):
    return rot90(rot90(pattern,board_size),board_size)
def rot270(pattern,board_size):
    return rot90(rot180(pattern,board_size),board_size)
def flip_diag(pattern,board_size):
    new_pattern = []
    for (x,y) in pattern:
        new_pattern.append((y,x))
    return new_pattern
def flip_vert(pattern,board_size):
    new_pattern = []
    for (x,y) in pattern:
        new_pattern.append((x,board_size-y-1))
    return new_pattern
def flip_hor(pattern,board_size):
    new_pattern = []
    for (x,y) in pattern:
        new_pattern.append((board_size-x-1,y))
    return new_pattern
def flip_antidiag(pattern,board_size):
    new_pattern = []
    for (x,y) in pattern:
        new_pattern.append((board_size-y-1,board_size-x-1))
    return new_pattern



import copy
import random
import math
import numpy as np
from collections import defaultdict
import gym
from gym import spaces
import matplotlib.pyplot as plt
import os

# Game environment and color mapping code remain the same

class NTupleApproximator:
    def __init__(self, board_size, patterns, syms):
        self.patterns = patterns
        self.board_size = board_size
        self.weights = [defaultdict(float) for _ in range(len(patterns))]
        self.symmetry_patterns = syms
        
    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log2(tile))
            
    def get_features(self, board, coords):
        feature = tuple(self.tile_to_index(board[x, y]) for x, y in coords)
        return feature
        
    def value(self, board):
        total = 0
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_features(board, pattern)
            total += self.weights[i // 8][feature]
        return total
        
    def update(self, board, delta, alpha):
        """
        Update the weights of the n-tuple network.
        
        Args:
            board: The board state (afterstate)
            delta: TD error
            alpha: Learning rate
        """
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_features(board, pattern)
            self.weights[i //8 ][feature] += alpha * delta
            
    def save_model(self, filename):
        """
        Save the trained model (weights) to a file.
        
        Args:
            filename: Path to save the model
        """
        import pickle
        
        # Create a dictionary with all necessary data to reconstruct the model
        model_data = {
            'board_size': self.board_size,
            'patterns': self.patterns,
            'weights': self.weights,
            # We don't save symmetry_patterns as they can be regenerated
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename, symmetry_patterns=None):
        """
        Load a trained model from a file.
        
        Args:
            filename: Path to the saved model
            symmetry_patterns: Optional pre-generated symmetry patterns
            
        Returns:
            NTupleApproximator: Loaded model
        """
        import pickle
        
        # Load the model data
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract model parameters
        board_size = model_data['board_size']
        patterns = model_data['patterns']
        weights = model_data['weights']
        
        # If symmetry patterns not provided, regenerate them
        if symmetry_patterns is None:
            symmetry_patterns = set()
            symmetry_patterns = generate_symmetry(patterns, symmetry_patterns)
        
        # Create a new model instance
        model = cls(board_size, patterns, symmetry_patterns)
        model.weights = [defaultdict(float) for _ in range(len(patterns))]

        # Load the weights
        for i in range(len(weights)):
            model.weights[i].update(weights[i])
        
        print(f"Model loaded from {filename}")
        return model
    
    def evaluate(self, board, score):
        """
        Evaluate all possible actions from the current board state
        and return the best action and its value.
        """
        tmp_env = Game2048Env()
        tmp_env.board = board.copy()
        tmp_env.score = score
        
        legal_moves = [a for a in range(tmp_env.action_space.n) if tmp_env.is_move_legal(a)]
        
        if not legal_moves:
            return None, 0
            
        max_value = float('-inf')
        max_action = legal_moves[0]
        
        for action in legal_moves:
            # Create a copy of the environment to simulate the move
            sim_env = copy.deepcopy(tmp_env)
            
            # Get the afterstate (state after move but before new tile)
            afterstate, after_score, _, _ = sim_env.step(action, spawn=False)
            
            # Calculate reward from this action
            reward = after_score - score
            
            # The value is the immediate reward plus the value of the afterstate
            total = reward + self.value(sim_env.board)
            
            if total > max_value:
                max_value = total
                max_action = action
                
        return max_action, max_value

def td_learning(env, approximator, num_episodes=1000, alpha=0.1, gamma=1.0):
    """
    TD Learning algorithm for training the N-Tuple approximator using afterstate values.
    
    Args:
        env: The game environment
        approximator: N-Tuple approximator
        num_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        
    Returns:
        list: Final scores for each episode
        list: Maximum tile values reached in each episode
    """
    final_scores = []
    success_flags = []
    max_tiles = []
    scores = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        score = 0
        max_tile = np.max(state)
        
        # Play one episode
        while not done:
            # Get legal moves
            legal_moves = [a for a in range(env.action_space.n) if env.is_move_legal(a)]
            
            if not legal_moves:
                break
                
            # Find best action using current approximator
            best_action = None
            best_value = float('-inf')
            best_afterstate = None
            best_reward = 0
            
            # Evaluate all legal actions and select the best one
            for action in legal_moves:
                # Create a copy of the environment
                sim_env = copy.deepcopy(env)
                
                # Get the afterstate (state after move but before new tile)
                sim_env.step(action, spawn=False)
                afterstate = sim_env.board.copy()
                reward = sim_env.score - score
                
                # Value is immediate reward plus estimated future rewards
                value = reward + approximator.value(afterstate)
                
                if value > best_value:
                    best_value = value
                    best_action = action
                    best_afterstate = afterstate
                    best_reward = reward
            
            # Take the best action in the real environment
            env.step(best_action, spawn=False)
            current_afterstate = env.board.copy()
            reward = env.score - score
            score = env.score
            
            # Add the random tile
            env.add_random_tile()
            next_state = env.board.copy()
            
            # Update max tile
            max_tile = max(max_tile, np.max(next_state))
            
            # Check if the game is over
            done = env.is_game_over()
            
            if not done:
                # Find the best next action from the new state
                next_legal_moves = [a for a in range(env.action_space.n) if env.is_move_legal(a)]
                
                if next_legal_moves:
                    next_best_action = None
                    next_best_value = float('-inf')
                    next_best_afterstate = None
                    next_best_reward = 0
                    
                    # Evaluate all legal next actions
                    for next_action in next_legal_moves:
                        # Create a copy of the environment
                        next_sim_env = copy.deepcopy(env)
                        
                        # Get the next afterstate
                        next_sim_env.step(next_action, spawn=False)
                        next_afterstate = next_sim_env.board.copy()
                        next_reward = next_sim_env.score - score
                        
                        # Value is immediate reward plus estimated future rewards
                        next_value = next_reward + approximator.value(next_afterstate)
                        
                        if next_value > next_best_value:
                            next_best_value = next_value
                            next_best_action = next_action
                            next_best_afterstate = next_afterstate
                            next_best_reward = next_reward
                    
                    # Calculate TD error based on afterstates
                    # delta = r_{t+1} + γ * V(s'_{t+1}) - V(s'_t)
                    delta = (next_best_reward + gamma * approximator.value(next_best_afterstate)) - approximator.value(current_afterstate)
                    
                    # Update approximator weights
                    approximator.update(current_afterstate, delta, alpha)
                else:
                    # If no legal next moves, use only the current reward
                    delta = 0 - approximator.value(current_afterstate)
                    approximator.update(current_afterstate, delta, alpha)
            else:
                # If game is over, the future value is zero
                delta = 0 - approximator.value(current_afterstate)
                approximator.update(current_afterstate, delta, alpha)
        
        # Record results
        print(f"Episode {episode+1}/{num_episodes} | Score: {env.score} | Max Tile: {max_tile}")
        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        max_tiles.append(max_tile)
        
        # Report progress periodically
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Max Tile: {max_tile}")
        if (episode + 1) % 1000 == 0:
            # Save model every 1000 episodes
            approximator.save_model(f"reward_2048_model_retrainlonglong_{episode+1}.pkl")
            scores.append(np.mean(final_scores[-1000:]))
            plt.plot(scores)
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.title("TD Learning Scores")
            plt.savefig("td_learning_scores.png")

    return final_scores, max_tiles

# Symmetry utility functions remain the same
# Pattern definition remains the same

# Training code
def train_2048_agent(load_from=None, save_to="2048_model.pkl", num_episodes=50000):
    # Generate symmetry patterns
    symm = set()
    sym = generate_symmetry(patterns, symm)
    print(f"Num of Symmetry patterns generated: {len(sym)}")
    
    # Initialize approximator and environment
    if load_from:
        # Load an existing model
        print(f"Loading model from {load_from}")
        approximator = NTupleApproximator.load_model(load_from, sym)
    else:
        # Create a new model
        approximator = NTupleApproximator(board_size=4, patterns=patterns, syms=sym)
    
    env = Game2048Env()
    
    # Training parameters
    alpha = 0.0025  # Learning rate from the paper
    gamma = 1.0  # Discount factor
    
    # Train the agent
    final_scores, max_tiles = td_learning(env, approximator, num_episodes=num_episodes, alpha=alpha, gamma=gamma)
    
    # Save the trained model
    if save_to:
        approximator.save_model(save_to)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(final_scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("TD Learning Scores")
    
    plt.subplot(1, 2, 2)
    plt.plot([np.mean(final_scores[max(0, i-100):i]) for i in range(1, len(final_scores)+1)])
    plt.xlabel("Episode")
    plt.ylabel("Average Score (last 100 episodes)")
    plt.title("TD Learning Moving Average Scores")
    
    plt.tight_layout()
    plt.savefig("td_learning_scores.png")
    plt.close()
    
    # Calculate success rate
    success_rate = sum(1 for tile in max_tiles if tile >= 2048) / len(max_tiles)
    print(f"Overall Success Rate: {success_rate:.4f}")
    print(f"Average Score: {np.mean(final_scores):.2f}")
    
    return approximator

def play_game(model_path, num_games=1, render=True):
    """
    Play the game using a trained model.
    
    Args:
        model_path: Path to the saved model
        num_games: Number of games to play
        render: Whether to render the game
    
    Returns:
        win_rate: Percentage of games won
        avg_score: Average score
    """
    # Generate symmetry patterns
    symm = set()
    sym = generate_symmetry(patterns, symm)
    
    # Load the model
    approximator = NTupleApproximator.load_model(model_path, sym)
    
    env = Game2048Env()
    
    wins = 0
    scores = []
    max_tiles = []
    
    for game in range(num_games):
        state = env.reset()
        if render:
            print(f"\nGame {game+1}/{num_games}")
            env.render()
        
        done = False
        while not done:
            action, _ = approximator.evaluate(env.board, env.score)
            if action is None:
                break
                
            state, score, done, _ = env.step(action)
            
            if render:
                env.render(action=action)
        
        # Record results
        scores.append(env.score)
        max_tile = np.max(env.board)
        max_tiles.append(max_tile)
        
        if max_tile >= 2048:
            wins += 1
            
        print(f"Game {game+1} - Score: {env.score}, Max Tile: {max_tile}")
    
    win_rate = wins / num_games
    avg_score = np.mean(scores)
    
    print(f"\nResults after {num_games} games:")
    print(f"Win Rate: {win_rate:.4f}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Max Tiles: {max_tiles}")
    
    return win_rate, avg_score
def generate_symmetry(patterns, sym):
    
    for pattern in patterns:
        sym.add(tuple(pattern))
        sym.add(tuple(rot90(pattern,4)))
        sym.add(tuple(rot180(pattern,4)))
        sym.add(tuple(rot270(pattern,4)))
        sym.add(tuple(flip_diag(pattern,4)))
        sym.add(tuple(flip_vert(pattern,4)))
        sym.add(tuple(flip_hor(pattern,4)))
        sym.add(tuple(flip_antidiag(pattern,4)))
    return list(sym)
patterns = [
    [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],
    [(0,1),(0,2),(1,1),(1,2),(2,1),(3,1)],
    [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1)],
    [(0,0),(0,1),(1,1),(1,2),(1,3),(2,2)],
    [(0,0),(0,1),(0,2),(1,1),(2,1),(2,2)],
    [(0,0),(0,1),(1,1),(2,1),(3,1),(3,2)],
    [(0,0),(0,1),(1,1),(2,0),(2,1),(3,1)],
    [(0,0),(0,1),(0,2),(1,0),(1,2),(2,2)]
]

# patterns = [
#     [(0, 0), (0, 1), (0, 2), (0,3), (1, 0), (1, 1)],
#     [(1, 0), (1, 1), (1, 2), (1,3), (2, 0), (2, 1)],
#     [(2, 0), (2, 1), (2, 2), (2,3), (3, 0), (3, 1)],
#     [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],
#     [(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
# ]
    
if __name__ == "__main__":
    import argparse
    
    # Define patterns (as in your original code)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train or play 2048 with N-Tuple TD Learning')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'], 
                        help='Mode: train a new model or play using an existing model')
    parser.add_argument('--load', type=str, default=None, 
                        help='Path to load a model from (for continuing training or playing)')
    parser.add_argument('--save', type=str, default='2048_model.pkl', 
                        help='Path to save the trained model')
    parser.add_argument('--episodes', type=int, default=100000, 
                        help='Number of episodes for training')
    parser.add_argument('--games', type=int, default=10, 
                        help='Number of games to play in play mode')
    parser.add_argument('--no-render', action='store_true', 
                        help='Disable rendering in play mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train the agent
        approximator = train_2048_agent(
            load_from=args.load,
            save_to=args.save,
            num_episodes=args.episodes
        )
    else:  # args.mode == 'play'
        if args.load is None:
            print("Error: Must specify a model to load in play mode")
        else:
            # Play using the trained model
            play_game(
                model_path=args.load,
                num_games=args.games,
                render=not args.no_render
            )




# # # Matzusaki patterns
# patterns=[
#     [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],
#     [(0,1),(0,2),(1,1),(1,2),(2,1),(3,1)],
#     [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1)],
#     [(0,0),(0,1),(1,1),(1,2),(1,3),(2,2)],
#     [(0,0),(0,1),(0,2),(1,1),(2,1),(2,2)],
#     [(0,0),(0,1),(1,1),(2,1),(3,1),(3,2)],
#     [(0,0),(0,1),(1,1),(2,0),(2,1),(3,1)],
#     [(0,0),(0,1),(0,2),(1,0),(1,2),(2,2)]
# ]



# symm=set()
# sym = generate_symmetry(patterns,symm)
# print("Num of Symmetry patterns generated:", len(sym))
# approximator = NTupleApproximator(board_size=4, patterns=patterns,syms=sym)

# env = Game2048Env()

# num_episodes = 1000
# alpha = 0.1
# gamma = 1.0

# final_scores = td_learning(env, approximator, num_episodes=num_episodes, alpha=alpha, gamma=gamma)
# plt.plot(final_scores)
# plt.xlabel("Episode")
# plt.ylabel("Score")
# plt.title("TD Learning Scores")
# plt.savefig("td_learning_scores.png")
