import copy
import random
import math
import numpy as np
from collections import defaultdict
from Game2048Env import Game2048Env
from Approximator import NTupleApproximator

# UCT Node for MCTS with afterstate handling
class UCTNode:
    def __init__(self, state, score, parent=None, action=None, is_chance_node=False):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        is_chance_node: True if this is a chance node (afterstate), False if it's a decision node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.is_chance_node = is_chance_node
        
        # For decision nodes, we track untried actions
        self.untried_actions = []

    def set_untried_actions(self, legal_actions):
        """
        Set the list of untried actions for this node.
        """
        self.untried_actions = legal_actions.copy()

    def fully_expanded(self):
        """
        A node is fully expanded if no legal actions remain untried.
        """
        return len(self.untried_actions) == 0
        
    def is_leaf(self):
        """
        A node is a leaf if it's not fully expanded or has no children.
        """
        return not self.fully_expanded() or not self.children


class UCTMCTS:
    def __init__(self, env, value_function, iterations=500, exploration_constant=0.01):
        """
        env: The 2048 game environment
        value_function: A function that takes a state and returns its value estimation
        iterations: Number of MCTS iterations to run
        exploration_constant: UCB exploration parameter
        """
        self.env = env
        self.value_function = value_function
        self.iterations = iterations
        self.c = exploration_constant
        
        # For dynamic normalization (like in MCTS.py)
        self.min_value_seen = float('inf')
        self.max_value_seen = float('-inf')

    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        """
        Uses the UCB formula to select a child of the given node.
        """
        if node.is_chance_node:
            # For chance nodes, randomly select based on 2 (90%) or 4 (10%) probabilities
            keys = list(node.children.keys())
            weights = [0.9 if k[0] == 2 else 0.1 for k in keys]
            
            # If we have both 2 and 4 options, use weighted random choice
            if keys and sum(weights) > 0:
                return random.choices(keys, weights=weights, k=1)[0]
            elif keys:  # Fallback if weights calculation went wrong
                return random.choice(keys)
            return None  # No children
        else:
            # For decision nodes, use UCB formula
            best_action = None
            best_ucb_value = float('-inf')
            
            for action, child in node.children.items():
                if child.visits == 0:
                    # Use approximator value for unvisited nodes (similar to MCTS.py)
                    ucb_value = self.value_function(child.state)
                else:
                    # UCB formula: Q + c * sqrt(ln(parent_visits)/child_visits)
                    avg_reward = child.total_reward / child.visits
                    exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
                    ucb_value = avg_reward + exploration
                
                if ucb_value > best_ucb_value:
                    best_ucb_value = ucb_value
                    best_action = action
                    
            return best_action

    def select(self, root):
        """
        Select a path from root to leaf, tracking cumulative reward along the way.
        """
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        r_sum = 0  # Track cumulative reward along the path
        
        while not node.is_leaf():
            if not node.is_chance_node:  # Decision node
                action = self.select_child(node)
                if action is None or action not in node.children:
                    break
                    
                # Track reward from this action
                prev_score = sim_env.score
                board, new_score, done, _ = sim_env.step(action, spawn=False)
                reward = new_score - prev_score
                r_sum += reward
                
                node = node.children[action]
                
            else:  # Chance node
                action = self.select_child(node)
                if action is None:
                    break
                    
                node = node.children[action]
                # Update simulation environment to match the new state
                sim_env = self.create_env_from_state(node.state, node.score)
                
        return node, sim_env, r_sum

    def expand(self, node, sim_env):
        """
        Expand a node by adding one child if possible.
        """
        if sim_env.is_game_over():
            return  # Game over, no expansion

        if not node.is_chance_node:  # Decision node
            if not node.fully_expanded() and node.untried_actions:
                # Choose an untried action
                action = node.untried_actions.pop(0)
                
                # Apply the action to get the afterstate
                new_env = self.create_env_from_state(node.state, node.score)
                new_state, new_score, done, _ = new_env.step(action, spawn=False)
                
                # Create a new chance node
                new_node = UCTNode(new_state, new_score, node, action, is_chance_node=True)
                node.children[action] = new_node
        
        else:  # Chance node
            if not hasattr(node, 'expanded') or not node.expanded:
                # Expand all possible next states (adding 2 or 4 in empty cells)
                empty_cells = np.where(node.state == 0)
                empty_positions = list(zip(empty_cells[0], empty_cells[1]))
                
                for row, col in empty_positions:
                    for value in [2, 4]:
                        # Create a new state with the value added
                        new_state = node.state.copy()
                        new_state[row, col] = value
                        
                        # Create a key representing this tile placement
                        key = (value, row, col)
                        
                        # Create a new decision node
                        new_node = UCTNode(new_state, node.score, node, key, is_chance_node=False)
                        
                        # Set legal actions for the new node
                        new_env = self.create_env_from_state(new_state, node.score)
                        legal_actions = [a for a in range(4) if new_env.is_move_legal(a)]
                        new_node.set_untried_actions(legal_actions)
                        
                        # Add as a child
                        node.children[key] = new_node
                
                # Mark as expanded
                node.expanded = True

    def evaluate_state(self, state, r_sum):
        """
        Evaluate a state using value function and cumulative reward.
        Includes normalization similar to MCTS.py.
        """
        state_value = self.value_function(state)
        total_value = r_sum + state_value
        
        # Dynamic normalization like in MCTS.py
        if self.c != 0:
            self.min_value_seen = min(self.min_value_seen, total_value)
            self.max_value_seen = max(self.max_value_seen, total_value)
            
            if self.max_value_seen == self.min_value_seen:
                normalized_value = 0.0
            else:
                normalized_value = 2 * (total_value - self.min_value_seen) / (self.max_value_seen - self.min_value_seen) - 1
            return normalized_value
        else:
            return total_value

    def backpropagate(self, node, reward):
        """
        Propagate the reward up the tree, updating visit counts and total rewards.
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
            ###reerererqwer
    def run_mcts(self, root_state, root_score):
        """
        Runs the full MCTS process to select the best action.
        
        Returns:
        - best_action: The action with the highest visit count
        - action_distribution: The visit count distribution over actions
        """
        # Create root node
        root = UCTNode(root_state, root_score)
        
        # Set legal actions for the root node
        temp_env = self.create_env_from_state(root_state, root_score)
        legal_actions = [a for a in range(4) if temp_env.is_move_legal(a)]
        root.set_untried_actions(legal_actions)
        
        # Run iterations
        for _ in range(self.iterations):
            # Run a single simulation
            self.run_simulation(root)
            
        # Return the best action and the action distribution
        return self.best_action_distribution(root)
    
    def run_simulation(self, root):
        """
        Runs a single MCTS simulation from the root node.
        """
        # Selection phase
        node, sim_env, r_sum = self.select(root)
        
        # Expansion phase
        self.expand(node, sim_env)
        
        # Evaluation phase - evaluate the node using approximator
        value = self.evaluate_state(node.state, r_sum)
        
        # Backpropagation phase
        self.backpropagate(node, value)
        
    def best_action_distribution(self, root):
        """
        Computes the visit count distribution for each action at the root node.
        Returns the best action and the distribution.
        """
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        
        for action, child in root.children.items():
            if isinstance(action, int):  # Only consider regular actions (0-3)
                distribution[action] = child.visits / total_visits if total_visits > 0 else 0
                if child.visits > best_visits:
                    best_visits = child.visits
                    best_action = action
                    
        return best_action, distribution

# Example usage with a NtupleApproximator
def play_with_mcts(env, approximator, iterations=500, exploration_constant=0.01):
    """
    Plays a game of 2048 using the improved MCTS with the given approximator.
    """
    # Define a value function using the approximator
    def value_function(state):
        return approximator.value(state)
    
    # Create the MCTS agent
    mcts = UCTMCTS(env, value_function, iterations, exploration_constant)
    
    # Play the game
    state = env.reset()
    done = False
    total_reward = 0
    moves = 0
    
    while not done:
        # Run MCTS to get the best action
        best_action, action_distribution = mcts.run_mcts(state, env.score)
        
        # Take the best action
        state, reward, done, _ = env.step(best_action)
        total_reward = env.score
        moves += 1
        
        # Print progress (optional)
        if moves % 10 == 0:
            print(f"Move {moves}, Current score: {total_reward}")
            print(f"Action distribution: {action_distribution}")
            print(env.board)
        
    print(f"Game finished. Total score: {total_reward}, Moves: {moves}")
    return total_reward, state

if __name__ == "__main__":
    # Example usage of the play_with_mcts function
    env = Game2048Env()
    print("1")
    approximator = NTupleApproximator.load_model("converted_model_for_submission.pkl")
    play_with_mcts(env, approximator, iterations=100, exploration_constant=0.001)