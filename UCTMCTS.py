import copy
import random
import math
import numpy as np
from Game2048Env import Game2048Env
from Approximator import NTupleApproximator

# TreeSearch implementation for 2048 using TD-trained value approximator
class DecisionNode:
    def __init__(self, state, score, parent=None, action=None, env=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}  # action -> RandomNode
        self.visits = 0
        self.total_reward = 0.0
        self.legal_actions = {}  # action -> (afterstate, after_score)
        if env is not None:
            for a in range(4):
                sim = copy.deepcopy(env)
                sim.board = state.copy()
                sim.score = score
                board, new_score, done, _ = sim.step(a, spawn=False)
                if not np.array_equal(state, board):
                    self.legal_actions[a] = (board, new_score)
        
        self.untried_actions = list(self.legal_actions.keys())

    def fully_expanded(self):
        if not self.legal_actions:
            return False
        return all(action in self.children for action in self.legal_actions)
        
    def is_leaf(self):
        return not self.fully_expanded()

class RandomNode:
    def __init__(self, state, score, parent, action):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}  # (pos, val) -> DecisionNode
        self.visits = 0
        self.total_reward = 0.0
        self.expanded = False  

    def is_leaf(self):
        return not self.expanded
    
    def fully_expanded(self, empty_tiles):
        return len(self.children) == len(empty_tiles) * 2  # For 2 and 4

# Main search algorithm
class TreeSearch:
    def __init__(self, env, approximator, iterations=100, exploration_constant=0.001, rollout_depth=10, gamma=1):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        
        self.approximator = approximator
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
    
    def evaluate_best_afterstate_value(self, sim_env, approximator):
        temp_node = DecisionNode(sim_env.board.copy(), sim_env.score, env=sim_env)
        if not temp_node.legal_actions:
            return 0
        
        max_value = float('-inf')
        for a, (board, new_score) in temp_node.legal_actions.items():
            reward = new_score - sim_env.score
            v = reward + approximator.value(board)
            max_value = max(max_value, v)
        return max_value
    
    def select_child(self, node):
        # Select child using UCB formula
        best_ucb_score = -float("inf")
        best_child = None
        best_action = None
        for action, child in node.children.items():
            if child.visits == 0:
                ucb_score = self.approximator.value(child.state)
            else:
                avg_reward = child.total_reward / child.visits
                exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
                ucb_score = avg_reward + exploration
            if ucb_score > best_ucb_score:
                best_child = child
                best_action = action
                best_ucb_score = ucb_score
        return best_action, best_child
    
    def select(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        r_sum = 0
        while not node.is_leaf():

            if isinstance(node, DecisionNode):
                action, _ = self.select_child(node)
                prev_score = sim_env.score
                _, new_score, done, _ = sim_env.step(action, spawn=False)
                reward = new_score - prev_score
                r_sum += reward

                if action not in node.children:
                    node.children[action] = RandomNode(sim_env.board.copy(), new_score, parent=node, action=action)
                node = node.children[action]

            elif isinstance(node, RandomNode):
                keys = list(node.children.keys())  # key: (pos, val)
                weights = [0.9 if val == 2 else 0.1 for (_, val) in keys]
                sampled_key = random.choices(keys, weights=weights, k=1)[0]

                node = node.children[sampled_key]
                sim_env = self.create_env_from_state(node.state, node.score)
        return node, sim_env, r_sum
    
    def expand(self, node, sim_env):
        if sim_env.is_game_over():
            return node, sim_env

        if isinstance(node, DecisionNode) and not node.children:
            for action, (board, new_score) in node.legal_actions.items():
                random_node = RandomNode(board.copy(), new_score, parent=node, action=action)
                node.children[action] = random_node
  
        elif isinstance(node, RandomNode) and not node.expanded:
            self.expand_random_node(node)

    def rollout(self, node, sim_env, r_sum):
        """
        Estimate node value using the approximator
        """
        if isinstance(node, DecisionNode):
            value = self.evaluate_best_afterstate_value(sim_env, self.approximator)
        elif isinstance(node, RandomNode):
            value = self.approximator.value(node.state)
        else:
            value = 0

        value = r_sum + value
        # Normalize values
        if self.c != 0:
            self.min_value_seen = min(self.min_value_seen, value)
            self.max_value_seen = max(self.max_value_seen, value)
            if self.max_value_seen == self.min_value_seen:
                normalized_return = 0.0
            else:
                normalized_return = 2 * (value - self.min_value_seen) / (self.max_value_seen - self.min_value_seen) - 1
        else:
            normalized_return = value

        return normalized_return

    def backpropagate(self, node, reward):
        # Update stats throughout the tree
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def expand_random_node(self, node):
        empty_tiles = list(zip(*np.where(node.state == 0)))

        for pos in empty_tiles:
            for val in [2, 4]:
                new_state = node.state.copy()
                new_state[pos] = val
                key = (pos, val)
                if key not in node.children:
                    child = DecisionNode(new_state, node.score, parent=node, action=key, env=self.env)
                    node.children[key] = child

        node.expanded = True
        
    def run_simulation(self, root):
        # Selection
        node, sim_env, r_sum = self.select(root)

        # Expansion
        self.expand(node, sim_env)

        # Rollout
        reward = self.rollout(node, sim_env, r_sum)

        # Backpropagation
        self.backpropagate(node, reward)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

if __name__ == "__main__":
    # Example usage
    env = Game2048Env()
    approximator = NTupleApproximator.load_model("../converted_model.pkl")
    tree_search = TreeSearch(env, approximator)
    
    # Initialize the game environment
    state = env.reset()
    env.render()
    done = False
    score = 0

    while not done:
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        if not legal_moves:
            break

        root_node = DecisionNode(state, score, env=env)
        for _ in range(tree_search.iterations):
            tree_search.run_simulation(root_node)

        best_action, distribution = tree_search.best_action_distribution(root_node)
        state, score, done, _ = env.step(best_action)  # Apply the selected action
        print(state)