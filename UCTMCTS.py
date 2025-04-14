import copy
import random
import math
import numpy as np
from Game2048Env import Game2048Env
from Approximator import NTupleApproximator

# TreeSearch implementation for 2048 using TD-trained value approximator
class ActionNode:
    def __init__(self, board, score, parent=None, action=None, env=None):
        self.board = board
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}  # action -> ChanceNode
        self.visits = 0
        self.value_sum = 0.0
        self.valid_moves = {}  # action -> (next_board, next_score)
        
        # Find valid moves if environment is provided
        if env is not None:
            for move in range(4):
                sim = copy.deepcopy(env)
                sim.board = board.copy()
                sim.score = score
                next_board, next_score, done, _ = sim.step(move, spawn=False)
                if not np.array_equal(board, next_board):
                    self.valid_moves[move] = (next_board, next_score)
        
        self.unexplored_moves = list(self.valid_moves.keys())

    def is_fully_expanded(self):
        return self.valid_moves and all(move in self.children for move in self.valid_moves)
        
    def is_leaf(self):
        return not self.is_fully_expanded()

class ChanceNode:
    def __init__(self, board, score, parent, action):
        self.board = board
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}  # (position, tile_value) -> ActionNode
        self.visits = 0
        self.value_sum = 0.0
        self.expanded = False

    def is_leaf(self):
        return not self.expanded
    
    def is_fully_expanded(self, empty_tiles):
        # Each empty tile can have a 2 or 4 value
        return len(self.children) == len(empty_tiles) * 2

# Main search algorithm
class MCTS:
    def __init__(self, env, value_model, iterations=50, explore_weight=0.0):
        self.env = env
        self.iterations = iterations
        self.explore_weight = explore_weight
        
        self.value_model = value_model
        self.min_value = float('inf')
        self.max_value = float('-inf')

    def clone_env(self, board, score):
        """Creates a copy of the environment with given board and score."""
        new_env = copy.deepcopy(self.env)
        new_env.board = board.copy()
        new_env.score = score
        return new_env
    
    def get_best_afterstate_value(self, env, model):
        """Find the value of the best possible action."""
        temp_node = ActionNode(env.board.copy(), env.score, env=env)
        if not temp_node.valid_moves:
            return 0
        
        best_value = float('-inf')
        for action, (board, new_score) in temp_node.valid_moves.items():
            immediate_reward = new_score - env.score
            state_value = immediate_reward + model.value(board)
            best_value = max(best_value, state_value)
        return best_value
    
    def select_best_child(self, node):
        """Select child using UCB formula."""
        best_score = -float("inf")
        best_child = None
        best_action = None
        
        for action, child in node.children.items():
            if child.visits == 0:
                ucb_score = self.value_model.value(child.board)
            else:
                avg_value = child.value_sum / child.visits
                exploration = self.explore_weight * math.sqrt(math.log(node.visits) / child.visits)
                ucb_score = avg_value + exploration
                
            if ucb_score > best_score:
                best_child = child
                best_action = action
                best_score = ucb_score
                
        return best_action, best_child
    
    def select_node(self, root):
        """Traverse the tree to find a leaf node for expansion."""
        node = root
        sim_env = self.clone_env(node.board, node.score)
        reward_sum = 0
        
        while not node.is_leaf():
            if isinstance(node, ActionNode):
                action, next_node = self.select_best_child(node)
                prev_score = sim_env.score
                _, new_score, done, _ = sim_env.step(action, spawn=False)
                immediate_reward = new_score - prev_score
                reward_sum += immediate_reward

                if action not in node.children:
                    node.children[action] = ChanceNode(sim_env.board.copy(), new_score, parent=node, action=action)
                node = node.children[action]

            elif isinstance(node, ChanceNode):
                # Randomly sample a child based on tile probability (90% for 2, 10% for 4)
                tile_positions = list(node.children.keys())
                weights = [0.9 if val == 2 else 0.1 for (_, val) in tile_positions]
                chosen_tile = random.choices(tile_positions, weights=weights, k=1)[0]

                node = node.children[chosen_tile]
                sim_env = self.clone_env(node.board, node.score)
                
        return node, sim_env, reward_sum
    
    def expand(self, node, sim_env):
        """Expand a leaf node."""
        if sim_env.is_game_over():
            return node, sim_env

        if isinstance(node, ActionNode) and not node.children:
            # Expand ActionNode by creating ChanceNode children for each valid move
            for move, (board, new_score) in node.valid_moves.items():
                chance_node = ChanceNode(board.copy(), new_score, parent=node, action=move)
                node.children[move] = chance_node
  
        elif isinstance(node, ChanceNode) and not node.expanded:
            # Expand ChanceNode by adding possible tile spawns (2 or 4) at empty positions
            self.expand_chance_node(node)

    def evaluate(self, node, sim_env, reward_sum):
        """Estimate node value using the value model."""
        if isinstance(node, ActionNode):
            state_value = self.get_best_afterstate_value(sim_env, self.value_model)
        elif isinstance(node, ChanceNode):
            state_value = self.value_model.value(node.board)
        else:
            state_value = 0

        total_value = reward_sum + state_value
        
        # Normalize values if exploration is enabled
        if self.explore_weight != 0:
            self.min_value = min(self.min_value, total_value)
            self.max_value = max(self.max_value, total_value)
            
            if self.max_value == self.min_value:
                normalized_value = 0.0
            else:
                normalized_value = 2 * (total_value - self.min_value) / (self.max_value - self.min_value) - 1
        else:
            normalized_value = total_value

        return normalized_value

    def backpropagate(self, node, value):
        """Update statistics throughout the tree."""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def expand_chance_node(self, node):
        """Add all possible tile spawns (2 or 4) at empty positions."""
        empty_positions = list(zip(*np.where(node.board == 0)))

        for pos in empty_positions:
            for tile_value in [2, 4]:
                new_board = node.board.copy()
                new_board[pos] = tile_value
                tile_key = (pos, tile_value)
                
                if tile_key not in node.children:
                    child = ActionNode(new_board, node.score, parent=node, action=tile_key, env=self.env)
                    node.children[tile_key] = child

        node.expanded = True
        
    def simulate(self, root):
        """Run a single Monte Carlo simulation."""
        # Selection
        node, sim_env, reward_sum = self.select_node(root)

        # Expansion
        self.expand(node, sim_env)

        # Evaluation (replacing rollout)
        value = self.evaluate(node, sim_env, reward_sum)

        # Backpropagation
        self.backpropagate(node, value)

    def get_best_move(self, root):
        """Find the best move based on visit counts."""
        total_visits = sum(child.visits for child in root.children.values())
        move_distribution = np.zeros(4)
        best_visits = -1
        best_move = None
        
        for move, child in root.children.items():
            move_distribution[move] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
                
        return best_move, move_distribution

if __name__ == "__main__":
    # Example usage
    env = Game2048Env()
    value_model = NTupleApproximator.load_model("../converted_model.pkl")
    mcts = MCTS(env, value_model)
    
    # Initialize the game environment
    board = env.reset()
    env.render()
    done = False
    score = 0
    
    while not done:
        legal_moves = [move for move in range(4) if env.is_move_legal(move)]
        if not legal_moves:
            break

        root = ActionNode(board, score, env=env)
        for _ in range(mcts.iterations):
            mcts.simulate(root)

        best_move, distribution = mcts.get_best_move(root)
        board, score, done, _ = env.step(best_move)  # Apply the selected move
        print(board)
        print(score)