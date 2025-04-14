# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math
import random
from collections import defaultdict
import gdown


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action, generate = True):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved and generate:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


class SymmetricNTupleNetwork:
    def __init__(self, tuple_shapes, num_values=15):
        self.num_values = num_values
        self.tuple_shapes = tuple_shapes
        self.symmetry_tuples = self._generate_all_symmetric_tuples(tuple_shapes)
        self.luts = [defaultdict(float) for _ in tuple_shapes]  # one shared LUT per original tuple

    def encode(self, board_value):
        if board_value == 0:
            return 0
        return min(int(np.log2(board_value)), self.num_values - 1)

    def _rotate_index(self, idx, k):
        x, y = idx
        for _ in range(k):
            x, y = y, 3 - x
        return x, y

    def _mirror_index(self, idx):
        x, y = idx
        return x, 3 - y

    def _generate_all_symmetric_tuples(self, base_tuples):
        sym_tuples = []
        for tup in base_tuples:
            for k in range(4):
                rotated = [self._rotate_index(i, k) for i in tup]
                sym_tuples.append(rotated)
                mirrored = [self._mirror_index(i) for i in rotated]
                sym_tuples.append(mirrored)
        return sym_tuples

    def get_index(self, board, tuple_indices):
        value = 0
        for idx in tuple_indices:
            x, y = idx
            value = value * self.num_values + self.encode(board[x][y])
        return value

    def value(self, board):
        val = 0
        for i, tup in enumerate(self.symmetry_tuples):
            lut_index = i // 8  # each original tuple has 8 symmetric variants
            index = self.get_index(board, tup)
            val += self.luts[lut_index][index]
        return val

    def update(self, board, target, alpha):
        value_before = self.value(board)
        # print("Value before update:", value_before)
        # print("Target:", target)
        td_error = target - value_before
        for i, tup in enumerate(self.symmetry_tuples):
            lut_index = i // 8
            index = self.get_index(board, tup)
            self.luts[lut_index][index] += alpha * td_error/len(self.symmetry_tuples)

    def save(self, path):
        serializable_luts = [dict(lut) for lut in self.luts]
        with open(path, 'wb') as f:
            pickle.dump((self.tuple_shapes, self.num_values, serializable_luts), f)

    def load(self, path):
        with open(path, 'rb') as f:
            print("start_load")
            self.tuple_shapes, self.num_values, saved_luts = pickle.load(f)
            print("tuple_shapes:", self.tuple_shapes)
            print("num_values:", self.num_values)
            # print("saved_luts:", saved_luts)
            self.symmetry_tuples = self._generate_all_symmetric_tuples(self.tuple_shapes)
            self.luts = [defaultdict(float, lut) for lut in saved_luts]



import os
# https://drive.google.com/file/d/1stLf9NqvkkmUTqvPe9dSREO0KfMAlN0s/view?usp=sharing
# Download the file only if not already downloaded
# file_id = "1stLf9NqvkkmUTqvPe9dSREO0KfMAlN0s"
# output_path = "ntuple_network.pkl"

# if not os.path.exists(output_path):
#     gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
#     print(f"Downloaded {output_path} from Google Drive.")

tuple_shapes = [
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)],
        [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (3, 1)],
        [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (1, 3)],
        [(1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (2, 2)],
        [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3)],
        #straight rows
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
    ]
approximator = SymmetricNTupleNetwork(tuple_shapes=tuple_shapes)
approximator.load("weights_compatible.pkl")
    
def get_action(state, score):
    global approximator
    print("moved")
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    td_mcts = TreeSearch(env, approximator, iterations=50)
    root = DecisionNode(state, score, env=env)

    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return 
    
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    best_action, distribution = td_mcts.best_action_distribution(root)
    return best_action # Choose a random action

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
                board, new_score, done, _ = sim.step(a, generate=False)
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
    def __init__(self, env, approximator, iterations=50, exploration_constant=0.0, rollout_depth=10, gamma=1):
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
                _, new_score, done, _ = sim_env.step(action, generate=False)
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

def play_game():
    env = Game2048Env()
    state = env.reset()
    done = False
    score = 0  
    while not done:
        action = get_action(state, score)
        state, score, done, _ = env.step(action)
        print("Action:", action)
        print("State:\n", state)

    print("Game Over! Final Score:", score)


if __name__ == "__main__":
    play_game()
    
