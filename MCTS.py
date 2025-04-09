import copy
import random
import math
import numpy as np
from collections import defaultdict
from Game2048Env import Game2048Env  # Assuming you have a Game2048Env class defined
from Approximator import NTupleApproximator  # Assuming you have a NtupleApproximator class defined

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
        if not is_chance_node:
            self.untried_actions = []
        else:
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


class UCTMCTS:
    def __init__(self, env, value_function, iterations=500, exploration_constant=0.01, vnorm=400000):
        """
        env: The 2048 game environment
        value_function: A function that takes a state and returns its value estimation
        iterations: Number of MCTS iterations to run
        exploration_constant: UCB exploration parameter
        vnorm: Value normalization constant (as described in the paper)
        """
        self.env = env
        self.value_function = value_function
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.vnorm = vnorm  # Normalization constant for value scaling

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
        
        For decision nodes, uses UCB formula.
        For chance nodes, selects randomly based on transition probabilities.
        """
        if node.is_chance_node:
            # For chance nodes, select child randomly according to probabilities
            # In 2048, new tiles have 90% chance of being a 2 and 10% chance of being a 4
            choices = list(node.children.keys())
            
            # We need to identify which children correspond to 2 and 4 tiles
            if random.random() < 0.9:
                # 90% chance: select a child representing a new '2' tile
                valid_choices = [k for k in choices if k[0] == 2]  # Adjust based on your encoding
            else:
                # 10% chance: select a child representing a new '4' tile
                valid_choices = [k for k in choices if k[0] == 4]  # Adjust based on your encoding
                
            if valid_choices:
                return random.choice(valid_choices)
        else:
            # For decision nodes, use UCB formula
            best_action = None
            best_value = -float('inf')
            
            for action, child in node.children.items():
                # UCB formula: Q + c * sqrt(ln(parent_visits)/child_visits)
                if child.visits == 0:
                    return action  # Prefer unvisited nodes
                else:
                    ucb_value = child.total_reward / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits)
                
                if ucb_value > best_value:
                    best_value = ucb_value
                    best_action = action
                    
            return best_action

    def evaluate_state(self, state, cumulative_reward):
        """
        Evaluates a state using the normalized value function.
        As per the paper, we normalize the value by dividing by vnorm.
        
        state: The state to evaluate
        cumulative_reward: The cumulative reward collected so far
        """
        state_value = self.value_function(state)
        # print(state_value)
        normalized_value = (cumulative_reward + state_value) / self.vnorm
        return normalized_value

    def backpropagate(self, node, reward):
        """
        Propagates the reward up the tree, updating visit counts and total rewards.
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
            
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
        legal_actions = [a for a in range(4) if self.env.is_move_legal(a )]
        root.set_untried_actions(legal_actions)
        
        # Run iterations
        for _ in range(self.iterations):
            # Create a simulation environment
            sim_env = self.create_env_from_state(root.state, root.score)
            
            # Run a single simulation
            self.run_simulation(root, sim_env)
            
        # Return the best action and the action distribution
        return self.best_action_distribution(root)
    
    def run_simulation(self, root, sim_env):
        """
        Runs a single MCTS simulation from the root node.
        
        As per the paper, the simulation consists of:
        1. Selection: Select a path from root to leaf using UCB.
        2. Expansion: If the leaf is not fully expanded, expand a child.
        3. Evaluation: Evaluate the leaf using the value function.
        4. Backpropagation: Update the statistics of all nodes in the path.
        """
        node = root
        cumulative_reward = 0
        root_score = node.score
        reward = node.score
        # Selection: Traverse the tree until reaching a non-fully expanded node or leaf
        while node.fully_expanded() and node.children:
            action = self.select_child(node)
            if node.is_chance_node:
                # If it's a chance node, the child is a decision node
                node = node.children[action] 
                ##node.state should add random tile
                sim_env.board = node.state.copy()
                sim_env.score = node.score
                reward = node.score
                # No reward for transitioning from afterstate to next state
            else:
                # If it's a decision node, the child is a chance node (afterstate)
                # Generate afterstate using step with spawn=False
                new_state, reward, done, _ = sim_env.step(action, spawn=False)
                node = node.children[action]
                
        cumulative_reward = reward - root_score
        # Expansion and Evaluation phase depends on the type of node
        if node.is_chance_node:
            # If chance node (afterstate), evaluate it directly
            # The afterstate value should already be computed when it was created
            # print(node.score)
            value = self.evaluate_state(node.state, cumulative_reward)
            #expand_afterstate(node, sim_env)  # Expand the afterstate node
            self.expand_afterstate(node, sim_env)  # Expand the afterstate node
        else:
            # If decision node, check if it can be expanded
            if not node.fully_expanded():
                # Expand one untried action
                ## expand all untried actions
                score = node.score
                best_value = -float('inf')
                best_node = None
                best_action = None
                curr_value = 0
                for action in node.untried_actions:
                    # Create a copy of the current state
                    sim2_env = self.create_env_from_state(node.state, node.score)
                    # Take the action in the simulation environment
                    new_state, new_score, _,_ = sim2_env.step(action, spawn=False)
                    # Create a new afterstate node
                    # Create a new node for this afterstate
                    new_node = UCTNode(new_state.copy(), new_score, node, action, is_chance_node=True)
                    # Set the new node as a child of the current node
                    node.children[action] = new_node
                    # Remove the action from untried actions
                    node.untried_actions.remove(action)  # Remove the action from untried actions
                    # new_node.set_untried_actions([a for a in range(4) if sim2_env.is_move_legal(a)])
                    curr_value = self.evaluate_state(new_node.state,  cumulative_reward + new_node.score - score) 
                    # Update the best value and action
                    if curr_value > best_value:
                        best_value = curr_value
                        best_action = action
                        best_node = new_node
                    
                    # Add this child to the current node's children
                
                # Generate afterstate using step with spawn=False
                # Update node to the newly created child
                node = best_node
                value = best_value
            # Evaluate the node
            
        
        # Backpropagation: Update the statistics of all nodes in the path
        self.backpropagate(node, value)
        
    def expand_afterstate(self, afterstate_node, sim_env):
        """
        Expands an afterstate node by generating all possible next states.
        In 2048, this means placing either a 2 (90% chance) or 4 (10% chance) in empty cells.
        
        afterstate_node: The afterstate node to expand
        sim_env: The simulation environment
        """
        empty_cells = self.get_empty_cells(afterstate_node.state) 
        for cell in empty_cells:
            # For each empty cell, we can place either a 2 or a 4
            for value, prob in [(2, 0.9), (4, 0.1)]:
                # Create a copy of the current state
                next_state = afterstate_node.state.copy()
                # Place the new tile
                row, col = cell
                next_state[row, col] = value
                
                # Create a new node for this next state
                # The key is a tuple encoding both the value and position
                key = (value, row, col)
                
                # Create the new decision node
                new_node = UCTNode(
                    next_state, 
                    afterstate_node.score, 
                    afterstate_node, 
                    key,
                    is_chance_node=False
                )
                
                sim2_env = self.create_env_from_state(next_state, afterstate_node.score)
                # Set legal actions for this new node
                legal_actions = [a for a in range(4) if sim2_env.is_move_legal(a)]
                new_node.set_untried_actions(legal_actions)
                
                # Add this next state as a child of the afterstate
                afterstate_node.children[key] = new_node
    
    def get_empty_cells(self, state):
        """
        Returns a list of empty cell coordinates in the given state.
        
        state: The game state (board) as a numpy array
        returns: List of (row, col) tuples for empty cells
        """
        # Find cells with value 0 (empty)
        empty = np.where(state == 0)
        # Convert to list of (row, col) tuples
        return list(zip(empty[0], empty[1]))
        
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
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
                
        return best_action, distribution

# Example usage with a NtupleApproximator
def play_with_mcts(env, approximator, iterations=4400, exploration_constant=0.01, vnorm=400000):
    """
    Plays a game of 2048 using MCTS with the given approximator.
    
    env: The 2048 game environment
    approximator: A NtupleApproximator with a value method
    iterations: Number of MCTS iterations per move
    exploration_constant: UCB exploration parameter
    vnorm: Normalization constant for value scaling
    """
    # Define a value function using the approximator
    def value_function(state):
        return approximator.value(state)
    
    # Create the MCTS agent
    mcts = UCTMCTS(env, value_function, iterations, exploration_constant, vnorm)
    
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
        total_reward =reward
        moves += 1
        
        # Print progress (optional)
        if moves % 10 == 0:
            print(f"Move {moves}, Current score: {total_reward}")
            print(f"Action distribution: {action_distribution}")
            # print(env.board)
        
    print(f"Game finished. Total score: {total_reward}, Moves: {moves}")
    return total_reward, state
if __name__ == "__main__":
    # Example usage of the play_with_mcts function
    env = Game2048Env()
    approximator =  NTupleApproximator.load_model("../converted_model.pkl", None) # Initialize your approximator here
    # print(approximator.patterns)
    play_with_mcts(env, approximator, iterations=100, exploration_constant=0.025, vnorm=400000)