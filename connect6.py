import sys
import numpy as np
import random
import copy
import math
from collections import Counter, defaultdict

class Connect6:
    def __init__(self, dimension=19):
        self.dimension = dimension
        self.grid = np.zeros((dimension, dimension), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.current_player = 1  # 1: Black, 2: White
        self.is_terminated = False
        self.player_interaction = True
        self.first_move = True
        self.previous_opponent_action = None

    def initialize_grid(self):
        """Resets the grid and game state."""
        self.grid.fill(0)
        self.current_player = 1
        self.is_terminated = False
        self.player_interaction = True
        self.first_move = True
        print("= ", flush=True)

    def modify_dimension(self, dimension):
        """Updates the grid dimension and resets the game."""
        self.dimension = dimension
        self.grid = np.zeros((dimension, dimension), dtype=int)
        self.current_player = 1
        self.is_terminated = False
        print("= ", flush=True)

    def detect_victory(self):
        """Determines if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        vectors = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for row in range(self.dimension):
            for col in range(self.dimension):
                if self.grid[row, col] != 0:
                    stone_color = self.grid[row, col]
                    for delta_r, delta_c in vectors:
                        # Skip if previous position in this direction has same color (already counted)
                        prev_r, prev_c = row - delta_r, col - delta_c
                        if 0 <= prev_r < self.dimension and 0 <= prev_c < self.dimension and self.grid[prev_r, prev_c] == stone_color:
                            continue
                        
                        sequence_count = 0
                        r, c = row, col
                        while 0 <= r < self.dimension and 0 <= c < self.dimension and self.grid[r, c] == stone_color:
                            sequence_count += 1
                            r += delta_r
                            c += delta_c
                        if sequence_count >= 6:
                            return stone_color
        return 0

    def column_to_letter(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def letter_to_column(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def evaluate_stone_potential(self, row, col, color):
        """Calculates the strength of a position based on connection potential."""
        vectors = [(0, 1), (1, 0), (1, 1), (1, -1)]
        potential = 0

        for delta_r, delta_c in vectors:
            connected = 1
            r, c = row + delta_r, col + delta_c
            while 0 <= r < self.dimension and 0 <= c < self.dimension and self.grid[r, c] == color:
                connected += 1
                r += delta_r
                c += delta_c
            r, c = row - delta_r, col - delta_c
            while 0 <= r < self.dimension and 0 <= c < self.dimension and self.grid[r, c] == color:
                connected += 1
                r -= delta_r
                c -= delta_c

            if connected >= 5:
                potential += 10000
            elif connected == 4:
                potential += 5000
            elif connected == 3:
                potential += 1000
            elif connected == 2:
                potential += 100
    
        return potential

    def place_stones(self, color, move_notation):
        """Executes stone placement and updates game state."""
        if self.is_terminated:
            print("? Game over")
            return

        stones = move_notation.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.letter_to_column(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.dimension and 0 <= col < self.dimension):
                print("? Move out of board range")
                return
            if self.grid[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.grid[row, col] = 1 if color.upper() == 'B' else 2

        self.current_player = 3 - self.current_player
        if hasattr(self, '_player_interaction') and self._player_interaction:
            print(f'= ', end='', flush=True)

    def create_rule_based_move(self, color):
        """Generates a move based on strategic rules and evaluations."""
        if self.is_terminated:
            print("? Game over", flush=True)
            return

        active_color = 1 if color.upper() == 'B' else 2
        rival_color = 3 - active_color
        vacant_spots = [(r, c) for r in range(self.dimension) for c in range(self.dimension) if self.grid[r, c] == 0]

        # 1. Check for winning move
        for r, c in vacant_spots:
            self.grid[r, c] = active_color
            if self.detect_victory() == active_color:
                self.grid[r, c] = 0
                move_str = f"{self.column_to_letter(c)}{r+1}"
                self.place_stones(color, move_str)
                return
            self.grid[r, c] = 0

        # 2. Check for defensive block
        for r, c in vacant_spots:
            self.grid[r, c] = rival_color
            if self.detect_victory() == rival_color:
                self.grid[r, c] = 0
                move_str = f"{self.column_to_letter(c)}{r+1}"
                self.place_stones(color, move_str)
                return
            self.grid[r, c] = 0

        # 3. Evaluate offensive potential
        optimal_position = None
        highest_score = -1
        for r, c in vacant_spots:
            score = self.evaluate_stone_potential(r, c, active_color)
            if score > highest_score:
                highest_score = score
                optimal_position = (r, c)

        # 4. Evaluate defensive potential
        for r, c in vacant_spots:
            opponent_threat = self.evaluate_stone_potential(r, c, rival_color)
            if opponent_threat >= highest_score:
                highest_score = opponent_threat
                optimal_position = (r, c)

        # 5. Execute best identified move
        if optimal_position:
            r, c = optimal_position
            move_str = f"{self.column_to_letter(c)}{r+1}"
            self.place_stones(color, move_str)
            return

        # 6. Consider nearby moves to opponent's last play
        if self.previous_opponent_action:
            last_r, last_c = self.previous_opponent_action
            nearby_moves = [(r, c) for r in range(max(0, last_r - 2), min(self.dimension, last_r + 3))
                                    for c in range(max(0, last_c - 2), min(self.dimension, last_c + 3))
                                    if self.grid[r, c] == 0]
            if nearby_moves:
                chosen = random.choice(nearby_moves)
                move_str = f"{self.column_to_letter(chosen[1])}{chosen[0]+1}"
                self.place_stones(color, move_str)
                return

        # 7. Default to random move
        chosen = random.choice(vacant_spots)
        move_str = f"{self.column_to_letter(chosen[1])}{chosen[0]+1}"
        self.place_stones(color, move_str)

    def suggest_move(self, color):
        """Determines the next move for the AI player."""
        
        if self.is_terminated:
            print("? Game over")
            return

        grid_state = copy.deepcopy(self.grid)

        # Handle special case for first move
        if self.first_move == True and self.current_player == 1:
            vacant_spots = [(r, c) for r in range(7,12) for c in range(7,12) if self.grid[r, c] == 0]
            chosen = random.sample(vacant_spots, 1)
            move_str = ",".join(f"{self.column_to_letter(c)}{r+1}" for r, c in chosen)
            self.place_stones(color, move_str)

            print(f"{move_str}\n\n", end='', flush=True)
            print(move_str, file=sys.stderr)
        else:
            self.first_move = False
            if not hasattr(self, '_tree_search_pending_moves'):
                self._tree_search_pending_moves = []

            if not self._tree_search_pending_moves:
                # Generate the move using Monte Carlo Tree Search
                moves = monte_carlo_search(grid_state, color)
                move_str = ",".join(f"{self.column_to_letter(c)}{r+1}" for r, c in moves)
                
                self._tree_search_pending_moves = move_str.split(',')
                
            # Process the generated move
            next_move_str = ','.join(self._tree_search_pending_moves)
            self._tree_search_pending_moves = None
            self.place_stones(color, next_move_str)
            print(f"{next_move_str}", file=sys.stderr)           
            print(next_move_str, flush=True)
            print(next_move_str, file=sys.stderr)
            print(f"current color is {color}", file=sys.stderr)

    def display_grid(self):
        """Shows a text representation of the game grid."""
        print("= ")
        for row in range(self.dimension - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.grid[row, col] == 1 else "O" if self.grid[row, col] == 2 else "." for col in range(self.dimension))
            print(line)
        col_labels = "   " + " ".join(self.column_to_letter(i) for i in range(self.dimension))
        print(col_labels)
        print(flush=True)

    def show_commands(self):
        """Displays available commands."""
        print("= ", flush=True)  

    def handle_command(self, command):
        """Interprets and executes GTP protocol commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print(flush=True)
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                dimension = int(parts[1])
                self.modify_dimension(dimension)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.initialize_grid()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                print(f"received move {parts[1]} {parts[2]}", file=sys.stderr)
                self.place_stones(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.suggest_move(parts[1])
        elif cmd == "showboard":
            self.display_grid()
        elif cmd == "list_commands":
            self.show_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop for command processing from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.handle_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")


class TreeNode:
    def __init__(self, grid, played_positions=[], player=1, ancestor=None, action=None):
        self.grid = grid
        self.played_positions = played_positions
        self.player = player
        self.ancestor = ancestor
        self.action = action
        self.descendants = []
        self.visit_count = 0
        self.points = 0.0
        self.available_actions = self.identify_legal_actions()
        self.tried_actions = []

    def expansion_complete(self):
        return len(self.tried_actions) == len(self.available_actions)

    def optimal_child(self, exploration_weight=1.41):
        return max(
            self.descendants,
            key=lambda child: (child.points / child.visit_count) + exploration_weight * math.sqrt(
                2 * math.log(self.visit_count) / child.visit_count)
        )

    def grow_tree(self):
        for move in self.available_actions:
            if move not in self.tried_actions:
                print(f"expanding", file=sys.stderr)
                new_grid = copy.deepcopy(self.grid)
                new_grid[move[0]], new_grid[move[1]] = self.player, self.player
                new_played_positions = copy.deepcopy(self.played_positions)
                new_played_positions.append(move[0])
                new_played_positions.append(move[1])           
                next_player = 3 - self.player
                
                child = TreeNode(new_grid, new_played_positions, next_player, ancestor=self, action=move)
                
                self.descendants.append(child)
                self.tried_actions.append(move)
                return child
        
        print(f"nothing to expand", file=sys.stderr)
        return None

    def evaluate_position(self):
        # Perform position evaluation with rule-based heuristics
        reward = 0
        sim_grid = copy.deepcopy(self.grid)
        relevant_area = self.find_relevant_area(sim_grid)
        critical_moves = self.detect_critical_patterns(sim_grid, relevant_area, self.player)
        
        # Winning position evaluation
        if len(critical_moves) > 2:
            reward = 1000
            return reward
        
        if len(critical_moves) == 2:
           reward = 100
           return reward
       
        if len(critical_moves) == 1:
            reward += 50
            
        # Evaluate opponent threats
        opponent_threats = self.identify_threat_patterns(sim_grid, relevant_area, self.player)
        reward -= len(opponent_threats) * 5
        return reward

    def propagate_result(self, result):
        self.visit_count += 1
        self.points += result
        if self.ancestor:
            self.ancestor.propagate_result(-result)
            
    def identify_threat_patterns(self, grid, relevant_area, player):
        grid_size = grid.shape[0]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        def valid_position(r, c):
            return 0 <= r < grid_size and 0 <= c < grid_size

        def extract_line(r, c, dr, dc):
            return [
                (r + dr * i, c + dc * i)
                for i in range(-7, 7)
                if valid_position(r + dr * i, c + dc * i)
            ]

        def find_pattern(symbol_string, pattern):
            if len(symbol_string) < len(pattern):
                return []
            matches = []
            for i in range(len(symbol_string) - len(pattern) + 1):
                is_match = True
                for j in range(len(pattern)):
                    s = symbol_string[i + j]
                    p = pattern[j]
                    if (p == '1' and s != '1') or (p == 'x' and s != 'x') or (p == '0' and s != '0') or (p == 'p' and s != '0'):
                        is_match = False
                        break
                if is_match:
                    matches.append(i)
            return matches

        patterns = {
            "2_threa": (1000, ["0p1110", "0p11p0", "0111p0", "011p10", "01p110"]),
            "1_threa": (100, ["00p111x", "x111p00", "p1p1p1", "p11p1p", 
                                     "p1p11p", "1p11pp", "pp11p1", "11ppp1", "1ppp11"]),
            "placement": (10, ["pp11pp", "p1p1pp", "pp1p1p", "1p1ppp", "p1pp1p", 
                                     "p11ppp", "ppp11pp", "ppp1p1", "p1ppp1", "1ppp1p"])
        }

        move_scores = defaultdict(int)

        for (r, c) in relevant_area:
            for dr, dc in directions:
                line_segment = extract_line(r, c, dr, dc)
                values = [grid[pos] if valid_position(*pos) else -1 for pos in line_segment]
                symbols = ['1' if v == player else '0' if v == 0 else 'x' for v in values]
                symbol_string = ''.join(symbols)

                for pattern_type, (weight, pattern_list) in patterns.items():
                    for pattern in pattern_list:
                        match_indices = find_pattern(symbol_string, pattern)
                        for idx in match_indices:
                            for j, ch in enumerate(pattern):
                                if ch == 'p':
                                    threat_pos = line_segment[idx + j]
                                    if grid[threat_pos] == 0:
                                        move_scores[threat_pos] += weight

        sorted_threats = sorted(move_scores.items(), key=lambda x: -x[1])
        return [move for move, score in sorted_threats]

    def detect_critical_patterns(self, grid, relevant_area, player):
        print(f"relevant area: {relevant_area}", file=sys.stderr)
        grid_size = grid.shape[0]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        opponent = 3 - player

        def valid_position(r, c):
            return 0 <= r < grid_size and 0 <= c < grid_size

        def extract_line(r, c, dr, dc):
            return [
                (r + dr * i, c + dc * i)
                for i in range(-7, 7)
                if valid_position(r + dr * i, c + dc * i)
            ]

        def find_pattern(symbol_string, pattern):
            if len(symbol_string) < len(pattern):
                return []
            matches = []
            for i in range(len(symbol_string) - len(pattern) + 1):
                is_match = True
                for j in range(len(pattern)):
                    s = symbol_string[i + j]
                    p = pattern[j]
                    if (p == '1' and s != '1') or (p == 'x' and s != 'x') or (p == '0' and s != '0') or (p == 'p' and s != '0'):
                        is_match = False
                        break
                if is_match:
                    matches.append(i)
            return matches

        urgent_patterns = {
            "winwin": ["p11111p"],
            "win": ["p1111p"],
            "maybewin": ["01p111p10", "111p11", "11p111", "111p11", "111p10", "01p111"],
            "canwin": ["11111p", "p11111"],
        }

        optional_patterns = {
            "nice": ["111pp1", "11pp11", "1pp111", "1p11p1"],
            "maybnice": ["x1111pp", "pp1111x"]
        }

        priority_moves = set()
        option_counter = Counter()
        threat_collection = []  # Each group is one threat with its potential blocks

        for (r, c) in relevant_area:
            for dr, dc in directions:
                line_segment = extract_line(r, c, dr, dc)
                values = [grid[pos] if valid_position(*pos) else -1 for pos in line_segment]
                symbols = ['1' if v == opponent else '0' if v == 0 else 'x' for v in values]
                symbol_string = ''.join(symbols)

                # Check high-priority patterns
                for pattern_list in urgent_patterns.values():
                    for pattern in pattern_list:
                        match_indices = find_pattern(symbol_string, pattern)
                        for idx in match_indices:
                            for j, ch in enumerate(pattern):
                                if ch == 'p':
                                    block_pos = line_segment[idx + j]
                                    if grid[block_pos] == 0:
                                        priority_moves.add(block_pos)

                # Check lower-priority patterns
                for pattern_list in optional_patterns.values():
                    for pattern in pattern_list:
                        match_indices = find_pattern(symbol_string, pattern)
                        for idx in match_indices:
                            options = set()
                            for j, ch in enumerate(pattern):
                                if ch == 'p':
                                    pos = line_segment[idx + j]
                                    if grid[pos] == 0:
                                        options.add(pos)
                            if options:
                                for pos in options:
                                    option_counter[pos] += 1
                                threat_collection.append(options)

        # Add highest-voted options from each threat group
        for group in threat_collection:
            best_option = max(group, key=lambda x: option_counter[x])
            priority_moves.add(best_option)

        return list(priority_moves)

    def find_relevant_area(self, grid):
        area = set()
        grid_size = grid.shape[0]
        for (r, c) in self.played_positions:
            for dr in [-1, 0, 1, 2]:
                for dc in [-2, -1, 0, 1, 2]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        if grid[nr][nc] == 0:
                            area.add((nr, nc))
        if not area:
            area = {(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r][c] == 0}
        return list(area)

    def identify_legal_actions(self):
        grid_size = self.grid.shape[0]
        grid = self.grid
        player = self.player
        
        print(f"0", file=sys.stderr)
        print(f"current grid is {grid}", file=sys.stderr)
        print(f"current player is {player}", file=sys.stderr)
        print(f"current played position is {self.played_positions}", file=sys.stderr)
        
        relevant_area = self.find_relevant_area(grid)
        print(f"1", file=sys.stderr)
        
        # Check for immediate winning patterns
        winning_patterns = ["001111", "011110", "011111", "111100", 
                            "111010", "010111", "111001", "110011", 
                            "100111", "101101"]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for r, c in relevant_area:
            for dr, dc in directions:
                segment = [(r + dr * i, c + dc * i) for i in range(-5, 6)
                        if 0 <= r + dr * i < grid_size and 0 <= c + dc * i < grid_size]
                values = [grid[pos] if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size else -1 for pos in segment]
                symbols = ''.join(['1' if v == player else '0' if v == 0 else 'x' for v in values])
                
                for pattern in winning_patterns:
                    if pattern in symbols:
                        for i in range(len(symbols) - len(pattern) + 1):
                            if symbols[i:i+len(pattern)] == pattern:
                                winning_moves = []
                                for j, ch in enumerate(pattern):
                                    if ch == '0':
                                        pos = segment[i + j]
                                        if grid[pos] == 0:
                                            winning_moves.append(pos)
                                if len(winning_moves) >= 2:
                                    return [(winning_moves[0], winning_moves[1])]
                                
        print(f"2", file=sys.stderr)

        critical_blocks = self.detect_critical_patterns(grid, relevant_area, player)
        print(f"places to block: {critical_blocks}", file=sys.stderr)
        
        if critical_blocks:
            if len(critical_blocks) == 1:
                threats = [m for m in self.identify_threat_patterns(grid, relevant_area, player) if m != critical_blocks[0]]
                if threats:
                    return [(critical_blocks[0], threats[0])]
                else:
                    alternative = next((pos for pos in relevant_area if pos != critical_blocks[0] and grid[pos] == 0), None)
                    return [(critical_blocks[0], alternative)] if alternative else []
            else:
                return [(critical_blocks[0], critical_blocks[1])]
                
        print(f"3", file=sys.stderr)

        offensive_options = self.identify_threat_patterns(grid, relevant_area, player)
        print(f"places to attack: {offensive_options}", file=sys.stderr)
        
        if offensive_options:
            if len(offensive_options) == 1:
                alternative = next((pos for pos in relevant_area if pos != offensive_options[0] and grid[pos] == 0), None)
                return [(offensive_options[0], alternative)]
            elif len(offensive_options) == 2:
                return [(offensive_options[0], offensive_options[1])]
            else:
                # Focus on top 3 threats
                offensive_options = offensive_options[:3]
                return [(offensive_options[0], offensive_options[1]), 
                        (offensive_options[0], offensive_options[2]), 
                        (offensive_options[1], offensive_options[2])]
        
        print(f"4", file=sys.stderr)
        
        # Evaluate positions by proximity to existing stones
        def strategic_score(pos):
            r, c = pos
            friendly = sum(1 for dr in range(-1, 2) for dc in range(-1, 2)
                         if 0 <= r+dr < grid_size and 0 <= c+dc < grid_size and grid[r+dr, c+dc] == player)
            enemy = sum(1 for dr in range(-1, 2) for dc in range(-1, 2)
                       if 0 <= r+dr < grid_size and 0 <= c+dc < grid_size and grid[r+dr, c+dc] == 3-player)
            return friendly - enemy
            
        prioritized_positions = sorted(relevant_area, key=strategic_score, reverse=True)[:3]
        
        return [(prioritized_positions[i], prioritized_positions[j]) 
                for i in range(len(prioritized_positions)) 
                for j in range(i+1, len(prioritized_positions))]


def monte_carlo_search(grid, color):
    player_id = 1 if color == 'B' else 2
        
    stone_positions = []
    # Identify placed stones
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r][c] != 0:
                stone_positions.append((r, c))
    
    root = TreeNode(grid, stone_positions, player_id)
    # Quick return if only one option available
    if len(root.available_actions) == 1:
        return root.available_actions[0]

    # Run Monte Carlo Tree Search iterations
    for iteration in range(100):
        print(f"iteration {iteration}", file=sys.stderr)
        node = root
        
        # Selection phase - navigate to unexpanded node
        while node.expansion_complete() and node.descendants:
            node = node.optimal_child()
            
        # Expansion phase - add a new node to the tree
        if not node.expansion_complete():
            node = node.grow_tree()
            
        # Simulation phase - evaluate the position
        position_value = node.evaluate_position()
        
        # Backpropagation phase - update statistics up the tree
        node.propagate_result(position_value)

    # Choose the most explored move
    best_action = max(root.descendants, key=lambda n: n.visit_count).action
    
    # Log move distribution statistics
    visit_counts = [child.visit_count for child in root.descendants]
    total_visits = sum(visit_counts)
    distribution = [v / total_visits for v in visit_counts]
    print(f"move distribution: {distribution}", file=sys.stderr)
    
    return best_action


if __name__ == "__main__":
    game = Connect6()
    game.run()