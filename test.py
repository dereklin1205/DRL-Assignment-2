from student_agent import get_action
from Game2048Env import Game2048Env



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