import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional


class Connect4Env(gym.Env):
    """Connect 4 game environment following OpenAI Gym interface."""
    
    metadata = {'render_modes': ['human', 'ascii']}
    
    def __init__(self, rows=6, cols=7):
        super().__init__()
        
        self.rows = rows
        self.cols = cols
        
        # Action space: column to drop piece (0 to cols-1)
        self.action_space = spaces.Discrete(cols)
        
        # Observation space: board state (0=empty, 1=player1, 2=player2)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(rows, cols), dtype=np.int32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.current_player = 1
        self.winner = None
        self.done = False
        self.info = {}
        
        return self.board.copy(), self.info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        if self.done:
            return self.board.copy(), 0, True, False, self.info
        
        # Check if action is valid
        if not self.is_valid_action(action):
            # Invalid move - negative reward and game ends
            self.done = True
            reward = -10 if self.current_player == 1 else 10
            return self.board.copy(), reward, True, False, {'invalid_move': True}
        
        # Make the move
        self.make_move(action, self.current_player)
        
        # Check for winner
        if self.check_winner(self.current_player):
            self.winner = self.current_player
            self.done = True
            reward = 1 if self.current_player == 1 else -1
        elif self.is_board_full():
            self.done = True
            reward = 0  # Draw
        else:
            reward = 0  # Game continues
        
        # Switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
        return self.board.copy(), reward, self.done, False, self.info
    
    def is_valid_action(self, action: int) -> bool:
        """Check if the action (column) is valid."""
        if action < 0 or action >= self.cols:
            return False
        return self.board[0, action] == 0
    
    def make_move(self, col: int, player: int):
        """Drop a piece in the specified column."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                break
    
    def check_winner(self, player: int) -> bool:
        """Check if the current player has won."""
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row, col + i] == player for i in range(4)):
                    return True
        
        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(self.board[row + i, col] == player for i in range(4)):
                    return True
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return True
        
        # Check diagonal (top-right to bottom-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if all(self.board[row + i, col - i] == player for i in range(4)):
                    return True
        
        return False
    
    def is_board_full(self) -> bool:
        """Check if the board is full."""
        return np.all(self.board[0, :] != 0)
    
    def get_valid_actions(self) -> list:
        """Get list of valid actions (columns that aren't full)."""
        return [col for col in range(self.cols) if self.is_valid_action(col)]
    
    def render(self, mode='ascii'):
        """Render the current state."""
        if mode == 'ascii':
            print("\n" + "=" * (self.cols * 2 + 1))
            for row in range(self.rows):
                row_str = "|"
                for col in range(self.cols):
                    if self.board[row, col] == 0:
                        row_str += " "
                    elif self.board[row, col] == 1:
                        row_str += "X"
                    else:
                        row_str += "O"
                    row_str += "|"
                print(row_str)
            print("=" * (self.cols * 2 + 1))
            print(" " + " ".join([str(i) for i in range(self.cols)]))
            print()
    
    def clone(self):
        """Create a copy of the current environment state."""
        env_copy = Connect4Env(self.rows, self.cols)
        env_copy.board = self.board.copy()
        env_copy.current_player = self.current_player
        env_copy.winner = self.winner
        env_copy.done = self.done
        return env_copy