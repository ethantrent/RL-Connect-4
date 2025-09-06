import numpy as np
import pickle
import random
from typing import Dict, Tuple, List, Optional
from collections import defaultdict


class QLearningAgent:
    """Q-Learning agent for Connect 4."""
    
    def __init__(
        self, 
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate of epsilon decay
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: maps (state, action) -> Q-value
        self.q_table = defaultdict(float)
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0
        }
    
    def get_state_key(self, board: np.ndarray) -> str:
        """Convert board state to string key for Q-table."""
        return str(board.flatten())
    
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair."""
        state_key = self.get_state_key(state)
        return self.q_table[(state_key, action)]
    
    def set_q_value(self, state: np.ndarray, action: int, value: float):
        """Set Q-value for state-action pair."""
        state_key = self.get_state_key(state)
        self.q_table[(state_key, action)] = value
    
    def choose_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current board state
            valid_actions: List of valid column indices
            training: Whether agent is in training mode
        
        Returns:
            Selected action (column index)
        """
        if not valid_actions:
            return random.choice(range(7))  # Fallback
        
        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(valid_actions)
        else:
            # Exploit: choose best known action
            q_values = [self.get_q_value(state, action) for action in valid_actions]
            max_q = max(q_values)
            
            # Handle ties by randomly selecting among best actions
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        next_valid_actions: List[int],
        done: bool
    ):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_valid_actions: Valid actions in next state
            done: Whether episode is finished
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state - no future rewards
            target = reward
        else:
            # Non-terminal state - estimate future rewards
            next_q_values = [self.get_q_value(next_state, a) for a in next_valid_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward + self.discount_factor * max_next_q
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (target - current_q)
        self.set_q_value(state, action, new_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_stats(self, reward: float, result: str):
        """Update training statistics."""
        self.training_stats['episodes'] += 1
        self.training_stats['total_reward'] += reward
        
        if result == 'win':
            self.training_stats['wins'] += 1
        elif result == 'loss':
            self.training_stats['losses'] += 1
        elif result == 'draw':
            self.training_stats['draws'] += 1
    
    def get_win_rate(self) -> float:
        """Calculate current win rate."""
        total_games = self.training_stats['episodes']
        if total_games == 0:
            return 0.0
        return self.training_stats['wins'] / total_games
    
    def save_model(self, filepath: str):
        """Save the Q-table and agent parameters to file."""
        model_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_stats': self.training_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the Q-table and agent parameters from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(float, model_data['q_table'])
            self.learning_rate = model_data['learning_rate']
            self.discount_factor = model_data['discount_factor']
            self.epsilon = model_data['epsilon']
            self.epsilon_min = model_data['epsilon_min']
            self.epsilon_decay = model_data['epsilon_decay']
            self.training_stats = model_data['training_stats']
            
            print(f"Model loaded from {filepath}")
        except FileNotFoundError:
            print(f"No model file found at {filepath}. Starting with fresh agent.")
    
    def get_policy_info(self, state: np.ndarray, valid_actions: List[int]) -> Dict:
        """Get information about the agent's policy for a given state."""
        q_values = {action: self.get_q_value(state, action) for action in valid_actions}
        best_action = max(q_values.keys(), key=lambda k: q_values[k]) if q_values else None
        
        return {
            'q_values': q_values,
            'best_action': best_action,
            'epsilon': self.epsilon,
            'state_visits': len([k for k in self.q_table.keys() if k[0] == self.get_state_key(state)])
        }