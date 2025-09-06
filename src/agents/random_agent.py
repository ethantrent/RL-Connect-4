import random
from typing import List
import numpy as np


class RandomAgent:
    """Random agent for Connect 4 - useful as baseline and training opponent."""
    
    def __init__(self, name: str = "Random"):
        self.name = name
    
    def choose_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """Choose a random valid action."""
        if not valid_actions:
            return random.choice(range(7))  # Fallback
        return random.choice(valid_actions)
    
    def update_q_value(self, *args, **kwargs):
        """Random agent doesn't learn - no-op."""
        pass
    
    def decay_epsilon(self):
        """Random agent doesn't have epsilon - no-op."""
        pass
    
    def update_stats(self, *args, **kwargs):
        """Random agent doesn't track stats - no-op."""
        pass