# API Reference

This document provides detailed API documentation for all components in the Connect 4 RL project.

## Table of Contents

- [Environment API](#environment-api)
- [Agent API](#agent-api)
- [Training API](#training-api)
- [UI API](#ui-api)

## Environment API

### Connect4Env

Main game environment implementing OpenAI Gym interface.

```python
class Connect4Env(gym.Env):
    """Connect 4 game environment following OpenAI Gym interface."""
```

#### Constructor

```python
def __init__(self, rows=6, cols=7):
    """
    Initialize Connect4 environment.

    Args:
        rows (int): Number of rows (default: 6)
        cols (int): Number of columns (default: 7)
    """
```

#### Core Methods

##### reset()

```python
def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
    """
    Reset the environment to initial state.

    Args:
        seed (int, optional): Random seed
        options (dict, optional): Additional options

    Returns:
        Tuple[np.ndarray, dict]: (initial_state, info)
    """
```

##### step()

```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """
    Execute one step in the environment.

    Args:
        action (int): Column to drop piece (0 to cols-1)

    Returns:
        Tuple containing:
            - state (np.ndarray): New board state
            - reward (float): Reward for the action
            - done (bool): Whether game is finished
            - truncated (bool): Whether episode was truncated
            - info (dict): Additional information
    """
```

##### render()

```python
def render(self, mode='ascii') -> None:
    """
    Render the current state.

    Args:
        mode (str): Rendering mode ('ascii' or 'human')
    """
```

#### Utility Methods

##### is_valid_action()

```python
def is_valid_action(self, action: int) -> bool:
    """
    Check if the action (column) is valid.

    Args:
        action (int): Column index

    Returns:
        bool: True if action is valid
    """
```

##### get_valid_actions()

```python
def get_valid_actions(self) -> List[int]:
    """
    Get list of valid actions (columns that aren't full).

    Returns:
        List[int]: Valid column indices
    """
```

##### check_winner()

```python
def check_winner(self, player: int) -> bool:
    """
    Check if the specified player has won.

    Args:
        player (int): Player number (1 or 2)

    Returns:
        bool: True if player has won
    """
```

##### is_board_full()

```python
def is_board_full(self) -> bool:
    """
    Check if the board is full.

    Returns:
        bool: True if board is full
    """
```

##### clone()

```python
def clone(self) -> 'Connect4Env':
    """
    Create a copy of the current environment state.

    Returns:
        Connect4Env: Deep copy of environment
    """
```

#### Properties

- `board` (np.ndarray): Current board state (rows × cols)
- `current_player` (int): Current player (1 or 2)
- `winner` (Optional[int]): Winner if game is done (None for draw)
- `done` (bool): Whether game is finished
- `rows` (int): Number of board rows
- `cols` (int): Number of board columns

## Agent API

### QLearningAgent

Q-learning agent with epsilon-greedy exploration.

```python
class QLearningAgent:
    """Q-Learning agent for Connect 4."""
```

#### Constructor

```python
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
```

#### Core Methods

##### choose_action()

```python
def choose_action(
    self,
    state: np.ndarray,
    valid_actions: List[int],
    training: bool = True
) -> int:
    """
    Choose action using epsilon-greedy policy.

    Args:
        state: Current board state
        valid_actions: List of valid column indices
        training: Whether agent is in training mode

    Returns:
        int: Selected action (column index)
    """
```

##### update_q_value()

```python
def update_q_value(
    self,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    next_valid_actions: List[int],
    done: bool
) -> None:
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
```

#### Q-Table Management

##### get_q_value()

```python
def get_q_value(self, state: np.ndarray, action: int) -> float:
    """
    Get Q-value for state-action pair.

    Args:
        state: Board state
        action: Action (column)

    Returns:
        float: Q-value
    """
```

##### set_q_value()

```python
def set_q_value(self, state: np.ndarray, action: int, value: float) -> None:
    """
    Set Q-value for state-action pair.

    Args:
        state: Board state
        action: Action (column)
        value: New Q-value
    """
```

#### Model Persistence

##### save_model()

```python
def save_model(self, filepath: str) -> None:
    """
    Save the Q-table and agent parameters to file.

    Args:
        filepath: Path to save model
    """
```

##### load_model()

```python
def load_model(self, filepath: str) -> None:
    """
    Load the Q-table and agent parameters from file.

    Args:
        filepath: Path to load model from
    """
```

#### Statistics and Analysis

##### update_stats()

```python
def update_stats(self, reward: float, result: str) -> None:
    """
    Update training statistics.

    Args:
        reward: Episode reward
        result: Episode result ('win', 'loss', 'draw')
    """
```

##### get_win_rate()

```python
def get_win_rate(self) -> float:
    """
    Calculate current win rate.

    Returns:
        float: Win rate (0.0 to 1.0)
    """
```

##### get_policy_info()

```python
def get_policy_info(self, state: np.ndarray, valid_actions: List[int]) -> Dict:
    """
    Get information about the agent's policy for a given state.

    Args:
        state: Board state
        valid_actions: Valid actions

    Returns:
        Dict: Policy information including Q-values and best action
    """
```

##### decay_epsilon()

```python
def decay_epsilon(self) -> None:
    """Decay exploration rate."""
```

#### Properties

- `q_table` (defaultdict): Q-value storage
- `training_stats` (dict): Training statistics
- `learning_rate` (float): Learning rate
- `discount_factor` (float): Discount factor
- `epsilon` (float): Current exploration rate
- `epsilon_min` (float): Minimum exploration rate
- `epsilon_decay` (float): Exploration decay rate

### RandomAgent

Random baseline agent for training and comparison.

```python
class RandomAgent:
    """Random agent for Connect 4 - useful as baseline and training opponent."""
```

#### Constructor

```python
def __init__(self, name: str = "Random"):
    """
    Initialize random agent.

    Args:
        name: Agent name
    """
```

#### Methods

##### choose_action()

```python
def choose_action(
    self,
    state: np.ndarray,
    valid_actions: List[int],
    training: bool = True
) -> int:
    """
    Choose a random valid action.

    Args:
        state: Current board state (ignored)
        valid_actions: List of valid actions
        training: Training mode (ignored)

    Returns:
        int: Randomly selected action
    """
```

##### No-op Methods

The following methods are provided for interface compatibility but do nothing:

- `update_q_value(*args, **kwargs)`: No-op for learning
- `decay_epsilon()`: No-op for exploration
- `update_stats(*args, **kwargs)`: No-op for statistics

## Training API

### Trainer

Training infrastructure for agent development.

```python
class Trainer:
    """Trainer class for Connect 4 RL agent."""
```

#### Constructor

```python
def __init__(self, env: Connect4Env):
    """
    Initialize trainer.

    Args:
        env: Connect4 environment
    """
```

#### Training Methods

##### train_vs_random()

```python
def train_vs_random(
    self,
    agent: QLearningAgent,
    episodes: int = 10000,
    eval_frequency: int = 1000,
    save_frequency: int = 5000,
    model_path: str = "models/q_learning_agent.pkl"
) -> QLearningAgent:
    """
    Train agent against random opponent.

    Args:
        agent: Q-learning agent to train
        episodes: Number of training episodes
        eval_frequency: How often to evaluate and log progress
        save_frequency: How often to save model
        model_path: Path to save trained model

    Returns:
        QLearningAgent: Trained agent
    """
```

##### evaluate_agent()

```python
def evaluate_agent(self, agent: QLearningAgent, games: int = 100) -> float:
    """
    Evaluate agent performance against random opponent.

    Args:
        agent: Agent to evaluate
        games: Number of evaluation games

    Returns:
        float: Win rate (0.0 to 1.0)
    """
```

##### plot_training_progress()

```python
def plot_training_progress(self, save_path: str = "training_progress.png") -> None:
    """
    Plot and save training progress.

    Args:
        save_path: Path to save plot
    """
```

#### Properties

- `env` (Connect4Env): Training environment
- `training_history` (dict): Training metrics over time

## UI API

### GameUI

Command-line interface for human vs AI gameplay.

```python
class GameUI:
    """Simple CLI interface for playing Connect 4."""
```

#### Constructor

```python
def __init__(self, env: Connect4Env):
    """
    Initialize game UI.

    Args:
        env: Connect4 environment
    """
```

#### Gameplay Methods

##### play_human_vs_agent()

```python
def play_human_vs_agent(
    self,
    agent_type: str = 'qlearning',
    model_path: str = 'models/q_learning_agent.pkl'
) -> None:
    """
    Play human vs agent game.

    Args:
        agent_type: Type of agent ('qlearning' or 'random')
        model_path: Path to trained model
    """
```

##### play_agent_vs_agent()

```python
def play_agent_vs_agent(
    self,
    agent1_type: str = 'qlearning',
    agent2_type: str = 'random',
    model_path: str = 'models/q_learning_agent.pkl',
    games: int = 10
) -> None:
    """
    Watch two agents play against each other.

    Args:
        agent1_type: First agent type
        agent2_type: Second agent type
        model_path: Path to trained model
        games: Number of games to play
    """
```

##### demonstrate_learning()

```python
def demonstrate_learning(self, model_path: str = 'models/q_learning_agent.pkl') -> None:
    """
    Demonstrate the agent's learning by showing Q-values for different states.

    Args:
        model_path: Path to trained model
    """
```

#### Utility Methods

##### load_agent()

```python
def load_agent(self, agent_type: str, model_path: Optional[str] = None) -> object:
    """
    Load an agent for gameplay.

    Args:
        agent_type: Agent type ('qlearning' or 'random')
        model_path: Path to model file

    Returns:
        object: Loaded agent
    """
```

##### get_human_move()

```python
def get_human_move(self) -> int:
    """
    Get move input from human player.

    Returns:
        int: Column index or -1 to quit
    """
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameters or arguments
- `FileNotFoundError`: Model file not found during loading
- `IndexError`: Invalid board access or action
- `KeyError`: Missing dictionary keys in saved models

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| -1 | Invalid action (out of bounds) | Check action is in range [0, cols-1] |
| -10 | Invalid move (column full) | Use `get_valid_actions()` |
| 0 | Normal operation | No action needed |

## Type Hints

The codebase uses comprehensive type hints for better IDE support and code clarity:

```python
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

# Common type aliases
State = np.ndarray  # Board state
Action = int        # Column index
Reward = float      # Reward value
```

## Thread Safety

**Note**: The current implementation is not thread-safe. For concurrent usage:

1. Create separate environment and agent instances per thread
2. Avoid sharing Q-tables between threads during training
3. Use file locking when saving/loading models concurrently

## Memory Considerations

- Q-table size grows with unique states encountered
- Typical memory usage: 100MB-1GB after full training
- State representation uses 42 integers (6×7 board)
- Consider state space reduction for memory-constrained environments

---

For usage examples and tutorials, see the main [documentation](README.md).