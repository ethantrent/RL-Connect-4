# Connect 4 Reinforcement Learning Documentation

## Overview

This project implements a reinforcement learning agent that learns to play Connect 4 using Q-learning. The agent starts with no knowledge of the game and improves through self-play against a random opponent.

## Table of Contents

- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Training Process](#training-process)
- [Performance Analysis](#performance-analysis)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Architecture

### Core Components

```
src/
├── environment/          # Game environment implementation
│   ├── connect4_env.py   # Connect4 game logic using OpenAI Gym interface
│   └── __init__.py
├── agents/               # Agent implementations
│   ├── q_learning_agent.py  # Q-learning agent with epsilon-greedy exploration
│   ├── random_agent.py      # Random baseline agent
│   └── __init__.py
├── training/             # Training infrastructure
│   ├── train_agent.py    # Training loops and evaluation
│   └── __init__.py
└── ui/                   # User interface
    ├── play_game.py      # Interactive CLI for human vs AI games
    └── __init__.py
```

### Design Principles

1. **Modular Architecture**: Each component has a single responsibility
2. **OpenAI Gym Interface**: Standard RL environment interface for compatibility
3. **Configurable Parameters**: Hyperparameters can be easily adjusted
4. **Comprehensive Testing**: 62 unit tests ensure code reliability
5. **Save/Load Functionality**: Models can be persisted and restored

## Getting Started

### Prerequisites

- Python 3.7+
- Virtual environment (recommended)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # source venv/bin/activate    # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

#### Train a New Agent
```bash
python src/training/train_agent.py
```

#### Play Against the AI
```bash
python src/ui/play_game.py
```

#### Run Tests
```bash
cd tests
python run_tests.py
```

## API Reference

### Environment: Connect4Env

The Connect4 environment implements the OpenAI Gym interface:

```python
from environment import Connect4Env

env = Connect4Env(rows=6, cols=7)
state, info = env.reset()
state, reward, done, truncated, info = env.step(action)
```

**Key Methods:**
- `reset()`: Initialize new game
- `step(action)`: Make a move
- `render()`: Display current board
- `get_valid_actions()`: Get available columns
- `check_winner(player)`: Check if player has won

### Agent: QLearningAgent

Q-learning agent with epsilon-greedy exploration:

```python
from agents import QLearningAgent

agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995
)

action = agent.choose_action(state, valid_actions, training=True)
agent.update_q_value(state, action, reward, next_state, next_valid_actions, done)
```

**Key Features:**
- Q-table for state-action value storage
- Epsilon-greedy exploration strategy
- Model persistence (save/load)
- Training statistics tracking
- Policy extraction for analysis

### Training: Trainer

Training infrastructure for agent development:

```python
from training.train_agent import Trainer

trainer = Trainer(env)
trained_agent = trainer.train_vs_random(
    agent=agent,
    episodes=50000,
    eval_frequency=2000,
    save_frequency=10000
)
```

## Training Process

### Algorithm: Q-Learning

The agent uses Q-learning with the following update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- `α` = learning rate (0.1)
- `γ` = discount factor (0.95)
- `r` = immediate reward
- `s,a` = current state-action pair
- `s'` = next state

### Reward Structure

- **Win**: +1
- **Loss**: -1
- **Draw**: 0
- **Invalid Move**: -10

### Training Strategy

1. **Self-Play vs Random**: Agent trains against random opponent
2. **Epsilon-Greedy**: Balances exploration vs exploitation
3. **Epsilon Decay**: Exploration rate decreases over time
4. **Player Randomization**: Agent learns from both player positions

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 0.1 | Speed of Q-value updates |
| Discount Factor | 0.95 | Weight of future rewards |
| Initial Epsilon | 1.0 | Starting exploration rate |
| Min Epsilon | 0.01 | Minimum exploration rate |
| Epsilon Decay | 0.9995 | Rate of exploration reduction |

## Performance Analysis

### Metrics

The training system tracks several metrics:

1. **Win Rate**: Percentage of games won against random opponent
2. **Average Reward**: Mean reward per episode
3. **Epsilon**: Current exploration rate
4. **Q-table Size**: Number of learned state-action pairs

### Expected Performance

After 50,000 training episodes:
- **Win Rate**: 60-80% against random opponent
- **Q-table Size**: ~50,000-100,000 state-action pairs
- **Strategy**: Preference for center columns, blocking opponent wins

### Evaluation Tools

```bash
# Quick performance check
python quick_eval.py

# Detailed performance analysis
python test_trained_agent.py

# Training progress visualization
# (Automatically generated during training)
```

## Testing

### Test Suite

The project includes comprehensive tests:

```bash
cd tests
python run_tests.py                    # Run all tests
python run_tests.py test_connect4_env  # Run specific module
```

### Test Coverage

- **Environment Tests**: Game logic, win detection, board management
- **Agent Tests**: Q-learning algorithm, action selection, persistence
- **Training Tests**: Training pipeline, evaluation, model saving
- **Integration Tests**: End-to-end scenarios

### Test Categories

1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Component interaction
3. **Performance Tests**: Algorithm correctness over time
4. **Edge Case Tests**: Boundary conditions and error handling

## Troubleshooting

### Common Issues

#### 1. Training Takes Too Long
- Reduce training episodes
- Increase epsilon decay rate
- Use smaller Q-table (reduce state space)

#### 2. Agent Performs Poorly
- Check hyperparameters
- Verify reward structure
- Increase training episodes
- Review epsilon decay schedule

#### 3. Memory Issues
- Large Q-tables can consume significant memory
- Consider state space reduction
- Implement Q-table pruning

#### 4. Import Errors
- Ensure all dependencies are installed
- Verify Python path includes src directory
- Check virtual environment activation

### Performance Optimization

1. **State Representation**: Current implementation uses full board state
2. **Memory Usage**: Q-table grows with unique states encountered
3. **Training Speed**: ~50,000 episodes takes 10-20 minutes
4. **Evaluation**: Win rate evaluation (1000 games) takes ~1 minute

### Debugging Tips

1. **Enable Verbose Logging**: Set debug flags in training
2. **Monitor Q-values**: Use `get_policy_info()` to inspect learning
3. **Visualize Training**: Check training_progress.png plots
4. **Test Components**: Run individual test modules to isolate issues

## Advanced Topics

### Extending the Agent

1. **Deep Q-Networks (DQN)**: Replace Q-table with neural network
2. **Experience Replay**: Store and replay past experiences
3. **Target Networks**: Stabilize training with separate target network
4. **Multi-Agent Training**: Train multiple agents against each other

### Alternative Algorithms

- **SARSA**: On-policy temporal difference learning
- **Policy Gradient**: Direct policy optimization
- **Actor-Critic**: Combine value and policy methods
- **Monte Carlo Tree Search**: Planning-based approach

### Performance Improvements

1. **State Space Reduction**: Use symmetry and abstraction
2. **Function Approximation**: Neural networks for large state spaces
3. **Transfer Learning**: Apply learned knowledge to similar games
4. **Parallel Training**: Multiple training processes

---

For more detailed information, see the individual component documentation in the API reference section.