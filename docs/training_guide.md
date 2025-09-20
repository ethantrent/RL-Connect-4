# Connect 4 RL Training Guide

## Overview

This guide provides comprehensive information about training Connect 4 reinforcement learning agents, including hyperparameter tuning, training strategies, and performance optimization.

## Table of Contents

- [Training Process](#training-process)
- [Hyperparameter Guide](#hyperparameter-guide)
- [Training Strategies](#training-strategies)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Advanced Techniques](#advanced-techniques)

## Training Process

### Basic Training Flow

```python
# 1. Create environment and agent
env = Connect4Env()
agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995
)

# 2. Create trainer
trainer = Trainer(env)

# 3. Train agent
trained_agent = trainer.train_vs_random(
    agent=agent,
    episodes=50000,
    eval_frequency=2000,
    save_frequency=10000
)
```

### Training Phases

1. **Exploration Phase** (Episodes 0-10,000)
   - High epsilon (1.0 → 0.1)
   - Agent explores random moves
   - Q-table grows rapidly
   - Performance improves quickly

2. **Learning Phase** (Episodes 10,000-30,000)
   - Medium epsilon (0.1 → 0.03)
   - Agent balances exploration/exploitation
   - Strategic patterns emerge
   - Win rate stabilizes

3. **Refinement Phase** (Episodes 30,000+)
   - Low epsilon (0.03 → 0.01)
   - Agent fine-tunes strategy
   - Marginal improvements
   - Q-values converge

## Hyperparameter Guide

### Learning Rate (α)

Controls how quickly the agent updates Q-values.

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.01 | Slow, stable learning | Fine-tuning pre-trained models |
| 0.1 | Standard rate | Most training scenarios |
| 0.3 | Fast learning | Quick experiments, simple tasks |
| 0.5+ | Very fast, potentially unstable | Research only |

**Recommendation**: Start with 0.1, reduce to 0.05 for fine-tuning.

### Discount Factor (γ)

Determines how much the agent values future rewards.

| Value | Effect | Strategic Behavior |
|-------|--------|-------------------|
| 0.9 | Moderate future focus | Balanced short/long-term |
| 0.95 | Standard setting | Good strategic planning |
| 0.99 | High future focus | Very strategic, slow initial learning |

**Recommendation**: Use 0.95 for Connect 4 (games are short enough).

### Epsilon Parameters

Control exploration vs exploitation balance.

#### Initial Epsilon
- **1.0**: Complete exploration (recommended start)
- **0.5**: Partial exploration (for continuing training)
- **0.1**: Minimal exploration (for evaluation)

#### Epsilon Minimum
- **0.01**: Standard minimum (recommended)
- **0.05**: Higher minimum for continued exploration
- **0.001**: Very low for pure exploitation

#### Epsilon Decay
- **0.999**: Slow decay (200,000+ episodes)
- **0.9995**: Standard decay (50,000 episodes)
- **0.995**: Fast decay (10,000 episodes)

### Training Duration

| Episodes | Expected Performance | Training Time | Use Case |
|----------|---------------------|---------------|----------|
| 5,000 | 45-55% win rate | 2-3 minutes | Quick test |
| 20,000 | 55-65% win rate | 8-10 minutes | Basic training |
| 50,000 | 65-75% win rate | 20-25 minutes | Standard training |
| 100,000+ | 70-80% win rate | 45+ minutes | High performance |

## Training Strategies

### 1. Standard Self-Play Training

```python
trainer.train_vs_random(
    agent=agent,
    episodes=50000,
    eval_frequency=2000,
    save_frequency=10000
)
```

**Pros**: Simple, stable, well-tested
**Cons**: Limited by random opponent skill

### 2. Curriculum Learning

Start with easier opponents, gradually increase difficulty:

```python
# Phase 1: vs Random (episodes 0-20,000)
trainer.train_vs_random(agent, episodes=20000)

# Phase 2: vs Self (episodes 20,000-40,000)
# Implement self-play training

# Phase 3: vs Strategic opponent (episodes 40,000+)
# Implement training vs rule-based opponent
```

### 3. Multi-Agent Training

Train multiple agents simultaneously:

```python
agents = [QLearningAgent() for _ in range(3)]
# Rotate opponents during training
# Best for diversity and robustness
```

### 4. Transfer Learning

Use pre-trained agent as starting point:

```python
# Load existing agent
base_agent = QLearningAgent()
base_agent.load_model("models/basic_agent.pkl")

# Continue training with new parameters
base_agent.epsilon = 0.3  # Reset exploration
trainer.train_vs_random(base_agent, episodes=20000)
```

## Performance Optimization

### Memory Optimization

1. **Q-Table Pruning**
   ```python
   # Remove low-frequency state-action pairs
   min_visits = 5
   pruned_q_table = {k: v for k, v in agent.q_table.items()
                     if visit_count[k] >= min_visits}
   ```

2. **State Space Reduction**
   - Use board symmetry (mirror states)
   - Implement state abstraction
   - Focus on relevant features

### Speed Optimization

1. **Efficient State Representation**
   ```python
   # Use hash instead of string conversion
   def get_state_hash(self, board):
       return hash(board.tobytes())
   ```

2. **Vectorized Operations**
   ```python
   # Batch Q-value updates
   # Use NumPy operations where possible
   ```

3. **Parallel Training**
   ```python
   # Multiple training processes
   # Shared Q-table with locking
   ```

### Convergence Acceleration

1. **Adaptive Learning Rate**
   ```python
   # Decrease learning rate over time
   current_lr = initial_lr * (decay_rate ** episode)
   ```

2. **Experience Replay** (Advanced)
   ```python
   # Store and replay important experiences
   # Helps with sample efficiency
   ```

3. **Target Network** (Advanced)
   ```python
   # Separate target Q-network
   # Update periodically for stability
   ```

## Troubleshooting

### Common Issues

#### 1. Agent Not Learning
**Symptoms**: Win rate stays around 50%
**Causes**:
- Learning rate too low
- Epsilon decay too fast
- Insufficient training episodes

**Solutions**:
- Increase learning rate to 0.2-0.3
- Slow down epsilon decay
- Train for more episodes

#### 2. Unstable Training
**Symptoms**: Win rate fluctuates wildly
**Causes**:
- Learning rate too high
- Poor evaluation frequency
- Random seed issues

**Solutions**:
- Reduce learning rate to 0.05
- Increase evaluation games
- Set random seeds for reproducibility

#### 3. Overfitting to Random Opponent
**Symptoms**: High win rate vs random, poor vs strategic play
**Causes**:
- Only training against random opponent
- Insufficient exploration
- Limited state space coverage

**Solutions**:
- Introduce diverse opponents
- Maintain higher epsilon longer
- Use curriculum learning

#### 4. Memory Issues
**Symptoms**: RAM usage grows excessively
**Causes**:
- Very long training runs
- No Q-table management
- Large state representations

**Solutions**:
- Implement Q-table pruning
- Use state abstraction
- Periodic garbage collection

### Performance Benchmarks

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| Win Rate vs Random | <55% | 55-65% | 65-75% | >75% |
| Q-Table Size | <10K | 10K-50K | 50K-100K | >100K |
| Training Speed | <500/sec | 500-1000/sec | 1000-2000/sec | >2000/sec |
| Memory Usage | >500MB | 100-500MB | 50-100MB | <50MB |

## Advanced Techniques

### 1. Deep Q-Networks (DQN)

Replace Q-table with neural network:

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, output_size=7):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)
```

### 2. Double DQN

Reduces overestimation bias:

```python
# Use two networks: main and target
# Update target network periodically
target_update_frequency = 1000
```

### 3. Dueling DQN

Separates value and advantage functions:

```python
class DuelingDQN(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, output_size=7):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
```

### 4. Prioritized Experience Replay

Sample important experiences more frequently:

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.priorities.append(priority)
        self.buffer.append(experience)
```

### 5. Multi-Agent Learning

Train multiple agents simultaneously:

```python
def multi_agent_training():
    agents = [QLearningAgent() for _ in range(4)]

    for episode in range(episodes):
        # Randomly pair agents
        agent1, agent2 = random.sample(agents, 2)

        # Play game and update both agents
        play_game(agent1, agent2, update_both=True)
```

## Best Practices

### 1. Experiment Tracking
- Log all hyperparameters
- Save training curves
- Version control models
- Document experiment goals

### 2. Reproducibility
```python
import random
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
```

### 3. Model Validation
- Test on multiple random seeds
- Evaluate against different opponents
- Cross-validate hyperparameters
- Monitor for overfitting

### 4. Incremental Training
- Save checkpoints frequently
- Support training resumption
- Track training time
- Monitor resource usage

### 5. Performance Monitoring
```python
# Track key metrics during training
metrics = {
    'episode': episode,
    'win_rate': current_win_rate,
    'epsilon': agent.epsilon,
    'q_table_size': len(agent.q_table),
    'avg_reward': avg_reward
}
```

---

This guide provides a comprehensive foundation for training effective Connect 4 RL agents. Experiment with different configurations to find the optimal setup for your specific goals.