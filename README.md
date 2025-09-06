# Reinforcement Learning Connect 4 Agent

A reinforcement learning agent that learns to play Connect 4 using Q-learning and Deep Q-Networks (DQN).

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# source venv/bin/activate    # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
├── environment/          # Connect 4 game environment
├── agents/              # RL agents (Q-learning, DQN)
├── training/            # Training scripts and utilities
└── ui/                  # User interface for human vs agent play
tests/                   # Unit tests
docs/                    # Documentation
```

## Usage

Run the training script:
```bash
python src/training/train_agent.py
```

Play against the trained agent:
```bash
python src/ui/play_game.py
```