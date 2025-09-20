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

3. Train your first agent:
```bash
python src/training/train_agent.py
```
This creates a trained model in `models/q_learning_agent.pkl` (takes ~20 minutes).

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

### Basic Usage

Train a new agent:
```bash
python src/training/train_agent.py
```

Play against the trained agent:
```bash
python src/ui/play_game.py
```

### Advanced Tools

**Performance Benchmarking:**
```bash
python benchmark_agent.py                    # Comprehensive performance analysis
python performance_dashboard.py --mode check # Quick health check
python performance_dashboard.py --mode benchmark # Detailed benchmark
```

**Model Comparison:**
```bash
python compare_models.py                     # Compare multiple trained models
```

**Real-time Monitoring:**
```bash
python performance_dashboard.py --mode monitor --interval 30
```

**Testing:**
```bash
cd tests && python run_tests.py              # Run all 62 tests
python quick_eval.py                         # Quick agent evaluation
python test_trained_agent.py                 # Detailed agent testing
```