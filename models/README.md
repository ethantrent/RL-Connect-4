# Models Directory

This directory contains trained Connect 4 RL agent models.

## Model Files

Model files (`.pkl`) are **not tracked in git** due to their large size (100MB+).

### Getting Started

1. **Train a new model:**
   ```bash
   python src/training/train_agent.py
   ```
   This will create `q_learning_agent.pkl` in this directory.

2. **Download pre-trained model:**
   - Check the [Releases](https://github.com/ethantrent/RL-Connect-4/releases) page
   - Or train your own using the instructions above

### Model Files Structure

- `q_learning_agent.pkl` - Main trained Q-learning agent
- `*_backup.pkl` - Backup models from different training runs
- `experimental_*.pkl` - Models with different hyperparameters

### Model Information

A typical trained model contains:
- **Q-table**: 50,000-100,000 state-action pairs
- **Training stats**: Episodes, wins, losses, rewards
- **Hyperparameters**: Learning rate, epsilon, discount factor
- **File size**: 50-400 MB (depending on training length)

### Performance Expectations

After 50,000 training episodes:
- **Win rate vs random**: 65-75%
- **Q-table size**: ~75,000 entries
- **Training time**: ~20-25 minutes
- **Decision speed**: 1000+ decisions/second

### Using Models

```python
from agents import QLearningAgent

# Load trained model
agent = QLearningAgent()
agent.load_model("models/q_learning_agent.pkl")

# Use for gameplay
action = agent.choose_action(state, valid_actions, training=False)
```

### Model Comparison

Use the comparison tools to analyze different models:

```bash
python compare_models.py      # Compare multiple models
python benchmark_agent.py     # Detailed performance analysis
```

---

**Note**: Due to GitHub's 100MB file size limit, trained models are not included in the repository. Train your own or check releases for downloadable models.