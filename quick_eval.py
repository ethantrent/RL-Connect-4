#!/usr/bin/env python3
"""Quick evaluation of the trained agent."""

import sys
import os
sys.path.append('src')

from environment import Connect4Env
from agents import QLearningAgent, RandomAgent
from training.train_agent import Trainer

def quick_eval():
    """Quick evaluation of the trained agent."""
    print("=== Quick Trained Agent Evaluation ===\n")
    
    # Load environment and agent
    env = Connect4Env()
    agent = QLearningAgent()
    
    # Load model
    if os.path.exists("models/q_learning_agent.pkl"):
        agent.load_model("models/q_learning_agent.pkl")
        print("Loaded trained model")
        print(f"Episodes: {agent.training_stats.get('episodes', 0)}")
        print(f"Q-table size: {len(agent.q_table)}")
        print(f"Epsilon: {agent.epsilon:.4f}\n")
    else:
        print("No trained model found")
        return
    
    # Quick performance test (100 games)
    trainer = Trainer(env)
    print("Testing against Random (100 games)...")
    win_rate = trainer.evaluate_agent(agent, games=100)
    print(f"Win rate: {win_rate:.3f} ({win_rate*100:.1f}%)\n")
    
    # Show preferred opening moves
    state, _ = env.reset()
    valid_actions = env.get_valid_actions()
    policy_info = agent.get_policy_info(state, valid_actions)
    
    print("Agent's preferred opening moves:")
    sorted_moves = sorted(policy_info['q_values'].items(), key=lambda x: x[1], reverse=True)
    for i, (col, q_val) in enumerate(sorted_moves[:3]):
        print(f"  {i+1}. Column {col}: Q = {q_val:.4f}")
    
    # Assessment
    if win_rate >= 0.6:
        print(f"\nGOOD: Agent learned meaningful strategy!")
    elif win_rate >= 0.5:
        print(f"\nOK: Agent slightly better than random")
    else:
        print(f"\nPOOR: Agent needs more training")

if __name__ == "__main__":
    quick_eval()