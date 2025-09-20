#!/usr/bin/env python3
"""Test the trained agent's performance."""

import sys
import os
sys.path.append('src')

from environment import Connect4Env
from agents import QLearningAgent, RandomAgent
from training.train_agent import Trainer

def evaluate_trained_agent():
    """Evaluate the performance of the trained agent."""
    print("=== Trained Agent Performance Test ===\n")
    
    # Load environment and agents
    env = Connect4Env()
    trained_agent = QLearningAgent()
    
    # Try to load the trained model
    model_path = "models/q_learning_agent.pkl"
    if os.path.exists(model_path):
        trained_agent.load_model(model_path)
        print(f"Loaded trained model from {model_path}")
        print(f"Training stats:")
        print(f"   Episodes completed: {trained_agent.training_stats.get('episodes', 0)}")
        print(f"   Wins: {trained_agent.training_stats.get('wins', 0)}")
        print(f"   Losses: {trained_agent.training_stats.get('losses', 0)}")
        print(f"   Draws: {trained_agent.training_stats.get('draws', 0)}")
        print(f"   Q-table size: {len(trained_agent.q_table)}")
        print(f"   Current epsilon: {trained_agent.epsilon:.4f}")
    else:
        print(f"No trained model found at {model_path}")
        return
    
    # Create trainer for evaluation
    trainer = Trainer(env)
    
    print(f"\nEvaluating agent performance...")
    
    # Test against random opponent
    print("Testing against Random opponent (1000 games)...")
    win_rate_vs_random = trainer.evaluate_agent(trained_agent, games=1000)
    print(f"Win rate vs Random: {win_rate_vs_random:.3f} ({win_rate_vs_random*100:.1f}%)")
    
    # Show some example games
    print(f"\nSample game (Trained AI vs Random):")
    play_sample_game(env, trained_agent)
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if win_rate_vs_random >= 0.8:
        print("EXCELLENT - Agent has learned strong Connect 4 strategy!")
    elif win_rate_vs_random >= 0.65:
        print("GOOD - Agent shows solid strategic understanding")
    elif win_rate_vs_random >= 0.55:
        print("LEARNING - Agent has basic strategy, room for improvement")
    elif win_rate_vs_random >= 0.45:
        print("STRUGGLING - Agent performance similar to random")
    else:
        print("POOR - Agent may need retraining or parameter tuning")

def play_sample_game(env, trained_agent):
    """Play a sample game showing the trained agent in action."""
    random_opponent = RandomAgent()
    
    state, _ = env.reset()
    done = False
    moves = 0
    
    # Trained agent goes first
    agent_player = 1
    current_player = 1
    
    print("Initial board:")
    env.render('ascii')
    
    while not done and moves < 42:
        valid_actions = env.get_valid_actions()
        
        if current_player == agent_player:
            action = trained_agent.choose_action(state, valid_actions, training=False)
            print(f"Trained AI chooses column {action}")
        else:
            action = random_opponent.choose_action(state, valid_actions)
            print(f"Random opponent chooses column {action}")
        
        state, reward, done, _, _ = env.step(action)
        moves += 1
        
        if done or moves % 6 == 0:  # Show board every few moves or at end
            env.render('ascii')
        
        if done:
            if env.winner == agent_player:
                print("Trained AI wins!")
            elif env.winner:
                print("Random opponent wins!")
            else:
                print("Draw!")
            break
        
        current_player = 3 - current_player

def demonstrate_strategy():
    """Show the agent's learned strategy."""
    print("\nAgent Strategy Demonstration:")
    
    env = Connect4Env()
    agent = QLearningAgent()
    
    if os.path.exists("models/q_learning_agent.pkl"):
        agent.load_model("models/q_learning_agent.pkl")
    else:
        print("No trained model available")
        return
    
    # Empty board
    state, _ = env.reset()
    valid_actions = env.get_valid_actions()
    policy_info = agent.get_policy_info(state, valid_actions)
    
    print(f"\nEmpty board - Agent's preferred moves:")
    env.render('ascii')
    
    sorted_actions = sorted(policy_info['q_values'].items(), key=lambda x: x[1], reverse=True)
    for i, (action, q_val) in enumerate(sorted_actions[:3]):
        rank = f"{i+1}."
        print(f"   {rank} Column {action}: Q-value = {q_val:.4f}")

if __name__ == "__main__":
    evaluate_trained_agent()
    demonstrate_strategy()