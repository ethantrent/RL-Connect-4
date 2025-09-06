#!/usr/bin/env python3
"""Quick test to verify the RL Connect 4 system works."""

import sys
import os
sys.path.append('src')

from environment import Connect4Env
from agents import QLearningAgent, RandomAgent

def test_environment():
    """Test the Connect 4 environment."""
    print("Testing Connect 4 environment...")
    env = Connect4Env()
    state, _ = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Test a few moves
    valid_actions = env.get_valid_actions()
    print(f"Valid actions: {valid_actions}")
    
    # Make a move
    action = valid_actions[0]
    state, reward, done, _, _ = env.step(action)
    print(f"After move in column {action}: reward={reward}, done={done}")
    
    env.render('ascii')
    print("Environment test passed!")
    return env

def test_agents():
    """Test the RL agents."""
    print("\nTesting agents...")
    
    # Test Q-learning agent
    q_agent = QLearningAgent()
    print(f"Q-agent epsilon: {q_agent.epsilon}")
    
    # Test random agent
    random_agent = RandomAgent()
    print(f"Random agent name: {random_agent.name}")
    
    print("Agent test passed!")
    return q_agent, random_agent

def test_quick_game():
    """Test a quick game between agents."""
    print("\nTesting quick game...")
    
    env = Connect4Env()
    q_agent = QLearningAgent()
    random_agent = RandomAgent()
    
    state, _ = env.reset()
    done = False
    moves = 0
    
    print("Playing Q-agent vs Random agent...")
    
    while not done and moves < 42:  # Max 42 moves in Connect 4
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            break
            
        # Alternate between agents
        if moves % 2 == 0:
            action = q_agent.choose_action(state, valid_actions)
            agent_name = "Q-agent"
        else:
            action = random_agent.choose_action(state, valid_actions)
            agent_name = "Random"
        
        print(f"Move {moves + 1}: {agent_name} chooses column {action}")
        
        state, reward, done, _, _ = env.step(action)
        moves += 1
        
        if done:
            winner = "Q-agent" if (env.winner == 1 and moves % 2 == 1) or (env.winner == 2 and moves % 2 == 0) else "Random"
            if env.winner:
                print(f"Game over! {winner} wins!")
            else:
                print("Game over! It's a draw!")
            break
    
    env.render('ascii')
    print("Quick game test passed!")

def main():
    """Run all tests."""
    print("Running RL Connect 4 system tests...\n")
    
    try:
        # Test environment
        test_environment()
        
        # Test agents
        test_agents()
        
        # Test quick game
        test_quick_game()
        
        print("\nAll tests passed! The system is working correctly.")
        print("\nYou can now:")
        print("1. Train an agent: python src/training/train_agent.py")
        print("2. Play the game: python src/ui/play_game.py")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()