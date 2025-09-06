import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment import Connect4Env
from agents import QLearningAgent, RandomAgent


class Trainer:
    """Trainer class for Connect 4 RL agent."""
    
    def __init__(self, env: Connect4Env):
        self.env = env
        self.training_history = {
            'episode': [],
            'win_rate': [],
            'avg_reward': [],
            'epsilon': []
        }
    
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
            Trained agent
        """
        print(f"Starting training for {episodes} episodes...")
        print(f"Evaluation every {eval_frequency} episodes")
        
        random_opponent = RandomAgent("Random")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        for episode in range(1, episodes + 1):
            # Reset environment
            state, _ = self.env.reset()
            done = False
            
            # Randomly decide who goes first
            agent_player = np.random.choice([1, 2])
            current_player = 1
            
            episode_reward = 0
            moves = []
            
            while not done:
                valid_actions = self.env.get_valid_actions()
                
                if current_player == agent_player:
                    # Agent's turn
                    action = agent.choose_action(state, valid_actions, training=True)
                    moves.append((state.copy(), action))
                else:
                    # Random opponent's turn
                    action = random_opponent.choose_action(state, valid_actions)
                
                # Execute action
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Adjust reward from agent's perspective
                if current_player != agent_player:
                    reward = -reward  # Opponent's win is agent's loss
                
                episode_reward += reward
                
                # Update Q-values for agent's moves
                if current_player == agent_player and moves:
                    prev_state, prev_action = moves[-1]
                    next_valid_actions = self.env.get_valid_actions() if not done else []
                    agent.update_q_value(prev_state, prev_action, reward, next_state, next_valid_actions, done)
                
                state = next_state
                current_player = 3 - current_player  # Switch players
            
            # Update all moves in episode with final reward
            for i, (move_state, move_action) in enumerate(moves[:-1]):
                next_move_state = moves[i + 1][0] if i + 1 < len(moves) else state
                next_valid_actions = self.env.get_valid_actions() if i + 1 < len(moves) - 1 else []
                agent.update_q_value(move_state, move_action, episode_reward * 0.1, next_move_state, next_valid_actions, False)
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Determine result for stats
            if done and self.env.winner == agent_player:
                result = 'win'
            elif done and self.env.winner is not None:
                result = 'loss'
            else:
                result = 'draw'
            
            agent.update_stats(episode_reward, result)
            
            # Evaluation and logging
            if episode % eval_frequency == 0:
                win_rate = self.evaluate_agent(agent, games=100)
                avg_reward = agent.training_stats['total_reward'] / episode
                
                self.training_history['episode'].append(episode)
                self.training_history['win_rate'].append(win_rate)
                self.training_history['avg_reward'].append(avg_reward)
                self.training_history['epsilon'].append(agent.epsilon)
                
                print(f"Episode {episode:6d} | Win Rate: {win_rate:.3f} | "
                      f"Avg Reward: {avg_reward:.3f} | Epsilon: {agent.epsilon:.3f}")
            
            # Save model
            if episode % save_frequency == 0:
                agent.save_model(model_path)
        
        # Final save
        agent.save_model(model_path)
        print(f"\nTraining completed! Model saved to {model_path}")
        
        return agent
    
    def evaluate_agent(self, agent: QLearningAgent, games: int = 100) -> float:
        """
        Evaluate agent performance against random opponent.
        
        Args:
            agent: Agent to evaluate
            games: Number of evaluation games
        
        Returns:
            Win rate (0.0 to 1.0)
        """
        wins = 0
        random_opponent = RandomAgent("Random")
        
        for _ in range(games):
            state, _ = self.env.reset()
            done = False
            
            # Randomly decide who goes first
            agent_player = np.random.choice([1, 2])
            current_player = 1
            
            while not done:
                valid_actions = self.env.get_valid_actions()
                
                if current_player == agent_player:
                    # Agent's turn (no exploration during evaluation)
                    action = agent.choose_action(state, valid_actions, training=False)
                else:
                    # Random opponent's turn
                    action = random_opponent.choose_action(state, valid_actions)
                
                state, _, done, _, _ = self.env.step(action)
                current_player = 3 - current_player
            
            # Check if agent won
            if done and self.env.winner == agent_player:
                wins += 1
        
        return wins / games
    
    def plot_training_progress(self, save_path: str = "training_progress.png"):
        """Plot and save training progress."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        episodes = self.training_history['episode']
        
        # Win rate
        ax1.plot(episodes, self.training_history['win_rate'])
        ax1.set_title('Win Rate vs Random Opponent')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Win Rate')
        ax1.grid(True)
        
        # Average reward
        ax2.plot(episodes, self.training_history['avg_reward'])
        ax2.set_title('Average Reward per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.grid(True)
        
        # Epsilon (exploration rate)
        ax3.plot(episodes, self.training_history['epsilon'])
        ax3.set_title('Exploration Rate (Epsilon)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True)
        
        # Combined view
        ax4.plot(episodes, self.training_history['win_rate'], label='Win Rate')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(episodes, self.training_history['epsilon'], 'r--', label='Epsilon')
        ax4.set_title('Win Rate vs Exploration')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Win Rate', color='b')
        ax4_twin.set_ylabel('Epsilon', color='r')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training progress plot saved to {save_path}")


def main():
    """Main training function."""
    # Create environment
    env = Connect4Env()
    
    # Create agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995
    )
    
    # Create trainer
    trainer = Trainer(env)
    
    # Train agent
    trained_agent = trainer.train_vs_random(
        agent=agent,
        episodes=50000,
        eval_frequency=2000,
        save_frequency=10000
    )
    
    # Plot results
    trainer.plot_training_progress()
    
    # Final evaluation
    final_win_rate = trainer.evaluate_agent(trained_agent, games=1000)
    print(f"\nFinal evaluation: {final_win_rate:.3f} win rate over 1000 games")
    
    print(f"\nTraining Statistics:")
    print(f"Total Episodes: {trained_agent.training_stats['episodes']}")
    print(f"Wins: {trained_agent.training_stats['wins']}")
    print(f"Losses: {trained_agent.training_stats['losses']}")
    print(f"Draws: {trained_agent.training_stats['draws']}")
    print(f"Q-table size: {len(trained_agent.q_table)}")


if __name__ == "__main__":
    main()