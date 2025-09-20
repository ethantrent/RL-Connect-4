#!/usr/bin/env python3
"""Compare different Connect 4 RL model versions and hyperparameters."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import json

# Add src to path
sys.path.append('src')

from environment import Connect4Env
from agents import QLearningAgent
from training.train_agent import Trainer


class ModelComparison:
    """Tool for comparing different Connect 4 RL models."""

    def __init__(self):
        self.env = Connect4Env()
        self.trainer = Trainer(self.env)
        self.comparison_results = {}

    def load_model(self, model_path: str, model_name: str = None) -> QLearningAgent:
        """Load a model for comparison."""
        if not model_name:
            model_name = os.path.basename(model_path)

        agent = QLearningAgent()
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"✅ Loaded {model_name}")
            return agent
        else:
            print(f"❌ Model not found: {model_path}")
            return None

    def evaluate_model(self, agent: QLearningAgent, model_name: str, games: int = 500):
        """Evaluate a single model's performance."""
        print(f"\n📊 Evaluating {model_name}...")

        # Performance evaluation
        win_rate = self.trainer.evaluate_agent(agent, games)

        # Get model characteristics
        q_table_size = len(agent.q_table)
        training_episodes = agent.training_stats.get('episodes', 0)
        epsilon = agent.epsilon

        # Strategy analysis (opening preferences)
        empty_state, _ = self.env.reset()
        valid_actions = self.env.get_valid_actions()
        policy_info = agent.get_policy_info(empty_state, valid_actions)

        # Center preference analysis
        center_cols = [2, 3, 4]
        edge_cols = [0, 1, 5, 6]
        center_q = sum(policy_info['q_values'][col] for col in center_cols) / 3
        edge_q = sum(policy_info['q_values'][col] for col in edge_cols) / 4

        results = {
            'win_rate': win_rate,
            'q_table_size': q_table_size,
            'training_episodes': training_episodes,
            'epsilon': epsilon,
            'center_preference': center_q > edge_q,
            'center_q_avg': center_q,
            'edge_q_avg': edge_q,
            'opening_preferences': dict(policy_info['q_values'])
        }

        self.comparison_results[model_name] = results
        return results

    def head_to_head_comparison(self, agent1: QLearningAgent, agent2: QLearningAgent,
                              name1: str, name2: str, games: int = 200):
        """Compare two agents directly against each other."""
        print(f"\n⚔️  Head-to-Head: {name1} vs {name2} ({games} games)")

        agent1_wins = 0
        agent2_wins = 0
        draws = 0

        for game in range(games):
            self.env.reset()
            done = False

            # Randomly assign first player
            agent1_player = np.random.choice([1, 2])
            agent2_player = 3 - agent1_player
            current_player = 1

            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                if current_player == agent1_player:
                    action = agent1.choose_action(self.env.board.copy(), valid_actions, training=False)
                else:
                    action = agent2.choose_action(self.env.board.copy(), valid_actions, training=False)

                _, _, done, _, _ = self.env.step(action)
                current_player = 3 - current_player

            # Record result
            if self.env.winner == agent1_player:
                agent1_wins += 1
            elif self.env.winner == agent2_player:
                agent2_wins += 1
            else:
                draws += 1

        win_rate_1 = agent1_wins / games
        win_rate_2 = agent2_wins / games
        draw_rate = draws / games

        print(f"   {name1}: {agent1_wins} wins ({win_rate_1:.1%})")
        print(f"   {name2}: {agent2_wins} wins ({win_rate_2:.1%})")
        print(f"   Draws: {draws} ({draw_rate:.1%})")

        return {
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'draws': draws,
            'win_rate_1': win_rate_1,
            'win_rate_2': win_rate_2,
            'draw_rate': draw_rate
        }

    def compare_hyperparameters(self):
        """Analyze the effect of different hyperparameters."""
        print(f"\n🔬 Hyperparameter Analysis")

        if len(self.comparison_results) < 2:
            print("Need at least 2 models for hyperparameter comparison")
            return

        print(f"{'Model':<20} {'Win Rate':<10} {'Q-Table':<10} {'Episodes':<10} {'Epsilon':<8}")
        print("-" * 70)

        for name, results in self.comparison_results.items():
            print(f"{name:<20} {results['win_rate']:<10.3f} {results['q_table_size']:<10,} "
                  f"{results['training_episodes']:<10,} {results['epsilon']:<8.4f}")

    def visualize_comparison(self, save_path: str = "model_comparison.png"):
        """Create visualization comparing models."""
        if not self.comparison_results:
            print("No results to visualize")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        models = list(self.comparison_results.keys())
        win_rates = [self.comparison_results[m]['win_rate'] for m in models]
        q_table_sizes = [self.comparison_results[m]['q_table_size'] for m in models]
        episodes = [self.comparison_results[m]['training_episodes'] for m in models]
        epsilons = [self.comparison_results[m]['epsilon'] for m in models]

        # Win rates comparison
        bars1 = ax1.bar(models, win_rates, color='skyblue', alpha=0.7)
        ax1.set_title('Win Rate vs Random Opponent')
        ax1.set_ylabel('Win Rate')
        ax1.set_ylim(0, 1)
        for bar, rate in zip(bars1, win_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom')

        # Q-table sizes
        ax2.bar(models, q_table_sizes, color='lightgreen', alpha=0.7)
        ax2.set_title('Q-Table Size')
        ax2.set_ylabel('Number of State-Action Pairs')

        # Training episodes
        ax3.bar(models, episodes, color='orange', alpha=0.7)
        ax3.set_title('Training Episodes')
        ax3.set_ylabel('Episodes')

        # Current epsilon
        ax4.bar(models, epsilons, color='pink', alpha=0.7)
        ax4.set_title('Current Epsilon (Exploration Rate)')
        ax4.set_ylabel('Epsilon')

        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Comparison chart saved to {save_path}")

    def strategy_heatmap(self, save_path: str = "strategy_heatmap.png"):
        """Create heatmap of opening move preferences for each model."""
        if not self.comparison_results:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(self.comparison_results.keys())
        cols = list(range(7))

        # Create matrix of opening preferences
        heatmap_data = []
        for model in models:
            prefs = self.comparison_results[model]['opening_preferences']
            row = [prefs.get(col, 0) for col in cols]
            heatmap_data.append(row)

        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(range(7))
        ax.set_xticklabels([f'Col {i}' for i in range(7)])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Q-Value')

        # Add text annotations
        for i in range(len(models)):
            for j in range(7):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                             ha="center", va="center", color="black")

        ax.set_title('Opening Move Preferences by Model')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Strategy heatmap saved to {save_path}")

    def export_results(self, filename: str = "model_comparison_results.json"):
        """Export comparison results to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        print(f"Results exported to {filename}")

    def tournament_mode(self, model_paths: List[str], games_per_matchup: int = 100):
        """Run a tournament between multiple models."""
        print(f"\n🏆 TOURNAMENT MODE")

        agents = []
        names = []

        # Load all models
        for path in model_paths:
            name = os.path.basename(path).replace('.pkl', '')
            agent = self.load_model(path, name)
            if agent:
                agents.append(agent)
                names.append(name)

        if len(agents) < 2:
            print("Need at least 2 valid models for tournament")
            return

        print(f"Tournament with {len(agents)} agents: {', '.join(names)}")

        # Create tournament matrix
        results_matrix = np.zeros((len(agents), len(agents)))

        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    h2h = self.head_to_head_comparison(
                        agent1, agent2, names[i], names[j], games_per_matchup
                    )
                    results_matrix[i][j] = h2h['win_rate_1']

        # Calculate tournament standings
        standings = []
        for i, name in enumerate(names):
            wins = sum(results_matrix[i][j] > 0.5 for j in range(len(agents)) if i != j)
            avg_win_rate = np.mean([results_matrix[i][j] for j in range(len(agents)) if i != j])
            standings.append((name, wins, avg_win_rate))

        standings.sort(key=lambda x: (x[1], x[2]), reverse=True)

        print(f"\n🏅 TOURNAMENT STANDINGS:")
        for rank, (name, wins, avg_wr) in enumerate(standings, 1):
            print(f"   {rank}. {name}: {wins} wins, {avg_wr:.3f} avg win rate")


def main():
    """Main comparison execution."""
    comparison = ModelComparison()

    # Check for existing models
    model_dir = "models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        print(f"Found {len(model_files)} model files in {model_dir}/")

        if len(model_files) == 1:
            # Single model evaluation
            model_path = os.path.join(model_dir, model_files[0])
            agent = comparison.load_model(model_path)
            if agent:
                comparison.evaluate_model(agent, model_files[0].replace('.pkl', ''))
                comparison.compare_hyperparameters()

        elif len(model_files) > 1:
            # Multi-model comparison
            print("Multiple models found. Running comparison...")

            agents = []
            for model_file in model_files[:3]:  # Limit to first 3 models
                model_path = os.path.join(model_dir, model_file)
                agent = comparison.load_model(model_path)
                if agent:
                    name = model_file.replace('.pkl', '')
                    comparison.evaluate_model(agent, name)
                    agents.append((agent, name))

            if len(agents) >= 2:
                # Head-to-head comparison
                comparison.head_to_head_comparison(
                    agents[0][0], agents[1][0], agents[0][1], agents[1][1]
                )

            comparison.compare_hyperparameters()
            comparison.visualize_comparison()
            comparison.export_results()

    else:
        print("No models directory found. Train a model first.")


if __name__ == "__main__":
    main()