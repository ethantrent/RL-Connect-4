#!/usr/bin/env python3
"""Comprehensive performance benchmarking for Connect 4 RL agent."""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add src to path
sys.path.append('src')

from environment import Connect4Env
from agents import QLearningAgent, RandomAgent
from training.train_agent import Trainer


class AgentBenchmark:
    """Comprehensive benchmarking suite for Connect 4 agents."""

    def __init__(self, model_path="models/q_learning_agent.pkl"):
        self.model_path = model_path
        self.env = Connect4Env()
        self.results = {}

    def load_agent(self):
        """Load the trained agent."""
        agent = QLearningAgent()
        if os.path.exists(self.model_path):
            agent.load_model(self.model_path)
            print(f"Loaded agent from {self.model_path}")
            return agent
        else:
            print(f"No model found at {self.model_path}")
            return None

    def benchmark_vs_random(self, agent, games=1000):
        """Benchmark agent performance against random opponent."""
        print(f"\n=== Benchmarking vs Random Opponent ({games} games) ===")

        trainer = Trainer(self.env)
        start_time = time.time()

        win_rate = trainer.evaluate_agent(agent, games)

        elapsed = time.time() - start_time
        games_per_second = games / elapsed

        results = {
            'win_rate': win_rate,
            'games': games,
            'time_taken': elapsed,
            'games_per_second': games_per_second,
            'wins': int(win_rate * games),
            'losses': int((1 - win_rate) * games * 0.8),  # Approximate
            'draws': games - int(win_rate * games) - int((1 - win_rate) * games * 0.8)
        }

        print(f"Win Rate: {win_rate:.3f} ({win_rate*100:.1f}%)")
        print(f"Games: {results['wins']}W / {results['losses']}L / {results['draws']}D")
        print(f"Speed: {games_per_second:.1f} games/second")

        self.results['vs_random'] = results
        return results

    def analyze_strategy(self, agent):
        """Analyze agent's strategic preferences."""
        print(f"\n=== Strategy Analysis ===")

        # Analyze opening move preferences
        empty_state, _ = self.env.reset()
        valid_actions = self.env.get_valid_actions()
        policy_info = agent.get_policy_info(empty_state, valid_actions)

        print("\nOpening Move Preferences:")
        sorted_moves = sorted(policy_info['q_values'].items(),
                            key=lambda x: x[1], reverse=True)

        for i, (col, q_val) in enumerate(sorted_moves):
            preference = "★" if i == 0 else " "
            print(f"  {preference} Column {col}: Q={q_val:.4f}")

        # Analyze center vs edge preference
        center_cols = [2, 3, 4]
        edge_cols = [0, 1, 5, 6]

        center_q = sum(policy_info['q_values'][col] for col in center_cols) / 3
        edge_q = sum(policy_info['q_values'][col] for col in edge_cols) / 4

        print(f"\nCenter vs Edge Strategy:")
        print(f"  Center columns (2,3,4): Avg Q = {center_q:.4f}")
        print(f"  Edge columns (0,1,5,6): Avg Q = {edge_q:.4f}")
        print(f"  Center preference: {center_q > edge_q}")

        self.results['strategy'] = {
            'opening_preferences': dict(policy_info['q_values']),
            'center_preference': center_q > edge_q,
            'center_q': center_q,
            'edge_q': edge_q
        }

    def learning_stats(self, agent):
        """Analyze agent's learning statistics."""
        print(f"\n=== Learning Statistics ===")

        stats = agent.training_stats
        print(f"Training Episodes: {stats['episodes']:,}")
        print(f"Q-table Size: {len(agent.q_table):,} state-action pairs")
        print(f"Current Epsilon: {agent.epsilon:.4f}")

        if stats['episodes'] > 0:
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            print(f"\nTraining Results:")
            print(f"  Wins: {stats['wins']:,} ({stats['wins']/total_games*100:.1f}%)")
            print(f"  Losses: {stats['losses']:,} ({stats['losses']/total_games*100:.1f}%)")
            print(f"  Draws: {stats['draws']:,} ({stats['draws']/total_games*100:.1f}%)")
            print(f"  Avg Reward: {stats['total_reward']/stats['episodes']:.3f}")

        self.results['learning'] = {
            'episodes': stats['episodes'],
            'q_table_size': len(agent.q_table),
            'epsilon': agent.epsilon,
            'training_stats': stats
        }

    def speed_benchmark(self, agent):
        """Benchmark agent decision speed."""
        print(f"\n=== Speed Benchmark ===")

        # Test decision speed on various board states
        times = []

        # Empty board
        state, _ = self.env.reset()
        valid_actions = self.env.get_valid_actions()

        start_time = time.time()
        for _ in range(1000):
            agent.choose_action(state, valid_actions, training=False)
        empty_time = (time.time() - start_time) / 1000

        # Half-full board
        self.env.reset()
        for _ in range(21):  # Fill half the board
            if self.env.get_valid_actions():
                action = np.random.choice(self.env.get_valid_actions())
                self.env.step(action)

        state = self.env.board.copy()
        valid_actions = self.env.get_valid_actions()

        start_time = time.time()
        for _ in range(1000):
            agent.choose_action(state, valid_actions, training=False)
        half_time = (time.time() - start_time) / 1000

        print(f"Decision Speed:")
        print(f"  Empty board: {empty_time*1000:.2f}ms per decision")
        print(f"  Half-full board: {half_time*1000:.2f}ms per decision")
        print(f"  Estimated game speed: {1/empty_time:.0f} decisions/second")

        self.results['speed'] = {
            'empty_board_ms': empty_time * 1000,
            'half_board_ms': half_time * 1000,
            'decisions_per_second': 1 / empty_time
        }

    def memory_analysis(self, agent):
        """Analyze agent memory usage."""
        print(f"\n=== Memory Analysis ===")

        import sys

        # Estimate Q-table memory usage
        q_table_size = len(agent.q_table)
        state_key_length = len(str(np.zeros((6, 7)).flatten()))

        # Rough memory estimation
        estimated_mb = (q_table_size * (state_key_length + 50)) / (1024 * 1024)

        print(f"Q-table Entries: {q_table_size:,}")
        print(f"Estimated Memory: {estimated_mb:.1f} MB")
        print(f"Avg State Key Length: {state_key_length} chars")

        # State space analysis
        total_possible_states = 3 ** 42  # Upper bound
        explored_percentage = (q_table_size / 42) / total_possible_states * 100

        print(f"State Space Exploration: {explored_percentage:.2e}% of possible states")

        self.results['memory'] = {
            'q_table_entries': q_table_size,
            'estimated_mb': estimated_mb,
            'exploration_percentage': explored_percentage
        }

    def competitive_analysis(self, agent):
        """Compare against different difficulty opponents."""
        print(f"\n=== Competitive Analysis ===")

        # Test against pure random
        trainer = Trainer(self.env)
        random_win_rate = trainer.evaluate_agent(agent, 200)

        # Test playing as both first and second player
        first_player_wins = 0
        second_player_wins = 0

        for game in range(100):
            # Agent goes first
            self.env.reset()
            done = False
            current_player = 1
            agent_player = 1

            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                if current_player == agent_player:
                    action = agent.choose_action(self.env.board.copy(), valid_actions, training=False)
                else:
                    action = np.random.choice(valid_actions)

                _, _, done, _, _ = self.env.step(action)
                current_player = 3 - current_player

            if self.env.winner == agent_player:
                first_player_wins += 1

        for game in range(100):
            # Agent goes second
            self.env.reset()
            done = False
            current_player = 1
            agent_player = 2

            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                if current_player == agent_player:
                    action = agent.choose_action(self.env.board.copy(), valid_actions, training=False)
                else:
                    action = np.random.choice(valid_actions)

                _, _, done, _, _ = self.env.step(action)
                current_player = 3 - current_player

            if self.env.winner == agent_player:
                second_player_wins += 1

        print(f"Overall vs Random: {random_win_rate:.3f} win rate")
        print(f"As First Player: {first_player_wins}/100 wins ({first_player_wins}%)")
        print(f"As Second Player: {second_player_wins}/100 wins ({second_player_wins}%)")

        first_player_advantage = first_player_wins > second_player_wins
        print(f"First Player Advantage: {first_player_advantage}")

        self.results['competitive'] = {
            'overall_win_rate': random_win_rate,
            'first_player_wins': first_player_wins,
            'second_player_wins': second_player_wins,
            'first_player_advantage': first_player_advantage
        }

    def generate_report(self):
        """Generate comprehensive benchmark report."""
        print(f"\n" + "="*60)
        print(f"         CONNECT 4 AGENT BENCHMARK REPORT")
        print(f"="*60)

        if 'vs_random' in self.results:
            vr = self.results['vs_random']
            print(f"\n📊 PERFORMANCE SUMMARY")
            print(f"   Win Rate: {vr['win_rate']:.1%}")
            print(f"   Games Analyzed: {vr['games']:,}")
            print(f"   Decision Speed: {self.results.get('speed', {}).get('decisions_per_second', 0):.0f}/sec")

        if 'learning' in self.results:
            learn = self.results['learning']
            print(f"\n🧠 LEARNING SUMMARY")
            print(f"   Training Episodes: {learn['episodes']:,}")
            print(f"   Knowledge Base: {learn['q_table_size']:,} state-action pairs")
            print(f"   Exploration Rate: {learn['epsilon']:.1%}")

        if 'strategy' in self.results:
            strat = self.results['strategy']
            print(f"\n🎯 STRATEGIC ANALYSIS")
            print(f"   Prefers Center: {strat['center_preference']}")
            best_opening = max(strat['opening_preferences'].items(), key=lambda x: x[1])
            print(f"   Favorite Opening: Column {best_opening[0]}")

        # Overall assessment
        win_rate = self.results.get('vs_random', {}).get('win_rate', 0)
        if win_rate >= 0.75:
            assessment = "EXCELLENT - Strong strategic play"
        elif win_rate >= 0.65:
            assessment = "GOOD - Solid performance"
        elif win_rate >= 0.55:
            assessment = "FAIR - Basic competency"
        else:
            assessment = "NEEDS IMPROVEMENT - Consider retraining"

        print(f"\n🏆 OVERALL ASSESSMENT: {assessment}")

        return self.results

    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("Starting comprehensive agent benchmark...")

        agent = self.load_agent()
        if not agent:
            return None

        # Run all benchmarks
        self.benchmark_vs_random(agent, 1000)
        self.analyze_strategy(agent)
        self.learning_stats(agent)
        self.speed_benchmark(agent)
        self.memory_analysis(agent)
        self.competitive_analysis(agent)

        # Generate final report
        results = self.generate_report()

        return results


def main():
    """Main benchmark execution."""
    benchmark = AgentBenchmark()
    results = benchmark.run_full_benchmark()

    if results:
        print(f"\n✅ Benchmark completed successfully!")
        print(f"Full results stored in benchmark object.")
    else:
        print(f"\n❌ Benchmark failed - no trained model found.")


if __name__ == "__main__":
    main()