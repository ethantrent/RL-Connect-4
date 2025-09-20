#!/usr/bin/env python3
"""Interactive performance dashboard for Connect 4 RL agent monitoring."""

import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

from environment import Connect4Env
from agents import QLearningAgent
from training.train_agent import Trainer


class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""

    def __init__(self, model_path="models/q_learning_agent.pkl"):
        self.model_path = model_path
        self.env = Connect4Env()
        self.trainer = Trainer(self.env)
        self.agent = None
        self.metrics_history = []
        self.start_time = time.time()

    def load_agent(self):
        """Load the current agent."""
        if os.path.exists(self.model_path):
            self.agent = QLearningAgent()
            self.agent.load_model(self.model_path)
            return True
        return False

    def collect_metrics(self):
        """Collect current performance metrics."""
        if not self.agent:
            return None

        timestamp = time.time()

        # Quick performance test (20 games for speed)
        win_rate = self.trainer.evaluate_agent(self.agent, games=20)

        # Agent characteristics
        q_table_size = len(self.agent.q_table)
        epsilon = self.agent.epsilon
        training_episodes = self.agent.training_stats.get('episodes', 0)

        # Strategy analysis
        empty_state, _ = self.env.reset()
        valid_actions = self.env.get_valid_actions()
        policy_info = self.agent.get_policy_info(empty_state, valid_actions)

        best_opening = max(policy_info['q_values'].items(), key=lambda x: x[1])

        metrics = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).strftime('%H:%M:%S'),
            'win_rate': win_rate,
            'q_table_size': q_table_size,
            'epsilon': epsilon,
            'training_episodes': training_episodes,
            'best_opening_col': best_opening[0],
            'best_opening_q': best_opening[1]
        }

        self.metrics_history.append(metrics)

        # Keep only last 50 measurements
        if len(self.metrics_history) > 50:
            self.metrics_history.pop(0)

        return metrics

    def print_current_status(self):
        """Print current agent status to console."""
        if not self.metrics_history:
            return

        latest = self.metrics_history[-1]
        uptime = time.time() - self.start_time

        print(f"\n{'='*60}")
        print(f"       CONNECT 4 AGENT PERFORMANCE DASHBOARD")
        print(f"{'='*60}")
        print(f"Time: {latest['datetime']} | Uptime: {uptime/60:.1f}m")
        print(f"Win Rate: {latest['win_rate']:.1%} | Q-Table: {latest['q_table_size']:,}")
        print(f"Epsilon: {latest['epsilon']:.4f} | Episodes: {latest['training_episodes']:,}")
        print(f"Best Opening: Column {latest['best_opening_col']} (Q={latest['best_opening_q']:.3f})")

        # Performance trend (if we have multiple measurements)
        if len(self.metrics_history) >= 2:
            prev_win_rate = self.metrics_history[-2]['win_rate']
            trend = latest['win_rate'] - prev_win_rate

            if trend > 0.02:
                trend_str = f"📈 +{trend:.3f}"
            elif trend < -0.02:
                trend_str = f"📉 {trend:.3f}"
            else:
                trend_str = "➡️ stable"

            print(f"Trend: {trend_str}")

    def generate_live_plots(self):
        """Generate live performance plots."""
        if len(self.metrics_history) < 2:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        timestamps = [m['datetime'] for m in self.metrics_history]
        win_rates = [m['win_rate'] for m in self.metrics_history]
        q_table_sizes = [m['q_table_size'] for m in self.metrics_history]
        epsilons = [m['epsilon'] for m in self.metrics_history]

        # Win rate over time
        ax1.plot(timestamps, win_rates, 'b-o', markersize=3)
        ax1.set_title('Win Rate Over Time')
        ax1.set_ylabel('Win Rate')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Q-table growth
        ax2.plot(timestamps, q_table_sizes, 'g-o', markersize=3)
        ax2.set_title('Q-Table Size Growth')
        ax2.set_ylabel('State-Action Pairs')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Epsilon decay
        ax3.plot(timestamps, epsilons, 'r-o', markersize=3)
        ax3.set_title('Exploration Rate (Epsilon)')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # Opening preferences (latest only)
        if self.agent:
            empty_state, _ = self.env.reset()
            valid_actions = self.env.get_valid_actions()
            policy_info = self.agent.get_policy_info(empty_state, valid_actions)

            cols = list(range(7))
            q_values = [policy_info['q_values'][col] for col in cols]

            bars = ax4.bar(cols, q_values, color='purple', alpha=0.7)
            ax4.set_title('Current Opening Preferences')
            ax4.set_xlabel('Column')
            ax4.set_ylabel('Q-Value')
            ax4.set_xticks(cols)

            # Highlight best move
            best_idx = q_values.index(max(q_values))
            bars[best_idx].set_color('gold')

        plt.tight_layout()
        plt.savefig('performance_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()

    def continuous_monitoring(self, interval_seconds=30):
        """Run continuous monitoring loop."""
        print("🚀 Starting Performance Dashboard...")
        print(f"Monitoring interval: {interval_seconds} seconds")
        print("Press Ctrl+C to stop")

        try:
            while True:
                if self.load_agent():
                    metrics = self.collect_metrics()
                    if metrics:
                        self.print_current_status()
                        self.generate_live_plots()
                        self.save_metrics_log()
                else:
                    print(f"⚠️ Agent not found at {self.model_path}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
            self.generate_final_report()

    def save_metrics_log(self):
        """Save metrics to log file."""
        with open('performance_log.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def generate_final_report(self):
        """Generate final performance report."""
        if not self.metrics_history:
            return

        print(f"\n📊 FINAL PERFORMANCE REPORT")
        print(f"{'='*50}")

        first = self.metrics_history[0]
        last = self.metrics_history[-1]

        print(f"Monitoring Duration: {len(self.metrics_history)} measurements")
        print(f"Initial Win Rate: {first['win_rate']:.1%}")
        print(f"Final Win Rate: {last['win_rate']:.1%}")
        print(f"Win Rate Change: {last['win_rate'] - first['win_rate']:+.3f}")

        print(f"Q-Table Growth: {last['q_table_size'] - first['q_table_size']:,} new pairs")
        print(f"Epsilon Decay: {first['epsilon']:.4f} → {last['epsilon']:.4f}")

        # Calculate averages
        avg_win_rate = sum(m['win_rate'] for m in self.metrics_history) / len(self.metrics_history)
        print(f"Average Win Rate: {avg_win_rate:.1%}")

        # Best performance
        best_measurement = max(self.metrics_history, key=lambda x: x['win_rate'])
        print(f"Peak Performance: {best_measurement['win_rate']:.1%} at {best_measurement['datetime']}")

    def quick_health_check(self):
        """Perform quick health check and return status."""
        if not self.load_agent():
            return "❌ CRITICAL: No agent found"

        metrics = self.collect_metrics()
        if not metrics:
            return "❌ CRITICAL: Cannot evaluate agent"

        win_rate = metrics['win_rate']
        q_table_size = metrics['q_table_size']

        if win_rate >= 0.7:
            status = "✅ EXCELLENT"
        elif win_rate >= 0.6:
            status = "✅ GOOD"
        elif win_rate >= 0.5:
            status = "⚠️ FAIR"
        else:
            status = "❌ POOR"

        return f"{status}: {win_rate:.1%} win rate, {q_table_size:,} Q-pairs"

    def benchmark_mode(self):
        """Run comprehensive benchmark and monitoring."""
        print("🔬 BENCHMARK MODE")

        if not self.load_agent():
            print("❌ No agent found for benchmarking")
            return

        # Comprehensive evaluation
        print("Running comprehensive evaluation...")
        detailed_win_rate = self.trainer.evaluate_agent(self.agent, games=500)

        # Speed test
        print("Testing decision speed...")
        start_time = time.time()
        for _ in range(1000):
            state, _ = self.env.reset()
            valid_actions = self.env.get_valid_actions()
            self.agent.choose_action(state, valid_actions, training=False)
        speed_test_time = time.time() - start_time
        decisions_per_second = 1000 / speed_test_time

        # Memory analysis
        q_table_size = len(self.agent.q_table)
        estimated_memory_mb = q_table_size * 100 / (1024 * 1024)  # Rough estimate

        print(f"\n📋 BENCHMARK RESULTS")
        print(f"{'='*40}")
        print(f"Win Rate (500 games): {detailed_win_rate:.1%}")
        print(f"Decision Speed: {decisions_per_second:.0f} decisions/second")
        print(f"Q-Table Size: {q_table_size:,} pairs")
        print(f"Estimated Memory: {estimated_memory_mb:.1f} MB")
        print(f"Training Episodes: {self.agent.training_stats.get('episodes', 0):,}")

        # Performance assessment
        if detailed_win_rate >= 0.75:
            assessment = "🏆 ELITE PERFORMANCE"
        elif detailed_win_rate >= 0.65:
            assessment = "🥈 STRONG PERFORMANCE"
        elif detailed_win_rate >= 0.55:
            assessment = "🥉 ADEQUATE PERFORMANCE"
        else:
            assessment = "📚 NEEDS IMPROVEMENT"

        print(f"Assessment: {assessment}")


def main():
    """Main dashboard execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Connect 4 Agent Performance Dashboard')
    parser.add_argument('--mode', choices=['monitor', 'check', 'benchmark'],
                       default='check', help='Dashboard mode')
    parser.add_argument('--interval', type=int, default=30,
                       help='Monitoring interval in seconds')
    parser.add_argument('--model', default='models/q_learning_agent.pkl',
                       help='Path to agent model')

    args = parser.parse_args()

    dashboard = PerformanceDashboard(args.model)

    if args.mode == 'monitor':
        dashboard.continuous_monitoring(args.interval)
    elif args.mode == 'check':
        status = dashboard.quick_health_check()
        print(f"\n🏥 HEALTH CHECK: {status}")
    elif args.mode == 'benchmark':
        dashboard.benchmark_mode()


if __name__ == "__main__":
    main()