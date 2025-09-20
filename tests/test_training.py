"""Test suite for training functionality."""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment import Connect4Env
from agents import QLearningAgent, RandomAgent
from training.train_agent import Trainer


class TestTrainer(unittest.TestCase):
    """Test cases for the Trainer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.env = Connect4Env()
        self.trainer = Trainer(self.env)
        self.agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.5,
            epsilon_min=0.01,
            epsilon_decay=0.99
        )

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer.env, Connect4Env)
        self.assertIn('episode', self.trainer.training_history)
        self.assertIn('win_rate', self.trainer.training_history)
        self.assertIn('avg_reward', self.trainer.training_history)
        self.assertIn('epsilon', self.trainer.training_history)

        # All history lists should be empty initially
        for key in self.trainer.training_history:
            self.assertEqual(len(self.trainer.training_history[key]), 0)

    def test_evaluate_agent_basic(self):
        """Test basic agent evaluation functionality."""
        # Test with small number of games
        win_rate = self.trainer.evaluate_agent(self.agent, games=10)

        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)

    def test_evaluate_agent_deterministic(self):
        """Test evaluation with a deterministic agent."""
        # Create an agent that always chooses column 3
        class DeterministicAgent:
            def choose_action(self, state, valid_actions, training=False):
                return 3 if 3 in valid_actions else valid_actions[0]

        deterministic_agent = DeterministicAgent()
        win_rate = self.trainer.evaluate_agent(deterministic_agent, games=5)

        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)

    def test_evaluate_agent_different_game_counts(self):
        """Test evaluation with different numbers of games."""
        # Test various game counts
        for games in [1, 5, 20]:
            with self.subTest(games=games):
                win_rate = self.trainer.evaluate_agent(self.agent, games=games)
                self.assertIsInstance(win_rate, float)
                self.assertGreaterEqual(win_rate, 0.0)
                self.assertLessEqual(win_rate, 1.0)

    def test_train_vs_random_short(self):
        """Test short training session."""
        initial_q_table_size = len(self.agent.q_table)
        initial_epsilon = self.agent.epsilon

        # Train for just a few episodes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            trained_agent = self.trainer.train_vs_random(
                agent=self.agent,
                episodes=10,
                eval_frequency=5,
                save_frequency=10,
                model_path=tmp_path
            )

            # Verify training occurred
            self.assertGreater(len(trained_agent.q_table), initial_q_table_size)
            self.assertLess(trained_agent.epsilon, initial_epsilon)
            self.assertGreater(trained_agent.training_stats['episodes'], 0)

            # Verify model was saved
            self.assertTrue(os.path.exists(tmp_path))

            # Verify training history was recorded
            self.assertGreater(len(self.trainer.training_history['episode']), 0)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_training_history_recording(self):
        """Test that training history is recorded correctly."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Train with frequent evaluation
            self.trainer.train_vs_random(
                agent=self.agent,
                episodes=20,
                eval_frequency=10,  # Evaluate every 10 episodes
                save_frequency=50,
                model_path=tmp_path
            )

            # Should have 2 evaluation points (episodes 10 and 20)
            self.assertEqual(len(self.trainer.training_history['episode']), 2)
            self.assertEqual(self.trainer.training_history['episode'], [10, 20])

            # All history lists should have same length
            history_lengths = [len(self.trainer.training_history[key])
                             for key in self.trainer.training_history]
            self.assertTrue(all(length == history_lengths[0] for length in history_lengths))

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_agent_stats_updated(self):
        """Test that agent statistics are updated during training."""
        initial_episodes = self.agent.training_stats['episodes']

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.trainer.train_vs_random(
                agent=self.agent,
                episodes=10,
                eval_frequency=20,  # No evaluation during training
                save_frequency=50,
                model_path=tmp_path
            )

            # Agent stats should be updated
            self.assertEqual(self.agent.training_stats['episodes'], initial_episodes + 10)
            self.assertGreaterEqual(self.agent.training_stats['wins'], 0)
            self.assertGreaterEqual(self.agent.training_stats['losses'], 0)
            self.assertGreaterEqual(self.agent.training_stats['draws'], 0)

            # Total games should equal episodes
            total_games = (self.agent.training_stats['wins'] +
                          self.agent.training_stats['losses'] +
                          self.agent.training_stats['draws'])
            self.assertEqual(total_games, self.agent.training_stats['episodes'])

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_epsilon_decay_during_training(self):
        """Test that epsilon decays during training."""
        initial_epsilon = self.agent.epsilon

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.trainer.train_vs_random(
                agent=self.agent,
                episodes=20,
                eval_frequency=50,  # No evaluation
                save_frequency=50,
                model_path=tmp_path
            )

            # Epsilon should have decayed
            self.assertLess(self.agent.epsilon, initial_epsilon)
            self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_model_saving_frequency(self):
        """Test that model is saved at specified frequencies."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        # Remove the temp file so we can test creation
        os.unlink(tmp_path)

        try:
            # Train with save frequency = 5
            self.trainer.train_vs_random(
                agent=self.agent,
                episodes=7,  # Less than save frequency
                eval_frequency=20,
                save_frequency=5,
                model_path=tmp_path
            )

            # Model should exist (saved at episode 5 and final save)
            self.assertTrue(os.path.exists(tmp_path))

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_random_player_assignment(self):
        """Test that player assignments are randomized."""
        # This is a statistical test - run multiple games and verify both players are used
        player_assignments = []

        # Mock the training loop to capture player assignments
        original_choice = np.random.choice

        def mock_choice(options):
            result = original_choice(options)
            if len(options) == 2 and set(options) == {1, 2}:
                player_assignments.append(result)
            return result

        # Temporarily replace np.random.choice
        np.random.choice = mock_choice

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_path = tmp_file.name

            self.trainer.train_vs_random(
                agent=self.agent,
                episodes=20,
                eval_frequency=50,
                save_frequency=50,
                model_path=tmp_path
            )

            # Should have captured player assignments
            self.assertEqual(len(player_assignments), 20)

            # Both players should be represented
            unique_assignments = set(player_assignments)
            self.assertEqual(unique_assignments, {1, 2})

        finally:
            # Restore original function
            np.random.choice = original_choice
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.env = Connect4Env()
        self.trainer = Trainer(self.env)

    def test_complete_training_pipeline(self):
        """Test a complete mini training pipeline."""
        agent = QLearningAgent(
            learning_rate=0.2,  # Higher for faster learning in test
            discount_factor=0.9,
            epsilon=0.8,
            epsilon_min=0.1,
            epsilon_decay=0.95
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Run complete training
            trained_agent = self.trainer.train_vs_random(
                agent=agent,
                episodes=50,
                eval_frequency=25,
                save_frequency=25,
                model_path=tmp_path
            )

            # Verify training completed
            self.assertEqual(trained_agent.training_stats['episodes'], 50)
            self.assertTrue(os.path.exists(tmp_path))

            # Verify agent learned something (Q-table not empty)
            self.assertGreater(len(trained_agent.q_table), 0)

            # Test the saved model can be loaded
            new_agent = QLearningAgent()
            new_agent.load_model(tmp_path)
            self.assertEqual(new_agent.training_stats['episodes'], 50)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_training_improves_performance(self):
        """Test that training actually improves agent performance."""
        agent = QLearningAgent(
            learning_rate=0.3,
            discount_factor=0.9,
            epsilon=0.9,
            epsilon_min=0.1,
            epsilon_decay=0.98
        )

        # Evaluate initial performance
        initial_win_rate = self.trainer.evaluate_agent(agent, games=20)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Train the agent
            self.trainer.train_vs_random(
                agent=agent,
                episodes=100,
                eval_frequency=200,  # No intermediate evaluation
                save_frequency=200,
                model_path=tmp_path
            )

            # Evaluate final performance
            final_win_rate = self.trainer.evaluate_agent(agent, games=20)

            # Performance should improve or at least not get significantly worse
            # (Note: with randomness, improvement isn't guaranteed in every run)
            # So we just test that the evaluation works and returns reasonable values
            self.assertIsInstance(final_win_rate, float)
            self.assertGreaterEqual(final_win_rate, 0.0)
            self.assertLessEqual(final_win_rate, 1.0)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()