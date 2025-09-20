"""Test suite for Random agent."""

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import RandomAgent
from environment import Connect4Env


class TestRandomAgent(unittest.TestCase):
    """Test cases for Random agent."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent = RandomAgent("TestRandom")

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "TestRandom")

        # Test default name
        default_agent = RandomAgent()
        self.assertEqual(default_agent.name, "Random")

    def test_action_choice_valid(self):
        """Test that agent chooses valid actions."""
        board = np.zeros((6, 7))
        valid_actions = [0, 2, 4, 6]

        # Test multiple times due to randomness
        for _ in range(50):
            action = self.agent.choose_action(board, valid_actions)
            self.assertIn(action, valid_actions)

    def test_action_choice_all_valid(self):
        """Test action choice when all columns are valid."""
        board = np.zeros((6, 7))
        valid_actions = list(range(7))

        actions_chosen = set()
        for _ in range(100):  # Run many times to get good coverage
            action = self.agent.choose_action(board, valid_actions)
            self.assertIn(action, valid_actions)
            actions_chosen.add(action)

        # Should have chosen multiple different actions (randomness test)
        self.assertGreater(len(actions_chosen), 1)

    def test_action_choice_single_valid(self):
        """Test action choice when only one action is valid."""
        board = np.zeros((6, 7))
        valid_actions = [3]

        # Should always choose the only valid action
        for _ in range(10):
            action = self.agent.choose_action(board, valid_actions)
            self.assertEqual(action, 3)

    def test_action_choice_empty_valid_actions(self):
        """Test behavior with empty valid actions list."""
        board = np.zeros((6, 7))
        valid_actions = []

        # Should return some action as fallback
        action = self.agent.choose_action(board, valid_actions)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 7)

    def test_training_parameter_ignored(self):
        """Test that training parameter doesn't affect behavior."""
        board = np.zeros((6, 7))
        valid_actions = [0, 1, 2]

        # Collect actions for both training modes
        training_actions = set()
        no_training_actions = set()

        for _ in range(50):
            action_training = self.agent.choose_action(board, valid_actions, training=True)
            action_no_training = self.agent.choose_action(board, valid_actions, training=False)

            training_actions.add(action_training)
            no_training_actions.add(action_no_training)

        # Both should show randomness (multiple actions chosen)
        self.assertGreater(len(training_actions), 1)
        self.assertGreater(len(no_training_actions), 1)

    def test_no_op_methods(self):
        """Test that learning-related methods are no-ops."""
        board = np.zeros((6, 7))

        # These should not raise exceptions
        self.agent.update_q_value(board, 0, 1.0, board, [1, 2, 3], False)
        self.agent.decay_epsilon()
        self.agent.update_stats(1.0, 'win')

        # Agent should remain unchanged (no attributes to verify for random agent)
        self.assertEqual(self.agent.name, "TestRandom")

    def test_different_board_states(self):
        """Test that agent works with different board states."""
        # Empty board
        empty_board = np.zeros((6, 7))
        valid_actions = list(range(7))
        action = self.agent.choose_action(empty_board, valid_actions)
        self.assertIn(action, valid_actions)

        # Partially filled board
        partial_board = np.zeros((6, 7))
        partial_board[5, 0] = 1
        partial_board[5, 1] = 2
        valid_actions = [2, 3, 4, 5, 6]
        action = self.agent.choose_action(partial_board, valid_actions)
        self.assertIn(action, valid_actions)

        # Nearly full board
        nearly_full = np.ones((6, 7))
        nearly_full[0, 6] = 0  # Only top-right is free
        valid_actions = [6]
        action = self.agent.choose_action(nearly_full, valid_actions)
        self.assertEqual(action, 6)


class TestRandomAgentIntegration(unittest.TestCase):
    """Integration tests for Random agent with Connect4 environment."""

    def setUp(self):
        """Set up test fixtures."""
        self.env = Connect4Env()
        self.agent = RandomAgent()

    def test_agent_environment_interaction(self):
        """Test that random agent can play a complete game."""
        state, _ = self.env.reset()
        done = False
        moves = 0

        while not done and moves < 42:  # Prevent infinite loops
            valid_actions = self.env.get_valid_actions()
            action = self.agent.choose_action(state, valid_actions)

            # Action should be valid
            self.assertIn(action, valid_actions)

            # Take step
            state, reward, done, _, _ = self.env.step(action)
            moves += 1

        # Game should have ended properly
        self.assertLessEqual(moves, 42)

    def test_random_vs_random_games(self):
        """Test multiple games between random agents."""
        agent1 = RandomAgent("Random1")
        agent2 = RandomAgent("Random2")

        results = {'wins_1': 0, 'wins_2': 0, 'draws': 0}

        for game in range(10):  # Play 10 games
            state, _ = self.env.reset()
            done = False
            current_player = 1

            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                if current_player == 1:
                    action = agent1.choose_action(state, valid_actions)
                else:
                    action = agent2.choose_action(state, valid_actions)

                state, reward, done, _, _ = self.env.step(action)
                current_player = 3 - current_player

            # Record result
            if self.env.winner == 1:
                results['wins_1'] += 1
            elif self.env.winner == 2:
                results['wins_2'] += 1
            else:
                results['draws'] += 1

        # Should have played all games
        total_games = results['wins_1'] + results['wins_2'] + results['draws']
        self.assertEqual(total_games, 10)

    def test_randomness_distribution(self):
        """Test that random agent shows reasonable randomness."""
        board = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]

        action_counts = {action: 0 for action in valid_actions}

        # Collect many samples
        num_samples = 700  # 100 per action on average
        for _ in range(num_samples):
            action = self.agent.choose_action(board, valid_actions)
            action_counts[action] += 1

        # Each action should have been chosen at least once with high probability
        for action in valid_actions:
            self.assertGreater(action_counts[action], 0,
                              f"Action {action} was never chosen in {num_samples} samples")

        # No action should dominate too heavily (rough check for uniformity)
        max_count = max(action_counts.values())
        min_count = min(action_counts.values())

        # With uniform random, we expect roughly equal counts
        # Allow for some variance but not too much
        expected_per_action = num_samples / len(valid_actions)

        # Each count should be within reasonable bounds of expected
        for action, count in action_counts.items():
            self.assertGreater(count, expected_per_action * 0.5,
                              f"Action {action} chosen too rarely: {count}")
            self.assertLess(count, expected_per_action * 1.5,
                           f"Action {action} chosen too frequently: {count}")


if __name__ == '__main__':
    unittest.main()