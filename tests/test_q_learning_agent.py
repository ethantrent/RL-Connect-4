"""Test suite for Q-Learning agent."""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import QLearningAgent
from environment import Connect4Env


class TestQLearningAgent(unittest.TestCase):
    """Test cases for Q-Learning agent."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1,
            epsilon_min=0.01,
            epsilon_decay=0.99
        )
        self.env = Connect4Env()

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.95)
        self.assertEqual(self.agent.epsilon, 0.1)
        self.assertEqual(self.agent.epsilon_min, 0.01)
        self.assertEqual(self.agent.epsilon_decay, 0.99)
        self.assertEqual(len(self.agent.q_table), 0)

    def test_state_key_generation(self):
        """Test state key generation for Q-table."""
        board1 = np.zeros((6, 7))
        board2 = np.zeros((6, 7))
        board2[5, 0] = 1

        key1 = self.agent.get_state_key(board1)
        key2 = self.agent.get_state_key(board2)

        self.assertNotEqual(key1, key2)
        self.assertIsInstance(key1, str)
        self.assertIsInstance(key2, str)

        # Same boards should produce same keys
        key1_repeat = self.agent.get_state_key(board1)
        self.assertEqual(key1, key1_repeat)

    def test_q_value_operations(self):
        """Test Q-value getting and setting."""
        board = np.zeros((6, 7))
        action = 3

        # Initial Q-value should be 0
        initial_q = self.agent.get_q_value(board, action)
        self.assertEqual(initial_q, 0.0)

        # Set and retrieve Q-value
        self.agent.set_q_value(board, action, 0.5)
        retrieved_q = self.agent.get_q_value(board, action)
        self.assertEqual(retrieved_q, 0.5)

    def test_action_choice_deterministic(self):
        """Test action choice in exploitation mode (epsilon=0)."""
        # Set epsilon to 0 for deterministic behavior
        self.agent.epsilon = 0.0

        board = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]

        # Set different Q-values
        self.agent.set_q_value(board, 0, 0.1)
        self.agent.set_q_value(board, 1, 0.5)  # Highest
        self.agent.set_q_value(board, 2, 0.3)

        # Should choose action with highest Q-value
        chosen_action = self.agent.choose_action(board, valid_actions, training=True)
        self.assertEqual(chosen_action, 1)

    def test_action_choice_ties(self):
        """Test action choice when multiple actions have same Q-value."""
        self.agent.epsilon = 0.0

        board = np.zeros((6, 7))
        valid_actions = [0, 1, 2]

        # Set same Q-values for multiple actions
        for action in valid_actions:
            self.agent.set_q_value(board, action, 0.5)

        # Should choose one of the tied actions
        chosen_action = self.agent.choose_action(board, valid_actions, training=True)
        self.assertIn(chosen_action, valid_actions)

    def test_action_choice_exploration(self):
        """Test that exploration occurs with epsilon > 0."""
        self.agent.epsilon = 1.0  # Always explore

        board = np.zeros((6, 7))
        valid_actions = [0, 1, 2]

        # Set clear best action
        self.agent.set_q_value(board, 0, 1.0)  # Much higher
        self.agent.set_q_value(board, 1, 0.0)
        self.agent.set_q_value(board, 2, 0.0)

        # With epsilon=1.0, should sometimes choose non-optimal actions
        actions_chosen = set()
        for _ in range(50):  # Run multiple times
            action = self.agent.choose_action(board, valid_actions, training=True)
            actions_chosen.add(action)

        # Should have explored multiple actions despite clear best choice
        self.assertGreater(len(actions_chosen), 1)

    def test_action_choice_no_training(self):
        """Test that no exploration occurs when training=False."""
        self.agent.epsilon = 1.0  # Would normally always explore

        board = np.zeros((6, 7))
        valid_actions = [0, 1, 2]

        # Set clear best action
        self.agent.set_q_value(board, 0, 1.0)
        self.agent.set_q_value(board, 1, 0.0)
        self.agent.set_q_value(board, 2, 0.0)

        # With training=False, should always exploit despite high epsilon
        for _ in range(10):
            action = self.agent.choose_action(board, valid_actions, training=False)
            self.assertEqual(action, 0)

    def test_q_value_update_terminal(self):
        """Test Q-value update for terminal states."""
        board = np.zeros((6, 7))
        action = 3
        reward = 1.0
        next_board = np.zeros((6, 7))
        next_valid_actions = []

        # Initial Q-value
        initial_q = self.agent.get_q_value(board, action)

        # Update Q-value for terminal state
        self.agent.update_q_value(board, action, reward, next_board, next_valid_actions, done=True)

        # For terminal states, target = reward
        expected_q = initial_q + self.agent.learning_rate * (reward - initial_q)
        actual_q = self.agent.get_q_value(board, action)

        self.assertAlmostEqual(actual_q, expected_q, places=6)

    def test_q_value_update_non_terminal(self):
        """Test Q-value update for non-terminal states."""
        board = np.zeros((6, 7))
        action = 3
        reward = 0.0
        next_board = np.zeros((6, 7))
        next_board[5, 3] = 1  # Different from current board
        next_valid_actions = [0, 1, 2, 4, 5, 6]

        # Set some Q-values for next state
        for a in next_valid_actions:
            self.agent.set_q_value(next_board, a, 0.1)
        self.agent.set_q_value(next_board, 2, 0.5)  # Highest

        initial_q = self.agent.get_q_value(board, action)

        # Update Q-value
        self.agent.update_q_value(board, action, reward, next_board, next_valid_actions, done=False)

        # Calculate expected value
        max_next_q = 0.5  # Highest Q-value in next state
        target = reward + self.agent.discount_factor * max_next_q
        expected_q = initial_q + self.agent.learning_rate * (target - initial_q)

        actual_q = self.agent.get_q_value(board, action)
        self.assertAlmostEqual(actual_q, expected_q, places=6)

    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()

        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)

        # Test that epsilon doesn't go below minimum
        self.agent.epsilon = self.agent.epsilon_min
        self.agent.decay_epsilon()
        self.assertEqual(self.agent.epsilon, self.agent.epsilon_min)

    def test_stats_tracking(self):
        """Test training statistics tracking."""
        # Initial stats
        self.assertEqual(self.agent.training_stats['episodes'], 0)
        self.assertEqual(self.agent.training_stats['wins'], 0)
        self.assertEqual(self.agent.training_stats['losses'], 0)
        self.assertEqual(self.agent.training_stats['draws'], 0)

        # Update stats
        self.agent.update_stats(1.0, 'win')
        self.agent.update_stats(-1.0, 'loss')
        self.agent.update_stats(0.0, 'draw')

        self.assertEqual(self.agent.training_stats['episodes'], 3)
        self.assertEqual(self.agent.training_stats['wins'], 1)
        self.assertEqual(self.agent.training_stats['losses'], 1)
        self.assertEqual(self.agent.training_stats['draws'], 1)
        self.assertEqual(self.agent.training_stats['total_reward'], 0.0)

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        # No games played
        self.assertEqual(self.agent.get_win_rate(), 0.0)

        # Play some games
        self.agent.update_stats(1.0, 'win')
        self.agent.update_stats(-1.0, 'loss')
        self.agent.update_stats(1.0, 'win')

        expected_win_rate = 2.0 / 3.0
        self.assertAlmostEqual(self.agent.get_win_rate(), expected_win_rate, places=6)

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Set up agent with some data
        board = np.zeros((6, 7))
        self.agent.set_q_value(board, 0, 0.5)
        self.agent.set_q_value(board, 1, 0.7)
        self.agent.update_stats(1.0, 'win')
        self.agent.epsilon = 0.5

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.agent.save_model(tmp_path)

            # Create new agent and load
            new_agent = QLearningAgent()
            new_agent.load_model(tmp_path)

            # Verify loaded data matches
            self.assertEqual(new_agent.get_q_value(board, 0), 0.5)
            self.assertEqual(new_agent.get_q_value(board, 1), 0.7)
            self.assertEqual(new_agent.epsilon, 0.5)
            self.assertEqual(new_agent.training_stats['wins'], 1)
            self.assertEqual(new_agent.training_stats['episodes'], 1)

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_load_nonexistent_model(self):
        """Test loading from non-existent file."""
        # Should not raise exception, just print message
        self.agent.load_model('/nonexistent/path.pkl')
        # Agent should remain in initial state
        self.assertEqual(len(self.agent.q_table), 0)

    def test_policy_info(self):
        """Test policy information retrieval."""
        board = np.zeros((6, 7))
        valid_actions = [0, 1, 2]

        # Set some Q-values
        self.agent.set_q_value(board, 0, 0.3)
        self.agent.set_q_value(board, 1, 0.7)  # Best
        self.agent.set_q_value(board, 2, 0.1)

        policy_info = self.agent.get_policy_info(board, valid_actions)

        self.assertEqual(policy_info['best_action'], 1)
        self.assertEqual(policy_info['q_values'][0], 0.3)
        self.assertEqual(policy_info['q_values'][1], 0.7)
        self.assertEqual(policy_info['q_values'][2], 0.1)
        self.assertEqual(policy_info['epsilon'], self.agent.epsilon)

    def test_empty_valid_actions(self):
        """Test behavior with empty valid actions list."""
        board = np.zeros((6, 7))
        valid_actions = []

        # Should return some action (fallback behavior)
        action = self.agent.choose_action(board, valid_actions, training=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 7)


class TestQLearningAgentIntegration(unittest.TestCase):
    """Integration tests for Q-Learning agent with Connect4 environment."""

    def setUp(self):
        """Set up test fixtures."""
        self.env = Connect4Env()
        self.agent = QLearningAgent(epsilon=0.1)

    def test_agent_environment_interaction(self):
        """Test that agent can interact with environment properly."""
        state, _ = self.env.reset()
        done = False
        moves = 0

        while not done and moves < 42:  # Prevent infinite loops
            valid_actions = self.env.get_valid_actions()
            action = self.agent.choose_action(state, valid_actions, training=True)

            # Action should be valid
            self.assertIn(action, valid_actions)

            # Take step
            next_state, reward, done, _, _ = self.env.step(action)

            # Update agent (simplified)
            if not done:
                next_valid_actions = self.env.get_valid_actions()
                self.agent.update_q_value(state, action, reward, next_state, next_valid_actions, done)

            state = next_state
            moves += 1

        # Game should have ended properly
        self.assertLessEqual(moves, 42)

    def test_learning_occurs(self):
        """Test that agent actually learns from experience."""
        # Play a few games and verify Q-table grows
        initial_q_table_size = len(self.agent.q_table)

        for episode in range(10):
            state, _ = self.env.reset()
            done = False

            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                action = self.agent.choose_action(state, valid_actions, training=True)
                next_state, reward, done, _, _ = self.env.step(action)

                if not done:
                    next_valid_actions = self.env.get_valid_actions()
                    self.agent.update_q_value(state, action, reward, next_state, next_valid_actions, done)

                state = next_state

        # Q-table should have grown
        final_q_table_size = len(self.agent.q_table)
        self.assertGreater(final_q_table_size, initial_q_table_size)


if __name__ == '__main__':
    unittest.main()