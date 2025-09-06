import os
import sys
import numpy as np
from typing import Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment import Connect4Env
from agents import QLearningAgent, RandomAgent


class GameUI:
    """Simple CLI interface for playing Connect 4."""
    
    def __init__(self, env: Connect4Env):
        self.env = env
        self.agents = {}
    
    def load_agent(self, agent_type: str, model_path: Optional[str] = None) -> object:
        """Load an agent for gameplay."""
        if agent_type.lower() == 'qlearning':
            agent = QLearningAgent()
            if model_path and os.path.exists(model_path):
                agent.load_model(model_path)
                print(f"Loaded trained Q-learning agent from {model_path}")
            else:
                print("No trained model found. Using untrained Q-learning agent.")
            return agent
        elif agent_type.lower() == 'random':
            return RandomAgent("Random")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_human_move(self) -> int:
        """Get move input from human player."""
        while True:
            try:
                move = input(f"Enter column (0-{self.env.cols-1}) or 'q' to quit: ").strip()
                if move.lower() == 'q':
                    return -1
                
                col = int(move)
                if 0 <= col < self.env.cols and self.env.is_valid_action(col):
                    return col
                else:
                    print(f"Invalid column. Choose from {self.env.get_valid_actions()}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
    
    def play_human_vs_agent(self, agent_type: str = 'qlearning', model_path: str = 'models/q_learning_agent.pkl'):
        """Play human vs agent game."""
        print("\n=== Connect 4: Human vs AI ===")
        print("You are 'X', AI is 'O'")
        print("Drop pieces by entering column numbers (0-6)")
        print("Get 4 in a row to win!\n")
        
        # Load agent
        agent = self.load_agent(agent_type, model_path)
        
        # Game setup
        human_player = 1  # Human is always player 1 (X)
        ai_player = 2     # AI is always player 2 (O)
        
        state, _ = self.env.reset()
        done = False
        current_player = 1
        
        self.env.render('ascii')
        
        while not done:
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions:
                print("Board is full! Game ends in a draw.")
                break
            
            if current_player == human_player:
                print("Your turn!")
                action = self.get_human_move()
                
                if action == -1:  # Quit
                    print("Game quit by human player.")
                    break
                    
            else:
                print("AI is thinking...")
                action = agent.choose_action(state, valid_actions, training=False)
                print(f"AI chooses column {action}")
            
            # Make move
            state, reward, done, _, info = self.env.step(action)
            self.env.render('ascii')
            
            if done:
                if self.env.winner == human_player:
                    print("🎉 Congratulations! You won!")
                elif self.env.winner == ai_player:
                    print("🤖 AI wins! Better luck next time.")
                else:
                    print("🤝 It's a draw!")
                break
            
            current_player = 3 - current_player
        
        print("\nThanks for playing!")
    
    def play_agent_vs_agent(self, agent1_type: str = 'qlearning', agent2_type: str = 'random', 
                           model_path: str = 'models/q_learning_agent.pkl', games: int = 10):
        """Watch two agents play against each other."""
        print(f"\n=== {agent1_type.upper()} vs {agent2_type.upper()} ===")
        print(f"Playing {games} games...\n")
        
        # Load agents
        agent1 = self.load_agent(agent1_type, model_path)
        agent2 = self.load_agent(agent2_type)
        
        results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
        
        for game in range(1, games + 1):
            print(f"Game {game}/{games}")
            
            state, _ = self.env.reset()
            done = False
            current_player = 1
            
            # Randomly assign players
            agent1_player = np.random.choice([1, 2])
            agent2_player = 3 - agent1_player
            
            move_count = 0
            while not done and move_count < 42:  # Max possible moves
                valid_actions = self.env.get_valid_actions()
                
                if current_player == agent1_player:
                    action = agent1.choose_action(state, valid_actions, training=False)
                    agent_name = agent1_type
                else:
                    action = agent2.choose_action(state, valid_actions, training=False)
                    agent_name = agent2_type
                
                print(f"  {agent_name} chooses column {action}")
                
                state, reward, done, _, _ = self.env.step(action)
                
                if done:
                    if self.env.winner == agent1_player:
                        results['agent1_wins'] += 1
                        print(f"  {agent1_type} wins!")
                    elif self.env.winner == agent2_player:
                        results['agent2_wins'] += 1
                        print(f"  {agent2_type} wins!")
                    else:
                        results['draws'] += 1
                        print("  Draw!")
                
                current_player = 3 - current_player
                move_count += 1
            
            if move_count >= 42 and not done:
                results['draws'] += 1
                print("  Draw (board full)!")
            
            print()
        
        # Results summary
        print("=== RESULTS ===")
        print(f"{agent1_type}: {results['agent1_wins']} wins ({results['agent1_wins']/games*100:.1f}%)")
        print(f"{agent2_type}: {results['agent2_wins']} wins ({results['agent2_wins']/games*100:.1f}%)")
        print(f"Draws: {results['draws']} ({results['draws']/games*100:.1f}%)")
    
    def demonstrate_learning(self, model_path: str = 'models/q_learning_agent.pkl'):
        """Demonstrate the agent's learning by showing Q-values for different states."""
        print("\n=== Agent Learning Demonstration ===")
        
        agent = self.load_agent('qlearning', model_path)
        
        if len(agent.q_table) == 0:
            print("No trained model found. Train the agent first!")
            return
        
        print(f"Agent has learned Q-values for {len(agent.q_table)} state-action pairs")
        print(f"Current epsilon (exploration rate): {agent.epsilon:.4f}")
        print(f"Training episodes completed: {agent.training_stats.get('episodes', 0)}")
        print(f"Win rate: {agent.get_win_rate():.3f}")
        
        # Show policy for current state
        state, _ = self.env.reset()
        self.env.render('ascii')
        
        valid_actions = self.env.get_valid_actions()
        policy_info = agent.get_policy_info(state, valid_actions)
        
        print("\nAgent's Q-values for current state:")
        for action in valid_actions:
            q_val = policy_info['q_values'][action]
            best_mark = " <-- BEST" if action == policy_info['best_action'] else ""
            print(f"  Column {action}: Q = {q_val:.4f}{best_mark}")


def main():
    """Main function with menu system."""
    env = Connect4Env()
    ui = GameUI(env)
    
    while True:
        print("\n" + "="*50)
        print("       CONNECT 4 - REINFORCEMENT LEARNING")
        print("="*50)
        print("1. Play against AI (Q-learning agent)")
        print("2. Play against Random AI")
        print("3. Watch AI vs Random (10 games)")
        print("4. Watch AI vs Random (100 games)")
        print("5. Demonstrate agent's learning")
        print("6. Train new agent (this will take a while!)")
        print("7. Quick test game (AI vs Random, 1 game with display)")
        print("8. Quit")
        print("="*50)
        
        choice = input("Choose option (1-8): ").strip()
        
        if choice == '1':
            ui.play_human_vs_agent('qlearning')
        elif choice == '2':
            ui.play_human_vs_agent('random')
        elif choice == '3':
            ui.play_agent_vs_agent('qlearning', 'random', games=10)
        elif choice == '4':
            ui.play_agent_vs_agent('qlearning', 'random', games=100)
        elif choice == '5':
            ui.demonstrate_learning()
        elif choice == '6':
            print("Starting training... This will take several minutes!")
            from training.train_agent import main as train_main
            train_main()
        elif choice == '7':
            ui.play_agent_vs_agent('qlearning', 'random', games=1)
        elif choice == '8':
            print("Thanks for playing!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()