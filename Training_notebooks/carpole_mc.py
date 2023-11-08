import gymnasium as gym 
import numpy as np
import math
import numpy as np
import statistics 

import pickle

path = "/zhome/11/1/193832/resquivel/RL/Training_notebooks/data/carpole/montecarlo/"

def save_object(obj,value,param):
    try:
        with open(f"{path}data_{param}_{value}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
 

class MonteCarlo:
    def __init__(self, environment_name, episodes, epsilon = 0.2, alpha=0.2, gamma=0.95, adj_param = ''):
        self.env = gym.make(environment_name)
        self.episodes = episodes
        self.episode_data = 500
        self.adj_param = adj_param
        self.ep_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': [], 'std': [], 'mid': []}
        
        # Initialize Q
        self.space_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]
        self.number_bins = 50
        self.Q = np.random.randn(self.number_bins, self.number_bins, self.number_bins, self.number_bins, self.action_size)
        self.C = np.zeros_like(self.Q)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.ep_rewards_t = []

    def select_e_greedy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state])

    def discretize_state(self, observation):
        pos_index = np.argmin(np.abs(np.linspace(self.lower_bounds[0], self.upper_bounds[0], num=self.number_bins).tolist() - observation[0]))
        vel_index = np.argmin(np.abs(np.linspace(self.lower_bounds[1], self.upper_bounds[1], num=self.number_bins).tolist() - observation[1]))
        ang_index = np.argmin(np.abs(np.linspace(self.lower_bounds[2], self.upper_bounds[2], num=self.number_bins).tolist() - observation[2]))
        ang_vel_index = np.argmin(np.abs(np.linspace(self.lower_bounds[3], self.upper_bounds[3], num=self.number_bins).tolist() - observation[3]))
        return pos_index, vel_index, ang_index, ang_vel_index

    def test_q(self):
        state, info = self.env.reset()
        score = 0 
        epsilon = 0.0
        d_state = self.discretize_state(state)
        action = self.select_e_greedy(d_state, epsilon)
        terminated = False
        truncated = False
        while not (terminated or truncated or score > 500):
            state_prime, reward, terminated, truncated, info = self.env.step(action)
            d_state_prime = self.discretize_state(state_prime)
            action_prime = self.select_e_greedy(d_state_prime, epsilon)
            d_state = d_state_prime
            action = action_prime
            score += reward
        self.env.close()
        return score

    def train(self):
        
        ep_rewards_t = []

        for episode in range(1, self.episodes + 1):
            state, info = self.env.reset()
            score = 0 
            d_state = self.discretize_state(state)
            #action = self.select_e_greedy(d_state, self.epsilon)
            terminated = False
            truncated = False
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            
            while not (terminated or truncated or score > 500):
                action = self.select_e_greedy(d_state, self.epsilon)
                episode_states.append(d_state)
                episode_actions.append(action)
                
                state_prime, reward, terminated, truncated, _ = self.env.step(action)
                d_state_prime = self.discretize_state(state_prime)
                episode_rewards.append(reward)
                
                if terminated or truncated:
                    break
                
                d_state = d_state_prime

            
            G = 0
            W = 1
            for t in range(len(episode_states)):
                t = len(episode_states)-t-1
                
                state_t = episode_states[t]
                action_t = episode_actions[t]
                G = episode_rewards[t] + self.gamma * sum(episode_rewards[t+1:])
                
                self.C[state_t + (action_t,)] += W
                self.Q[state_t + (action_t,)] += W/self.C[state_t + (action_t,)] * (G - self.Q[state_t + (action_t,)])
                W = W
            

            t_score = self.test_q()
            ep_rewards_t.append(t_score)

            if not episode % self.episode_data:
                avg_reward = sum(ep_rewards_t[-self.episode_data:]) / len(ep_rewards_t[-self.episode_data:])
                self.ep_rewards_table['ep'].append(episode)
                self.ep_rewards_table['avg'].append(avg_reward)
                self.ep_rewards_table['min'].append(min(ep_rewards_t[-self.episode_data:]))
                self.ep_rewards_table['max'].append(max(ep_rewards_t[-self.episode_data:]))
                self.ep_rewards_table['std'].append(np.std(ep_rewards_t[-self.episode_data:]))
                self.ep_rewards_table['mid'].append(statistics.median(ep_rewards_t[-self.episode_data:]))
                
                print(f"Episode:{episode} avg:{avg_reward} min:{min(ep_rewards_t[-self.episode_data:])} max:{max(ep_rewards_t[-self.episode_data:])} std:{np.std(ep_rewards_t[-self.episode_data:])}")

        self.env.close()
        



episodes = 50001
episode_data = 500
environment_name = "CartPole-v1"



    
# epsilons = [0.3, 0.2, 0.1, 0.05, 0.01]
# for epsilon in epsilons:
#     print(f"epsilon = {epsilon}")
#     agent = MonteCarlo(environment_name, episodes, epsilon = epsilon, alpha= 0.4, gamma= 0.95, adj_param='epsilon')
#     agent.train()
#     save_object(agent,epsilon,'epsilon')
    
    
# gammas = [1, 0.95, 0.9, 0.85, 0.8]
# for gamma in gammas:
#     print(f"gamma = {gamma}")
#     agent = MonteCarlo(environment_name, episodes, epsilon = 0.2, alpha= 0.4, gamma= gamma, adj_param='gamma')
#     agent.train()
#     save_object(agent,gamma,'gamma')
    

# for agent_n in range(10):
#     print(f"agent = {agent_n}")
#     agent = MonteCarlo(environment_name, episodes, epsilon = 0.2, alpha= 0.4, gamma= 0.9, adj_param='test')
#     agent.train()
#     save_object(agent,agent_n,'agent')
    
for agent_n in range(10):
    print(f"agent = {agent_n}")
    agent = MonteCarlo(environment_name, episodes, epsilon = 0.3, alpha= 0.5, gamma= 1, adj_param='test_gen')
    agent.train()
    save_object(agent,agent_n,'agent_gen')