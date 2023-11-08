import gymnasium as gym 
import os
import numpy as np
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics 

import pickle

path = "/zhome/11/1/193832/resquivel/RL/Training_notebooks/data/acrorobot/sarsa/"

def save_object(obj,value,param):
    try:
        with open(f"{path}data_{param}_{value}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
 

class sarsa:
    def __init__(self, environment_name, episodes, epsilon = 0.2, alpha=0.2, gamma=0.95, adj_param = ''):
        self.env = gym.make(environment_name)
        self.episodes = episodes
        self.episode_data = 500
        self.adj_param = adj_param
        self.ep_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': [], 'std': [], 'mid': []}
        
        # Initialize Q
        self.space_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1], self.env.observation_space.high[2], self.env.observation_space.high[3], self.env.observation_space.high[4], self.env.observation_space.high[5]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1], self.env.observation_space.low[2], self.env.observation_space.low[3], self.env.observation_space.low[4], self.env.observation_space.low[5]]
        self.number_bins = 20
        self.Q = np.random.randn(self.number_bins,self.number_bins,self.number_bins,self.number_bins,self.number_bins,self.number_bins,self.env.action_space.n)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        #self.ep_rewards_t = []

    def select_e_greedy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state])

    def discretize_state(self, observation):
        index_1 = np.argmin(np.abs(np.linspace(self.lower_bounds[0], self.upper_bounds[0], num=self.number_bins).tolist()-observation[0]))
        index_2 = np.argmin(np.abs(np.linspace(self.lower_bounds[1], self.upper_bounds[1], num=self.number_bins).tolist()-observation[1]))
        index_3 = np.argmin(np.abs(np.linspace(self.lower_bounds[2], self.upper_bounds[2], num=self.number_bins).tolist()-observation[2]))
        index_4 = np.argmin(np.abs(np.linspace(self.lower_bounds[3], self.upper_bounds[3], num=self.number_bins).tolist()-observation[3]))
        index_5 = np.argmin(np.abs(np.linspace(self.lower_bounds[4], self.upper_bounds[4], num=self.number_bins).tolist()-observation[4]))
        index_6 = np.argmin(np.abs(np.linspace(self.lower_bounds[5], self.upper_bounds[5], num=self.number_bins).tolist()-observation[5]))
        return index_1, index_2, index_3, index_4, index_5, index_6

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
            action = self.select_e_greedy(d_state, self.epsilon)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                state_prime, reward, terminated, truncated, info = self.env.step(action)
                d_state_prime = self.discretize_state(state_prime)
                action_prime = self.select_e_greedy(d_state_prime, self.epsilon)
                self.Q[d_state + (action,)] += self.alpha * (reward + self.gamma * self.Q[d_state_prime + (action_prime,)] - self.Q[d_state + (action,)])
                d_state = d_state_prime
                action = action_prime
                score += reward

            t_score = self.test_q()
            #ep_rewards.append(score)
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
environment_name = "Acrobot-v1"



# #alphas = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
# alphas = [0.1, 0.2]
# for alpha in alphas:
#     print(f"Alpha = {alpha}")
#     agent = sarsa(environment_name, episodes, epsilon = 0.2, alpha= alpha, gamma= 0.95, adj_param='alpha')
#     agent.train()
#     save_object(agent,alpha,'Alpha')
    
    
# epsilons = [0.4, 0.3, 0.2, 0.1]
# for epsilon in epsilons:
#     print(f"epsilon = {epsilon}")
#     agent = sarsa(environment_name, episodes, epsilon = epsilon, alpha= 0.2, gamma= 0.95, adj_param='epsilon')
#     agent.train()
#     save_object(agent,epsilon,'epsilon')
    
    
gammas = [0.9, 0.85, 0.8]
for gamma in gammas:
    print(f"gamma = {gamma}")
    agent = sarsa(environment_name, episodes, epsilon = 0.3, alpha= 0.2, gamma= gamma, adj_param='gamma')
    agent.train()
    save_object(agent,gamma,'gamma')