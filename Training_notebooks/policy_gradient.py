import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import statistics
import pickle

path = "/zhome/11/1/193832/resquivel/RL/Training_notebooks/data/carpole/policygradient/"

def save_object(obj,value,param):
    try:
        with open(f"{path}data_{param}_{value}.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
        

class PolicyNet(nn.Module):
    """Policy network"""

    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate, num_episodes, rollout_limit, discount_factor, val_freq):
        super(PolicyNet, self).__init__()
        # network
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_outputs)
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.ep_rewards_table = {'ep': [], 'avg': [],'std': []}
        self.num_episodes = num_episodes
        self.rollout_limit = rollout_limit
        self.discount_factor = discount_factor
        self.val_freq = val_freq

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)

    def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = np.zeros(len(rewards))
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards)-1)):
            returns[t] = rewards[t] + self.discount_factor * returns[t+1]
        return returns

    def train_policy(self, env):
        training_rewards, losses = [], []
        print('start training')
        for i in range(self.num_episodes):
            rollout = []
            s,_ = env.reset()
            for j in range(self.rollout_limit):
                # generate rollout by iteratively evaluating the current policy on the environment
                with torch.no_grad():
                    a_prob = self(torch.from_numpy(np.atleast_2d(s)).float())
                    a = torch.multinomial(a_prob, num_samples=1).squeeze().numpy()
                s1, r, done, trunc, _ = env.step(a)
                done = done or trunc
                rollout.append((s, a, r))
                s = s1
                if done: break
            # prepare batch
            rollout = np.array(rollout, dtype="object")
            states = np.vstack(rollout[:,0])
            actions = np.vstack(rollout[:,1])
            rewards = np.array(rollout[:,2], dtype=float)
            returns = self.compute_returns(rewards)
            # policy gradient update
            self.optimizer.zero_grad()
            a_probs = self(torch.from_numpy(states).float()).gather(1, torch.from_numpy(actions)).view(-1)
            loss = self.loss(a_probs, torch.from_numpy(returns).float())
            loss.backward()
            self.optimizer.step()
            # bookkeeping
            training_rewards.append(sum(rewards))
            losses.append(loss.item())
            # print
            if (i+1) % self.val_freq == 0:
                # validation
                validation_rewards = []
                for _ in range(100):
                    s,_ = env.reset()
                    reward = 0
                    for _ in range(self.rollout_limit):
                        with torch.no_grad():
                            a_prob = self(torch.from_numpy(np.atleast_2d(s)).float())
                            a = a_prob.argmax().item()
                        s, r, done, trunc, _ = env.step(a)
                        done = done or trunc
                        reward += r
                        if done: break
                    validation_rewards.append(reward)
                
                mean_value = statistics.mean(validation_rewards)
                std_deviation = statistics.stdev(validation_rewards)
                self.ep_rewards_table['ep'].append(i+1)
                self.ep_rewards_table['avg'].append(mean_value)
                self.ep_rewards_table['std'].append(std_deviation)
                
                print('{:4d}. mean training reward: {:6.2f}, mean validation reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, np.mean(training_rewards[-self.val_freq:]), np.mean(validation_rewards), np.mean(losses[-self.val_freq:])))
        print('done')

# setup environment
env = gym.make("CartPole-v1") # Create environment
n_inputs = env.observation_space.shape[0]
n_hidden = 64
n_outputs = env.action_space.n

# training settings
num_episodes = 5000
rollout_limit = 500 # max rollout length
discount_factor = 1.0 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.001 # you know this by now
val_freq = 100 # validation frequency

# setup policy network


# train policy network
for agent_n in range(10):
    print(f"agent = {agent_n}")
    policy = PolicyNet(n_inputs, n_hidden, n_outputs, learning_rate, num_episodes, rollout_limit, discount_factor, val_freq)
    policy.train_policy(env)
    save_object(policy,agent_n,'agent_gen2')