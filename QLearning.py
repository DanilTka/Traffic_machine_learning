import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch


class DeepQNetwork(nn.Module):  # +
    def __init__(self, lr, input_dims, n1_dims, n2_dims, n_actions):
        super().__init__()  # starting base class initializer
        self.input_dims = input_dims
        self.n1_dims = n1_dims
        self.n2_dims = n2_dims
        self.n_actions = n_actions  # All possible actions
        self.n1 = nn.Linear(*self.input_dims, self.n1_dims)
        self.n2 = nn.Linear(self.n1_dims, self.n2_dims)
        self.n3 = nn.Linear(self.n2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):  # Feeding network
        x = F.relu(self.n1(state))
        x = F.relu(self.n2(x))
        # x = nn.functional.softmax(x)
        actions = self.n3(x)
        return actions


def save_model(agent):
    T.save({'state_dict': agent.Q_eval.state_dict(),
            'optimizer': agent.Q_eval.optimizer.state_dict(),
            'loss': agent.loss,
            'epsilon': agent.epsilon,
            'state_memory':agent.state_memory,
            'next_state_memory': agent.next_state_memory,
            'action_memory': agent.action_memory,
            'reward_memory': agent.reward_memory,
            'done_memory': agent.done_memory,
            'gamma': agent.gamma,
            'batch_size': agent.batch_size,
            'lr': agent.lr}, 'best_model.pth.tar')


def load_model(lr, n_actions, input_dims, gamma, epsilon):
    model = DeepQNetwork(lr=lr, n_actions=n_actions, input_dims=input_dims,
                         n1_dims=input_dims[0], n2_dims=input_dims[0])
    agent = Agent(gamma=gamma, epsilon=epsilon, batch_size=10, n_actions=n_actions, input_dims=input_dims, lr=lr)
    b_m = T.load('best_model.pth.tar')
    model.load_state_dict(b_m['state_dict'])
    model.optimizer.load_state_dict(b_m['optimizer'])
    model.eval()
    agent.Q_eval = model
    agent.Q_next = model
    agent.loss = b_m['loss']
    agent.epsilon = b_m['epsilon']
    agent.state_memory = b_m['state_memory']
    agent.next_state_memory = b_m['next_state_memory']
    agent.action_memory = b_m['action_memory']
    agent.reward_memory = b_m['reward_memory']
    agent.done_memory = b_m['done_memory']
    agent.gamma = b_m['gamma']
    agent.batch_size = b_m['batch_size']
    agent.lr = b_m['lr']
    return agent



class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_memory=300000, eps_end=0.0, eps_dec=15e-5):
        self.gamma = gamma  # Discount factor. It’s used to balance immediate and future reward.
        self.epsilon = epsilon  # Chance to take random action
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr  # How much you accept the new value vs the old value
        self.action_space = [i for i in range(n_actions)]  # All possible actions
        self.mem_size = max_memory  # We don't want to make it infinity cause it's will be leak of memory. We doing it with max size, and overrating it.
        self.batch_size = batch_size
        self.mem_iter = 0
        self.counter = 0
        self.replace_target = 25 # 50

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                   n1_dims=input_dims[0], n2_dims=input_dims[0])
        self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                   n1_dims=input_dims[0], n2_dims=input_dims[0])  # Q values of future states
        self.loss = T.tensor([0, ])
        self.Q_next.eval()

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)  # For storing done flags

    def replay_memory(self, state, action, reward, next_state, flag):  # +
        try:
            i = self.mem_iter % self.mem_size  # Overrating memory
            self.state_memory[i] = state
            self.next_state_memory[i] = next_state
            self.reward_memory[i] = reward
            self.action_memory[i] = action
            self.done_memory[i] = flag
            self.mem_iter += 1
        except:
            print()

    def choose_action(self, observation):  # +
        action_type = ''
        state = T.tensor(observation, dtype=T.float32).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)  # Set of actions with Q value at each
        if np.random.random() > self.epsilon:
            action = T.argmax(actions).item()  # choosing action with max Q value. Exploit
            action_type = 'G'
        else:
            action = np.random.choice(self.action_space)  # random action. For explore
            action_type = 'R'

        return {'action': action, 'action_type': action_type, 'epsilon': self.epsilon, 'actions': actions,
                'loss': self.loss}

    def learn(self):
        if self.mem_iter < self.batch_size:  # If memory length < then batch_size
            return

        self.Q_eval.optimizer.zero_grad()

        # Forming batches
        batch = np.random.choice(min(self.mem_iter, self.mem_size), self.batch_size,
                                 replace=False)  # Random selection from max_mem list. Amount of needed elements = batch_size
        index = np.arange(self.batch_size, dtype=np.int32)  # Return list of number at 0 to batch_size

        # Receiving values from memory of states, new_states and e.t.c.
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.next_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]  # It can be just numpy array
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.done_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[index, action_batch]  # Value of taken action
        q_next = self.Q_eval.forward(new_state_batch)  # Value of next action
        q_next[terminal_batch] = 0.0  # Q value for losing action

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        self.loss = F.mse_loss(q_target, q_eval).to(self.Q_eval.device)
        self.Q_eval.optimizer.zero_grad()
        self.loss.backward()
        self.Q_eval.optimizer.step()

        self.counter += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

        if self.counter % self.replace_target == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())  # Copy hyper parameters
