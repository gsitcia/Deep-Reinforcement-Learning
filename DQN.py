import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from networks import *
from replay import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def s2t(state):
    return torch.cat(state).unsqueeze(0)

class DQN:
    def __init__(self, env, hparams):
        self.hparams = hparams
        self.env = env
        self.n = env.action_space.n
        self.Q = DCNN(4,self.n)
        self.T = DCNN(4,self.n)
        self.T.load_state_dict(self.Q.state_dict())
        self.T.eval()
        self.memory = ReplayMemory(hparams.memory_size)
        self.steps = 0
        self.state = env.reset()
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=hparams.lr, momentum=hparams.momentum)
        self.n_episodes = 0
    @torch.no_grad()
    def select_action(self):
        hparams = self.hparams
        start = hparams.eps_start
        end = hparams.eps_end
        time = hparams.eps_time
        steps = self.steps
        self.steps += 1
        if steps < time:
            epsilon = start - (start-end)*steps/time
        else:
            epsilon = end

        sample = random.random()

        if sample > epsilon:
            return self.Q(s2t(self.state).to(device)).max(1)[1].item()
        else:
            return self.env.action_space.sample()
    def sample_step(self, fs_min=2, fs_max=6):
        """repeats a single action between fs_min and fs_max (inclusive) times"""
        fs = random.randint(fs_min,fs_max)
        action = self.select_action()
        r = 0
        for _ in range(fs):
            new_state,reward,done,_ = self.env.step(action)
            self.memory.push(self.state, action, new_state if not done else None, reward)
            r += reward
            self.state = self.env.reset() if done else new_state
            if done:
                self.n_episodes += 1
        return r
    def optimize(self):
        hparams = self.hparams
        transitions = self.memory.sample(hparams.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.cat([s2t(state) for state in batch.state]).to(device)
        actions = torch.tensor(batch.action).unsqueeze(1).to(device)
        target_values = torch.tensor(batch.reward).unsqueeze(1).to(device).float()
        non_terminal_next_states = torch.cat([s2t(state) for state in batch.next_state if state is not None]).to(device)
        non_terminal_mask = torch.tensor([state is not None for state in batch.next_state]).to(device).unsqueeze(1)

        values = self.Q(states).gather(1,actions).float()
        target_values[non_terminal_mask] += hparams.gamma * self.T(non_terminal_next_states).detach().max(1)[0].float()

        #print(values.dtype,target_values.dtype)
        loss = F.smooth_l1_loss(values,target_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1,1) # maybe try sign_?

        self.optimizer.step()
        return loss

class DDQN(DQN):
    def optimize(self):
        hparams = self.hparams
        transitions = self.memory.sample(hparams.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.cat([s2t(state) for state in batch.state]).to(device)
        actions = torch.tensor(batch.action).unsqueeze(1).to(device)
        target_values = torch.tensor(batch.reward).unsqueeze(1).to(device).float()
        non_terminal_next_states = torch.cat([s2t(state) for state in batch.next_state if state is not None]).to(device)
        non_terminal_mask = torch.tensor([state is not None for state in batch.next_state]).to(device).unsqueeze(1)

        values = self.Q(states).gather(1,actions).float()
        a = self.Q(non_terminal_next_states).max(1)[1].unsqueeze(1)
        b = self.T(non_terminal_next_states).detach()
        target_values[non_terminal_mask] += hparams.gamma * b.gather(1,a).flatten()

        #print(values.dtype,target_values.dtype)
        loss = F.smooth_l1_loss(values,target_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1,1) # maybe try sign_?

        self.optimizer.step()
        return loss

class RND1(DDQN):
    def __init__(self, env, hparams):
        super().__init__(env, hparams)
        self.R = FCC()
        self.RP = FCC()
        self.R.eval()
        self.RP_optimizer = torch.optim.RMSprop(self.RP.parameters(), lr=hparams.lr, momentum=hparams.momentum)
    def intrinsic(self):
        A = self.R(self.state[0].unsqueeze(0)).detach()
        P = self.RP(self.state[0].unsqueeze(0))
        loss = F.mse_loss(P,A)
        self.RP_optimizer.zero_grad()
        loss.backward()
        self.RP_optimizer.step()
        return loss
    def sample_step(self, fs_min=2, fs_max=6):
        fs = random.randint(fs_min,fs_max)
        action = self.select_action()
        for _ in range(fs):
            new_state,reward,done,_ = self.env.step(action)
            self.memory.push(self.state, action, new_state if not done else None, reward+self.intrinsic())
            self.state = self.env.reset() if done else new_state
            if done:
                self.n_episodes += 1

class RND(DQN):
    def __init__(self, env, hparams):
        super().__init__(env, hparams)
        self.R = SMM()
        self.RP = SMM()
        self.R.eval()
        self.RP_optimizer = torch.optim.RMSprop(self.RP.parameters(), lr=hparams.lr, momentum=hparams.momentum)
    def intrinsic(self,states):
        A = self.R(states).detach()
        P = self.RP(states)
        loss = F.mse_loss(P,A)
        self.RP_optimizer.zero_grad()
        loss.backward()
        self.RP_optimizer.step()
        return F.mse_loss(P,A,reduction='none').detach()
    def optimize(self):
        hparams = self.hparams
        transitions = self.memory.sample(hparams.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.cat([s2t(state) for state in batch.state]).to(device)
        actions = torch.tensor(batch.action).unsqueeze(1).to(device)
        target_values = torch.tensor(batch.reward).unsqueeze(1).to(device).float()+self.intrinsic(states[:,:1,:,:])
        non_terminal_next_states = torch.cat([s2t(state) for state in batch.next_state if state is not None]).to(device)
        non_terminal_mask = torch.tensor([state is not None for state in batch.next_state]).to(device).unsqueeze(1)

        values = self.Q(states).gather(1,actions).float()
        a = self.Q(non_terminal_next_states).max(1)[1].unsqueeze(1)
        b = self.T(non_terminal_next_states).detach()
        target_values[non_terminal_mask] += hparams.gamma * b.gather(1,a).flatten()

        #print(values.dtype,target_values.dtype)
        loss = F.smooth_l1_loss(values,target_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1,1) # maybe try sign_?

        self.optimizer.step()
        return loss