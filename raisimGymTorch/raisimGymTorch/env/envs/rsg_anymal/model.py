from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, device, ob_dim, act_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(ob_dim * 4 + act_dim, 128, device = device)
        self.fc2 = nn.Linear(128, 128, device = device)
        self.fc3 = nn.Linear(128, 128, device = device)
        self.fc4 = nn.Linear(128, 64, device = device)
        self.fc5 = nn.Linear(64, ob_dim, device = device)
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, history_obs, act):
        for i in range(4):
            history_obs[i] = torch.from_numpy(history_obs[i]).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        obs = torch.cat(history_obs, dim = 1)
        x = torch.cat([obs, act], dim = 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        for i in range(4):
            history_obs[i] = history_obs[i].detach().cpu().numpy()
        return x
    
    def criterion(self, res, pred):
        res = torch.from_numpy(res).to(self.device)
        return torch.mean((res - pred)**2)
    
    def update(self, obs, act, res):
        self.optimizer.zero_grad()
        pred = self.forward(obs, act)
        loss = self.criterion(res, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()