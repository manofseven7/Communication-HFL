import torch, torch.nn as nn, copy

class MLP(nn.Module):
    def __init__(self, d, hidden, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, out))
    def forward(self, x): return self.net(x)

class CIFAR_CNN(nn.Module):   # EXACT arch that produced the reported CIFAR numbers
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        return self.fc2(torch.relu(self.fc1(x)))

def build_model(cfg, d, C):
    if cfg.dataset == "cifar":
        return CIFAR_CNN()
    if cfg.method == "fedkd":
        return FedKD_Student(d, C)
    return MLP(d, cfg.mlp_hidden, C)

# ---- FedKD teacher / student (disclosed adaptation) ----
class FedKD_Student(nn.Module):
    def __init__(self, d, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, out))
    def forward(self, x): return self.net(x)

class FedKD_Teacher(nn.Module):
    def __init__(self, d, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, out))
    def forward(self, x): return self.net(x)
