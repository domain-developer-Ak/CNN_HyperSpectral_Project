import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_capsules, num_routes):
        super(CapsuleLayer, self).__init__()
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, input_dim, output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x)
        return self.squash(u_hat)

    def squash(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (norm ** 2 / (1 + norm ** 2)) * x / (norm + 1e-8)

class CapsuleNetwork(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 256, kernel_size=9, stride=1, padding=0)
        self.primary_caps = CapsuleLayer(256, 32, num_capsules=8, num_routes=-1)
        self.digit_caps = CapsuleLayer(32, num_classes, num_capsules=16, num_routes=8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        return x
