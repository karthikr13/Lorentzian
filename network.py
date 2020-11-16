import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Network(nn.Module):
    def __init__(self, low, high, num_points, layer_sizes, num_osc):
        super(Network, self).__init__()
        self.num_points = num_points
        self.num_osc = num_osc
        self.op = None

        w = np.arange(low, high, (high - low) / self.num_points)
        self.w = torch.tensor(w)
        self.epsilon_inf = torch.tensor([5 + 0j], dtype=torch.cfloat)
        if torch.cuda.is_available():
            self.w = self.w.cuda()

        self.linears, self.batch_norms = [], []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            self.linears.append(nn.Linear(in_size, out_size))
            self.batch_norms.append(nn.BatchNorm1d(out_size))
        self.w0 = nn.Linear(layer_sizes[-1], self.num_osc)
        self.g = nn.Linear(layer_sizes[-1], self.num_osc)
        self.wp = nn.Linear(layer_sizes[-1], self.num_osc)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.batch_norms[i](self.linears[i](x)))
        x = self.batch_norms[-1](self.linears[-1](x))
        batch_size = x.size()[0]

        w0 = F.relu(self.w0(x))
        g = F.relu(self.g(x))
        wp = F.relu(self.wp(x))
        out = [w0, wp, g]

        w0 = w0.unsqueeze(2)
        g = g.unsqueeze(2) * 0.1
        wp = wp.unsqueeze(2)

        w0 = w0.expand(batch_size, self.num_osc, self.num_points)
        g = g.expand(batch_size, self.num_osc, self.num_points)
        wp = wp.expand(batch_size, self.num_osc, self.num_points)
        w = self.w.expand(batch_size, self.num_osc, self.num_points)

        # calculate T = e2
        numerator = torch.mul(torch.mul(w, g), torch.square(wp))
        denom = torch.add(torch.square(torch.sub(torch.square(w0), torch.square(w))), torch.mul(torch.square(w), torch.square(g)))

        T = e2 = torch.div(numerator, denom)

        return (T, *out)

