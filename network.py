import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, mul, div
import numpy as np


class Network(nn.Module):
    def __init__(self, flags):
        super(Network, self).__init__()
        self.flags = flags
        self.num_points = self.flags.num_spec_points
        self.num_osc = self.flags.num_lorentz_osc
        self.op = None
        self.low = self.flags.freq_low
        self.high = self.flags.freq_high

        w = np.arange(self.low, self.high, (self.high - self.low) / self.num_points)
        self.w = torch.tensor(w)
        if torch.cuda.is_available():
            self.w = self.w.cuda()

        self.linears, self.batch_norms = [], []
        for i in range(len(self.flags.linear) - 1):
            in_size = self.flags.linear[i]
            out_size = self.flags.linear[i + 1]
            self.linears.append(nn.Linear(in_size, out_size))
            self.batch_norms.append(nn.BatchNorm1d(out_size))
        self.w0 = nn.Linear(self.flags.linear[-1], self.num_osc)
        self.g = nn.Linear(self.flags.linear[-1], self.num_osc)
        self.wp = nn.Linear(self.flags.linear[-1], self.num_osc)
        if torch.cuda.is_available():
            self.w = self.w.cuda()
            self.w0 = self.w0.cuda()
            self.g = self.g.cuda()
            self.wp = self.wp.cuda()

    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = F.relu(self.batch_norms[i](self.linears[i](x)))
        x = self.batch_norms[-1](self.linears[-1](x))
        batch_size = x.size()[0]
        if torch.cuda.is_available():
            x = x.cuda()
        w0 = F.relu(self.w0(F.relu(x)))
        g = F.relu(self.g(F.relu(x)))
        wp = F.relu(self.wp(F.relu(x)))

        out = [w0, g, wp]
        w0 = w0.unsqueeze(2)
        g = g.unsqueeze(2) * 0.1
        wp = wp.unsqueeze(2)

        wp = wp.expand(batch_size, self.num_osc, self.num_points).float()
        w0 = w0.expand_as(wp).float()
        g = g.expand_as(w0).float()
        w_ex = self.w.expand_as(g).float()

        num = mul(mul(wp, wp), mul(w_ex, g))
        denom = add(mul(add(mul(w0, w0), -mul(w_ex, w_ex)), add(mul(w0, w0), -mul(w_ex, w_ex))), mul(mul(w_ex, w_ex), mul(g, g)))
        e2 = div(num, denom)
        e2 = torch.sum(e2, 1).type(torch.float)

        T = e2.float()
        if torch.cuda.is_available():
            T.cuda()
        return (T, *out)
