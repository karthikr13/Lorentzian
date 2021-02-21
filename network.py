"""
This is the module where the model is defined. It uses the nn.Module as a backbone to create the network structure
"""

import math
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt, square


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

        """
        General layer definitions:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1], bias=True))
            # torch.nn.init.uniform_(self.linears[ind].weight, a=1, b=2)

            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1], track_running_stats=True, affine=True))

        layer_size = flags.linear[-1]

        # Last layer is the Lorentzian parameter layer
        self.lin_w0 = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_wp = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_g = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.use_lorentz = True


    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G

        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            #print(out.size())
            if ind < len(self.linears) - 0:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
            else:
                out = bn(fc(out))

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:

            w0 = F.relu(self.lin_w0(F.relu(out)))
            wp = F.relu(self.lin_wp(F.relu(out)))
            g = F.relu(self.lin_g(F.relu(out)))

            w0_out = w0
            wp_out = wp
            g_out = g

            w0 = w0.unsqueeze(2) * 1
            wp = wp.unsqueeze(2) * 1
            g = g.unsqueeze(2) * 0.1

             # Expand them to parallelize, (batch_size, #osc, #spec_point)
            wp = wp.expand(out.size()[0], self.flags.num_lorentz_osc, self.flags.num_spec_points)
            w0 = w0.expand_as(wp)
            g = g.expand_as(w0)
            w_expand = self.w.expand_as(g)

            # Define dielectric function (real and imaginary parts separately)
            num1 = mul(square(wp), add(square(w0), -square(w_expand)))
            num2 = mul(square(wp), mul(w_expand, g))
            denom = add(square(add(square(w0), -square(w_expand))), mul(square(w_expand), square(g)))
            e1 = div(num1, denom)
            e2 = div(num2, denom)

            # self.e2 = e2.data.cpu().numpy()                 # This is for plotting the imaginary part
            # # self.e1 = e1.data.cpu().numpy()                 # This is for plotting the imaginary part

            e1 = torch.sum(e1, 1).type(torch.float)
            e2 = torch.sum(e2, 1).type(torch.float)

            T = e2.float()

            return T, w0_out, wp_out, g_out

        return out,out,out,out
