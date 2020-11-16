from network import Network

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
class NetworkWrapper():
    def __init__(self, low, high, num_points, layer_sizes, num_osc, opt, decay):
        self.model = Network(low, high, num_points, layer_sizes, num_osc)
        self.init_opt(opt)
        self.lr = lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='min',
                                       factor=decay,
                                       patience=10, verbose=True, threshold=1e-4)

    def init_opt(self, opt):
        if opt == 'Adam':
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif opt == 'AdamW':
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif opt == 'Adamax':
            self.opt = torch.optim.Adamax(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif opt == 'SparseAdam':
            self.opt = torch.optim.SparseAdam(self.model.parameters(), lr=self.flags.lr)
        elif opt == 'RMSprop':
            self.opt = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif opt == 'SGD':
            self.opt = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale, momentum=0.9, nesterov=True)
        elif opt == 'LBFGS':
            self.opt = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise Exception("Optimizer is not available at the moment.")

    def train_network(self, epochs):
        if torch.cuda.is_available():
            self.model.cuda()

        for epoch in range(epochs):
            train_loss, eval_loss = [], []
            self.model.train()
            for geometry, spectra in data:
                if torch.cuda.is_available():
                    geometry.cuda()
                    spectra.cuda()
                self.opt.zero_grad()
                out, w0, wp, g = self.model(geometry)
                loss = F.mse_loss(out, spectra)
                train_loss.append(np.copy(loss.cpu().data.numpy()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.opt.step()

                self.model.eval()
                out, w0, wp, g = self.model(geometry)
                loss = F.mse_loss(out, spectra)
                eval_loss.append(np.copy(loss.cpu().data.numpy()))
            mean_train_loss = np.mean(train_loss)
            mean_eval_loss = np.mean(eval_loss)
            self.lr.step(mean_train_loss)
