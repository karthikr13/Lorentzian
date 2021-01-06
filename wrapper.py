from network import Network

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import matplotlib
from torch import nn
matplotlib.use('agg')
import matplotlib.pyplot as plt


class NetworkWrapper():
    def __init__(self, flags, train, test):
        self.flags = flags
        self.model = Network(flags)
        self.lr = None
        self.train_data = train
        self.test_data = test
        self.best_loss = [float('inf')] * 2

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
            self.opt = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale,
                                       momentum=0.9, nesterov=True)
        elif opt == 'LBFGS':
            self.opt = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise Exception("Optimizer is not available at the moment.")

    def reset_lr(self, optm):
        """
        Reset the learning rate to to original lr
        :param optm: The optimizer
        :return: None
        """
        self.lr = lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='min',
                                                 factor=self.flags.lr_decay_rate,
                                                 patience=10, verbose=True, threshold=1e-4)
        for g in optm.param_groups:
            g['lr'] = self.flags.lr

    def train_stuck_by_lr(self, optm, lr_limit):
        """
        Detect whether the training is stuck with the help of LR scheduler which decay when plautue
        :param optm: The optimizer
        :param lr_limit: The limit it judge it is stuck
        :return: Boolean value of whether it is stuck
        """

        for g in optm.param_groups:
            if g['lr'] < lr_limit:
                return True
            else:
                return False

    def train_network(self):
        if torch.cuda.is_available():
            self.model.cuda()
        self.init_opt(self.flags.optim)

        self.init_weights()
        self.lr = lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='min',
                                                 factor=self.flags.lr_decay_rate,
                                                 patience=10, verbose=True, threshold=1e-4)

        train_err, test_err = [], []
        cuda = torch.cuda.is_available()
        for epoch in range(self.flags.train_step):
            train_loss, eval_loss = [], []
            self.model.train()
            for geometry, spectra in self.train_data:
                self.model.train()
                if cuda:
                    geometry.cuda()
                    spectra = spectra.cuda()
                self.opt.zero_grad()
                out, w0, wp, g = self.model(geometry)
                loss = F.mse_loss(out, spectra)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                train_loss.append(np.copy(loss.cpu().data.numpy()))
                self.opt.step()

                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_data):  # Loop through the eval set
                        if cuda:
                            geometry.cuda()
                            spectra = spectra.cuda()
                        logit, w0, wp, g = self.model(geometry)  # Get the output
                        # loss = self.local_lorentz_loss(w0, g, wp, logit,spectra, record)
                        loss = F.mse_loss(logit, spectra)  # compute the loss
                        # loss = self.make_custom_loss(logit, spectra)
                        eval_loss.append(np.copy(loss.cpu().data.numpy()))  # Aggregate the loss
                self.model.eval()
                out, w0, wp, g = self.model(geometry)
                loss = F.mse_loss(out, spectra)
                eval_loss.append(np.copy(loss.cpu().data.numpy()))
            mean_train_loss = np.mean(train_loss)
            mean_eval_loss = np.mean(eval_loss)
            train_err.append(mean_train_loss)
            test_err.append(mean_eval_loss)
            self.lr.step(mean_train_loss)
            if self.flags.use_warm_restart:
                if epoch % self.flags.lr_warm_restart == 0:
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = self.flags.lr / 10
                        print('Resetting learning rate to %.5f' % self.flags.lr)
            if epoch % 10 == 0:
                print("Mean train loss for epoch {}: {}".format(epoch, mean_train_loss))
                print("Mean eval for epoch {}: {}".format(epoch, mean_eval_loss))
            self.best_loss[0] = min(self.best_loss[0], mean_train_loss)
            self.best_loss[1] = min(self.best_loss[1], mean_eval_loss)

        x = list(range(self.flags.train_step))
        plt.clf()
        plt.title('Best Train Error: {}, Best Eval Error: {}'.format(self.best_loss[0], self.best_loss[1]))
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.plot(x, train_err, label='Training Data')
        plt.plot(x, test_err, label='Eval')
        plt.legend()
        plt.savefig('hypersweep2/{}'.format(self.flags.model_name + '.png'))

        return self.best_loss


    def compare_spectra(self, truth, pred, name,w0,wp,g):
        """
        A function to plot the compared spectra during training
        :param truth:
        :param pred:
        :return:
        """
        plt.close()
        f = plt.figure()
        plt.plot(truth,label='truth')
        plt.plot(pred,label='pred')
        plt.legend()
        plt.title('w0={},wp={},g={}'.format(w0,wp,g))
        plt.savefig('plots/'+name+'.png')

    def init_weights(self):
        for layer_name, child in self.model.named_children():
            print('layer_name = ', layer_name, 'child = ', child)
            for param in self.model.parameters():
                if ('w0' in layer_name or 'wp' in layer_name):
                    print("w0 or wp gets init")
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.1)
                elif ('g' in layer_name):
                    print("g gets init")
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.1)
                else:
                    if ((type(child) == nn.Linear) | (type(child) == nn.Conv2d)):
                        #print('this gets initialized:', param)
                        torch.nn.init.xavier_uniform_(child.weight)
                        if child.bias:
                            child.bias.data.fill_(0.00)

    def train_network_ascent(self):
        if torch.cuda.is_available():
            self.model.cuda()
        self.init_opt(self.flags.optim)
        self.init_weights()
        self.lr = lr_scheduler.ReduceLROnPlateau(optimizer=self.opt, mode='min',
                                                 factor=self.flags.lr_decay_rate,
                                                 patience=10, verbose=True, threshold=1e-4)
        train_err, test_err = [], []
        cuda = torch.cuda.is_available()
        epoch = 0
        gd = True
        while epoch < self.flags.train_step:
            gd = True
            if not gd:
                print('Epoch {} using gradient ascent'.format(epoch))
            train_loss, eval_loss = [], []
            self.model.train()
            ind = 0
            for geometry, spectra in self.train_data:
                ind += 1
                self.model.train()
                if cuda:
                    geometry.cuda()
                    spectra = spectra.cuda()
                #print("Geometry snapshot:")
                #print(geometry.cpu().numpy()[0:10, :])
                self.opt.zero_grad()
                out, w0, wp, g = self.model(geometry)
                if ind == 1 and epoch % 20 == 1:
                    self.compare_spectra(spectra.cpu().numpy()[0, :] ,out.detach().cpu().numpy()[0,:], 'epoch_{}'.format(epoch),
                                         w0=w0.detach().cpu().numpy()[0,:], wp=wp.detach().cpu().numpy()[0,:],
                                         g=g.detach().cpu().numpy()[0,:])

                loss = F.mse_loss(out, spectra)
                if not gd:
                    loss *= -0.01
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.opt.step()
                train_loss.append(np.copy(loss.cpu().data.numpy()))
                """
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_data):  # Loop through the eval set
                        if cuda:
                            geometry.cuda()
                            spectra = spectra.cuda()
                        logit, w0, wp, g = self.model(geometry)  # Get the output
                        # loss = self.local_lorentz_loss(w0, g, wp, logit,spectra, record)
                        loss = F.mse_loss(logit, spectra)  # compute the loss
                        # loss = self.make_custom_loss(logit, spectra)

                        eval_loss.append(np.copy(loss.cpu().data.numpy()))  # Aggregate the loss
                """
                self.model.eval()
                out, w0, wp, g = self.model(geometry)
                loss = F.mse_loss(out, spectra)
                eval_loss.append(np.copy(loss.cpu().data.numpy()))
            mean_train_loss = np.mean(train_loss)
            #if not gd:
            #    mean_train_loss *= -1
            mean_eval_loss = np.mean(eval_loss)
            train_err.append(mean_train_loss)
            test_err.append(mean_eval_loss)

            if gd:
                self.lr.step(mean_train_loss)
                if mean_train_loss > 0.001:
                    # If the LR changed (i.e. training stuck) and also loss is large
                    if self.train_stuck_by_lr(self.opt, self.flags.lr/8):
                        # Switch to the gradient ascend mode
                        gd = False
                else:
                    print(mean_train_loss)
                    x = list(range(len(train_err)))
                    plt.clf()
                    plt.figure(1)
                    plt.title(
                        'Best Train Error: {}, Best Eval Error: {}'.format(self.best_loss[0], self.best_loss[1]))
                    plt.xlabel('Epoch')
                    plt.ylabel('MSE')
                    plt.plot(x, train_err, label='Training Data')
                    plt.plot(x, test_err, label='Eval')
                    plt.legend()
                    plt.savefig('hypersweep_ga/{}'.format(self.flags.model_name + '.png'))
                    break
            else:
                gd = True
                self.reset_lr(self.opt)
            if epoch % 20 == 0:
                print("Mean train loss for epoch {}: {}".format(epoch, mean_train_loss))
                print("Mean eval for epoch {}: {}".format(epoch, mean_eval_loss))
            epoch += 1
            self.best_loss[0] = min(self.best_loss[0], mean_train_loss)
            self.best_loss[1] = min(self.best_loss[1], mean_eval_loss)

        x = list(range(epoch))
        plt.clf()
        plt.figure(1)
        plt.title(
            'Best Train Error: {}, Best Eval Error: {}'.format(self.best_loss[0], self.best_loss[1]))
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.plot(x, train_err, label='Training Data')
        plt.plot(x, test_err, label='Eval')
        plt.legend()
        plt.savefig('plots/training_curve.png')

        return self.best_loss
