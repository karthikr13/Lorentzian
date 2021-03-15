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
            self.optm = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif opt == 'AdamW':
            self.optm = torch.optim.AdamW(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif opt == 'Adamax':
            self.optm = torch.optim.Adamax(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif opt == 'SparseAdam':
            self.optm = torch.optim.SparseAdam(self.model.parameters(), lr=self.flags.lr)
        elif opt == 'RMSprop':
            self.optm = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif opt == 'SGD':
            self.optm = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale,
                                       momentum=0.9, nesterov=True)
        elif opt == 'LBFGS':
            self.optm = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise Exception("Optimizer is not available at the moment.")

    def reset_lr(self, optm):
        """
        Reset the learning rate to to original lr
        :param optm: The optimizer
        :return: None
        """
        self.lr = lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
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
    def make_MSE_loss(self, logit=None, labels=None):
        """
        Create a tensor that represents the loss. This is consistent both at training time \
        and inference time for a backward model
        :param logit: The output of the network
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss of the network
        return MSE_loss
    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed.
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'AdamW':
            op = torch.optim.AdamW(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'Adamax':
            op = torch.optim.Adamax(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SparseAdam':
            op = torch.optim.SparseAdam(self.model.parameters(), lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale, momentum=0.9, nesterov=True)
        elif self.flags.optim == 'LBFGS':
            op = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise Exception("Optimizer is not available at the moment.")
        return op

    def make_lr_scheduler(self):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        # return lr_scheduler.StepLR(optimizer=self.optm, step_size=50, gamma=0.75, last_epoch=-1)
        return lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
                                        factor=self.flags.lr_decay_rate,
                                          patience=10, verbose=True, threshold=1e-4)

    def train_network(self):
        print("Starting training process")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        # self.record_weight(name='start_of_train', batch=0, epoch=0)

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler()

        self.init_weights()

        cuda = torch.cuda.is_available()
        train_err = []
        for epoch in range(self.flags.train_step):
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []

            self.model.train()
            for (geometry, spectra) in self.train_data:
                if cuda:
                    geometry = geometry.cuda()  # Put data onto GPU
                    spectra = spectra.cuda()  # Put data onto GPU

                self.optm.zero_grad()  # Zero the gradient first
                # logit = self.model(geometry)
                # loss = self.make_MSE_loss(logit, spectra)

                logit, w0, wp, g = self.model(geometry)  # Get the output
                # loss = self.local_lorentz_loss(w0,g,wp,logit,spectra,record)
                loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                loss.backward()

                # Calculate the backward gradients
                # self.record_weight(name='after_backward', batch=j, epoch=epoch)

                # Clip gradients to help with training
                if self.flags.use_clip:
                    if self.flags.use_clip:
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.flags.grad_clip)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.grad_clip)

                self.optm.step()  # Move one step the optimizer
                # self.record_weight(name='after_optm_step', batch=j, epoch=epoch)

                train_loss.append(np.copy(loss.cpu().data.numpy()))  # Aggregate the loss
                #self.running_loss.append(np.copy(loss.cpu().data.numpy()))
                # #############################################
                # # Extra test for err_test < err_train issue #
                # #############################################
                self.model.eval()
                logit, w0, wp, g = self.model(geometry)  # Get the output
                # loss = self.local_lorentz_loss(w0,g,wp,logit,spectra,record)
                # loss = self.make_custom_loss(logit, spectra)
                loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                train_loss_eval_mode_list.append(np.copy(loss.cpu().data.numpy()))
                self.model.train()

            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)
            train_err.append(train_avg_eval_mode_loss)
            if epoch % self.flags.eval_step == 0:  # For eval steps, do the evaluations and tensor board
                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_data):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        logit, w0, wp, g = self.model(geometry)  # Get the output
                        # loss = self.local_lorentz_loss(w0, g, wp, logit,spectra, record)
                        loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                        # loss = self.make_custom_loss(logit, spectra)

                        test_loss.append(np.copy(loss.cpu().data.numpy()))  # Aggregate the loss

                test_avg_loss = np.mean(test_loss)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_eval_mode_loss, test_avg_loss))

            # # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
            # # self.lr_scheduler.step()

            if self.flags.use_warm_restart:
                if epoch % self.flags.lr_warm_restart == 0:
                    for param_group in self.optm.param_groups:
                        param_group['lr'] = self.flags.lr/10
                        print('Resetting learning rate to %.5f' % self.flags.lr)

        x = list(range(self.flags.train_step))
        best = min(train_err)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.plot(x, train_err)
        plt.title("best error: {}".format(best))
        plt.legend()
        plt.savefig('hypersweep5/{}'.format('gd.png'))

    def train_network_2(self):
        if torch.cuda.is_available():
            self.model.cuda()
        self.init_opt(self.flags.optim)

        self.init_weights()
        self.lr = lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
                                                 factor=self.flags.lr_decay_rate,
                                                 patience=10, verbose=True, threshold=1e-4)

        train_err, test_err = [], []
        cuda = torch.cuda.is_available()
        for epoch in range(self.flags.train_step):
            train_loss = []
            eval_loss = []
            self.model.train()
            for (geometry, spectra) in self.train_data:
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()  # Put data onto GPU

                self.optm.zero_grad()  # Zero the gradient first
                # logit = self.model(geometry)
                # loss = self.make_MSE_loss(logit, spectra)

                logit, w0, wp, g = self.model(geometry)  # Get the output
                # loss = self.local_lorentz_loss(w0,g,wp,logit,spectra,record)
                loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                loss.backward()

                # Clip gradients to help with training
                if self.flags.use_clip:
                    if self.flags.use_clip:
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.flags.grad_clip)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.grad_clip)
                train_loss.append(np.copy(loss.cpu().data.numpy()))

                self.optm.step()
                self.model.eval()
                loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                eval_loss.append(np.copy(loss.cpu().data.numpy()))
                self.model.train()

            mean_train_loss = np.mean(train_loss)
            mean_eval_loss = np.mean(eval_loss)
            train_err.append(mean_train_loss)
            test_err.append(mean_eval_loss)

            if epoch % self.flags.eval_step == 0:  # For eval steps, do the evaluations and tensor board
                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_data):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        logit, w0, wp, g = self.model(geometry)  # Get the output
                        # loss = self.local_lorentz_loss(w0, g, wp, logit,spectra, record)
                        loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                        # loss = self.make_custom_loss(logit, spectra)

                        test_loss.append(np.copy(loss.cpu().data.numpy()))  # Aggregate the loss
                    test_avg_loss = np.mean(test_loss)

                    print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                          % (epoch, mean_train_loss, mean_eval_loss))
            self.lr_scheduler.step(mean_train_loss)

            if self.flags.use_warm_restart:
                if epoch % self.flags.lr_warm_restart == 0:
                    for param_group in self.optm.param_groups:
                        param_group['lr'] = self.flags.lr / 10
                        print('Resetting learning rate to %.5f' % self.flags.lr)
            if epoch % 10 == 0:
                print("Mean train loss for epoch {}: {}".format(epoch, mean_train_loss))
                print("Mean eval for epoch {}: {}".format(epoch, mean_eval_loss))
            self.best_loss[0] = min(self.best_loss[0], mean_train_loss)
            self.best_loss[1] = min(self.best_loss[1], mean_eval_loss)

        x = list(range(self.flags.train_step))

        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        label = self.flags.model_name.split('_')[2:]
        plt.plot(x, train_err, label=label)

        plt.legend()
        plt.savefig('hypersweep4/{}'.format(self.flags.model_name + '.png'))

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
            for param in self.model.parameters():
                if ('w0' in layer_name or 'wp' in layer_name):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.1)
                elif ('g' in layer_name):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.1)
                else:
                    if ((type(child) == nn.Linear) | (type(child) == nn.Conv2d)):
                        #print('this gets initialized:', param)
                        torch.nn.init.xavier_uniform_(child.weight)
                        if child.bias:
                            child.bias.data.fill_(0.00)

    def train_network_3(self):
        print("Starting training process")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        # self.record_weight(name='start_of_train', batch=0, epoch=0)

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler()

        self.init_weights()

        cuda = torch.cuda.is_available()
        epoch = 0
        gd = True
        ascent_losses = {}
        train_err, test_err = [], []
        pos_train_losses = []
        while epoch < self.flags.train_step:
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()
            for (geometry, spectra) in self.train_data:
                if cuda:
                    geometry = geometry.cuda()  # Put data onto GPU
                    spectra = spectra.cuda()  # Put data onto GPU

                self.optm.zero_grad()  # Zero the gradient first
                # logit = self.model(geometry)
                # loss = self.make_MSE_loss(logit, spectra)

                logit, w0, wp, g = self.model(geometry)  # Get the output
                # loss = self.local_lorentz_loss(w0,g,wp,logit,spectra,record)
                loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                if not gd:
                    loss *= -1 * self.flags.strength

                loss.backward()

                if self.flags.use_clip:
                    if self.flags.use_clip:
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.flags.grad_clip)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.grad_clip)
                train_loss.append(np.copy(loss.cpu().data.numpy()))  # Aggregate the loss

                self.optm.step()  # Move one step the optimizer
                # self.record_weight(name='after_optm_step', batch=j, epoch=epoch)


                #self.running_loss.append(np.copy(loss.cpu().data.numpy()))
                # #############################################
                # # Extra test for err_test < err_train issue #
                # #############################################
                self.model.eval()
                logit, w0, wp, g = self.model(geometry)  # Get the output
                # loss = self.local_lorentz_loss(w0,g,wp,logit,spectra,record)
                # loss = self.make_custom_loss(logit, spectra)
                loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                train_loss_eval_mode_list.append(np.copy(loss.cpu().data.numpy()))
                self.model.train()
            train_avg_loss = np.mean(train_loss)
            train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)
            train_err.append(train_avg_loss)
            test_err.append(train_avg_eval_mode_loss)
            if epoch == 0 or epoch % self.flags.eval_step == 0:  # For eval steps, do the evaluations and tensor board
                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_data):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        logit, w0, wp, g = self.model(geometry)  # Get the output
                        # loss = self.local_lorentz_loss(w0, g, wp, logit,spectra, record)
                        loss = self.make_MSE_loss(logit, spectra)  # compute the loss
                        # loss = self.make_custom_loss(logit, spectra)

                        test_loss.append(np.copy(loss.cpu().data.numpy()))  # Aggregate the loss

                test_avg_loss = np.mean(test_loss)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_eval_mode_loss, test_avg_loss))
            # Calculate the avg loss of training

            if gd:
                pos_train_losses.append(train_avg_loss)
            if gd:
                self.lr_scheduler.step(train_avg_loss)
                self.best_loss[0] = min(self.best_loss[0], test_avg_loss)
                self.best_loss[1] = min(self.best_loss[1], train_avg_eval_mode_loss)
                if train_avg_loss > 0.001:
                    # If the LR changed (i.e. training stuck) and also loss is large
                    if self.train_stuck_by_lr(self.optm, self.flags.lr/80):
                        # Switch to the gradient ascend mode
                        print("ascent epoch " + str(epoch))
                        gd = False
                else:
                    print(train_avg_loss)
                    x = list(range(len(train_loss)))
                    plt.clf()
                    plt.figure(1)
                    self.best_loss[1] = min(test_err)
                    plt.title('Best Eval Error: {}'.format(self.best_loss[1]))
                    plt.xlabel('Epoch')
                    plt.ylabel('MSE')
                    plt.plot(x, train_loss, label='Training Data')
                    plt.plot(x, train_loss_eval_mode_list, label='Eval')
                    plt.legend()
                    plt.savefig('hypersweep5/{}'.format(self.flags.model_name + '.png'))
                    return
            else:
                gd = True
                print("Mean train loss for ascent epoch {}: {}".format(epoch, train_avg_loss))
                print("Mean eval for ascent epoch {}: {}".format(epoch, train_avg_eval_mode_loss))
                ascent_losses[epoch] = train_avg_loss
                plt.scatter(epoch, train_avg_loss, color='red', marker='o', s=12)
                self.reset_lr(self.optm)

            # # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
            # # self.lr_scheduler.step()

            if self.flags.use_warm_restart:
                if epoch % self.flags.lr_warm_restart == 0:
                    for param_group in self.optm.param_groups:
                        param_group['lr'] = self.flags.lr / 10
                        print('Resetting learning rate to %.5f' % self.flags.lr)
            epoch += 1
        x = list(range(epoch))
        plt.clf()
        plt.figure(1)
        plt.title(
            'Best Train Error: {}, Best Eval Error: {}'.format(self.best_loss[1], self.best_loss[0]))
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.plot(x, train_err, label='Training Data')
        plt.plot(x, test_err, label='Eval')
        plt.legend()
        plt.savefig('hs4_1/{}.png'.format(self.flags.model_name))

        print("\nAscent losses:\n")
        print(ascent_losses)
        return self.best_loss
    def train_network_ascent(self):
        if torch.cuda.is_available():
            self.model.cuda()
        self.init_opt(self.flags.optim)
        self.init_weights()
        self.lr = lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
                                                 factor=self.flags.lr_decay_rate,
                                                 patience=10, verbose=True, threshold=1e-4)
        train_err, test_err = [], []
        cuda = torch.cuda.is_available()
        epoch = 0
        gd = True
        ascent_losses = {}
        while epoch < self.flags.train_step:
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
                self.optm.zero_grad()
                out, w0, wp, g = self.model(geometry)
                if ind == 1 and epoch % 20 == 1:
                    self.compare_spectra(spectra.cpu().numpy()[0, :] ,out.detach().cpu().numpy()[0,:], 'epoch_{}'.format(epoch),
                                         w0=w0.detach().cpu().numpy()[0,:], wp=wp.detach().cpu().numpy()[0,:],
                                         g=g.detach().cpu().numpy()[0,:])

                loss = F.mse_loss(out, spectra)
                if not gd:
                    loss *= -1 * self.flags.strength
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optm.step()
                train_loss.append(np.copy(loss.cpu().data.numpy()))

                with torch.no_grad():
                    self.model.eval()
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
            #if not gd:
            #    mean_train_loss *= -1
            mean_eval_loss = np.mean(eval_loss)
            train_err.append(mean_train_loss)
            test_err.append(mean_eval_loss)

            if gd:
                self.lr_scheduler.step(mean_train_loss)
                if mean_train_loss > 0.001:
                    # If the LR changed (i.e. training stuck) and also loss is large
                    if self.train_stuck_by_lr(self.optm, self.flags.lr/8):
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
                print("Mean train loss for ascent epoch {}: {}".format(epoch, mean_train_loss))
                print("Mean eval for ascent epoch {}: {}".format(epoch, mean_eval_loss))
                ascent_losses[epoch] = mean_train_loss
                plt.scatter(epoch, mean_train_loss, color='red', marker='o', s=12)
                self.reset_lr(self.optm)
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
        plt.savefig('hypersweep5/{}.png'.format(self.flags.model_name))

        print("\nAscent losses:\n")
        print(ascent_losses)
        return self.best_loss
