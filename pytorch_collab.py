#!/usr/bin/env python

import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.multiprocessing import Process
from torch.optim.lr_scheduler import CosineAnnealingLR

from exp_dataset import load_cifar10_noniid
from pytorch_model import ResNet18
from util import Accuracy, Average, EMAverage
from util import flatten_torch_tensor, unflatten_torch_tensor
from threading import Thread
from _thread import start_new_thread

alpha = 0.5
seed = 102
world_size = 4
model = 'resnet'

noniid = True
num_epochs = 100
lr = 0.001 * world_size
log_str = "ow"
log_str += "_ub"
log_str = log_str

log_dir = 'trial/cifar10collab_sgd/{}_alpha{}_s{}_{}_lr{}_seed{}'.format(log_str, alpha, world_size, model, lr, seed) + time.strftime("%m-%d-%H_%M")


class Trainer(object):

    def __init__(self, net, optimizer, train_loader, presam_loader, test_loader, device):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.presam_loader = presam_loader
        self.test_loader = test_loader
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.com_tensor = torch.ones(1)
        self.epoch = 0
        self.step = 1
        self.writer = None
        ## importance sampling related
        self.next_batch_iter = None
        self.computed_samples = {'index': [], 'prob': []}
        self.should_compute_importance = True

    def fit(self, epochs):

        if self.rank == 0:
            self.writer = SummaryWriter(log_dir)
            print(log_str)

        scheduler = CosineAnnealingLR(self.optimizer, epochs)

        self.average_model()

        for epoch in range(1, epochs + 1):

            self.epoch = epoch
            _, _, _ = self.train()
            scheduler.step()
            if self.step * world_size > 10000000:
                break

    def get_next(self):
        if self.next_batch_iter is None:
            self.next_batch_iter = iter(self.presam_loader)
        try:
            b = self.next_batch_iter.next()
        except StopIteration:
            self.next_batch_iter = iter(self.presam_loader)
            b = self.next_batch_iter.next()
        return b

    def average_model(self):
        for p in self.net.parameters():
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= float(self.world_size)

    def update_samples(self, ema_loss, alpha=0.5):
        presam_losses_list = []
        labels = []
        datas = []
        index = []
        cnt = 0
        while self.should_compute_importance and cnt < 10:
            cnt += 1
            with torch.no_grad():
                idx, data, label = self.get_next()
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.net(data)
                presam_losses = F.cross_entropy(output, label, reduction='none')
                presam_losses_list.append(presam_losses)
                labels.append(label)
                datas.append(data)
                index.append(idx)

            presam_losses = torch.cat(presam_losses_list)
            presam_avg_loss = torch.mean(presam_losses)
            ema_loss.update(presam_avg_loss)
            presam_losses = presam_losses + alpha * ema_loss.value
            losses_pros = presam_losses / torch.sum(presam_losses)

            important_idx = torch.multinomial(losses_pros, self.train_loader.batch_size, replacement=True)

        return (losses_pros[important_idx] * presam_losses.size()[0]), torch.cat(datas)[important_idx], \
               torch.cat(labels)[important_idx], torch.cat(index)[important_idx], presam_avg_loss

    def train(self):
        running_train_loss = Average()
        presam_ema_loss = EMAverage()
        runing_train_acc = Accuracy()


        probs, i_data, i_label, _ , presam_avg_loss = self.update_samples(presam_ema_loss)

        for idx, data, label in self.train_loader:

            t0 = time.time()

            t4 = time.time()
            output = self.net(i_data)
            losses = F.cross_entropy(output, i_label, reduction="none")
            ff_time = time.time() - t4
            runing_train_acc.update(output, i_label)

            losses = torch.div(losses, probs)

            t1 = time.time()

            #probs, i_data, i_label, _, presam_avg_loss = self.update_samples(presam_ema_loss)
            is_time = time.time() - t1

            t5 = time.time()
            loss = torch.mean(losses)

            self.optimizer.zero_grad()
            loss.backward()
            bp_time = time.time() - t5

            # average gradients
            t3 = time.time()

            # self.should_compute_importance = True
            # start_new_thread(self.average_gradients, ())
            # probs, i_data, i_label, _, presam_avg_loss = self.update_samples(presam_ema_loss)

            self.should_compute_importance = True
            probs, i_data, i_label, _, presam_avg_loss = self.update_samples(presam_ema_loss)
            self.average_gradients()

            sync_time = time.time() - t3

            self.optimizer.step()

            running_train_loss.update(loss.item(), data.size(0))

            step_time = time.time() - t0

            if self.step % 100 == 0 and self.rank == 0:
                print('step:{},'.format(self.step),
                      'running train loss: {}, running train acc: {}, presam_ema_loss: {}, step time:{:.3f},'
                      'is_time:{:.3f}, sync_time:{:.3f}, ff_time:{:.3f}, bp_time:{:.3f}'.format(running_train_loss,
                                                                                                runing_train_acc,
                                                                                                presam_ema_loss,
                                                                                                step_time, is_time,
                                                                                                sync_time, ff_time,
                                                                                                bp_time))

            # evaluate on one node
            if self.step % 200 == 0 and self.rank == 0:
                train_loss, train_acc, test_loss, test_acc = self.evaluate()

                a = torch.nonzero(self.com_tensor).size(0)
                sr = 1 - a / self.com_tensor.numel()

                self.writer.add_scalar('train/acc', train_acc.accuracy, self.step)
                self.writer.add_scalar('test/acc', test_acc.accuracy, self.step)
                self.writer.add_scalar('train/loss', train_loss.average, self.step)
                self.writer.add_scalar('test/loss', test_loss.average, self.step)

                print(
                    '(Eval) Step: {},'.format(self.step),
                    'train loss: {}, train acc: {}'.format(train_loss, train_acc),
                    'test loss: {}, test acc: {}, sparse rate: {:.2f}'.format(test_loss, test_acc, sr))

            self.step += 1

        return running_train_loss, runing_train_acc, presam_ema_loss

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()

        train_loss = Average()
        train_acc = Accuracy()

        self.net.eval()

        with torch.no_grad():
            for idx, data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)
                loss = F.cross_entropy(output, label)

                train_loss.update(loss.item(), data.size(0))
                train_acc.update(output, label)

        with torch.no_grad():
            for idx, data, label in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)
                loss = F.cross_entropy(output, label)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)

        self.net.train()

        return train_loss, train_acc, test_loss, test_acc

    def average_gradients(self):

        grads = [p.grad.data for _, p in self.net.named_parameters()]

        flattened_grads = flatten_torch_tensor(grads)

        dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM)

        flattened_grads = flattened_grads / self.world_size  ## compute average

        avg_quan_grads = unflatten_torch_tensor(flattened_grads, grads)

        for quan_grad, p in zip(avg_quan_grads, self.net.parameters()):
            p.grad.data = quan_grad


def my_run(presam_loader, train_loader, test_loader):
    device = 'cuda'

    model = ResNet18(num_classes=10)

    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters', pytorch_total_params)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, optimizer, train_loader, presam_loader, test_loader, 'cuda')

    trainer.fit(num_epochs)


def init_processes(rank, size, presam_loader, train_loader, test_loader, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank%num_gpus)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '295001'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(presam_loader, train_loader, test_loader)


if __name__ == "__main__":
    np.random.seed(seed)
    if noniid:
        presam_loaders, train_loader, test_loader = load_cifar10_noniid(split_num=world_size, alpha=alpha)

    processes = []
    for rank in range(world_size):
        p = Process(target=init_processes,
                    args=(rank, world_size, presam_loaders[rank], train_loader, test_loader, my_run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
