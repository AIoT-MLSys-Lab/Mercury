import numpy as np
import os, pickle, sys
from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes
import random
import torch, torchvision
from PIL import Image
from torchvision.datasets.folder import ImageFolder
import torch.distributed as dist
import torch.distributions.bernoulli as ber
import torch.nn.functional as F
def _flatten(values):
    if isinstance(values, np.ndarray) or torch.is_tensor(values):
        yield values.flatten()
    else:
        for value in values:
            yield from _flatten(value)

def flatten(values):
    # flatten nested lists of np.ndarray to np.ndarray
    return np.concatenate(list(_flatten(values)))

def flatten_torch_tensor(values):
    # flatten nested lists of np.ndarray to np.ndarray
    return torch.cat(list(_flatten(values)), 0)

def _unflatten(flat_values, prototype, offset):
    if isinstance(prototype, np.ndarray):
        shape = prototype.shape
        new_offset = offset + np.product(shape)
        value = flat_values[offset:new_offset].reshape(shape)
        return value, new_offset
    else:
        result = []
        for value in prototype:
            value, offset = _unflatten(flat_values, value, offset)
            result.append(value)
        return result, offset

def unflatten(flat_values, prototype):
    # unflatten np.ndarray to nested lists with structure of prototype
    result, offset = _unflatten(flat_values, prototype, 0)
    assert(offset == len(flat_values))
    return result

def _unflatten_torch_tensor(flat_values, prototype, offset):
    if torch.is_tensor(prototype):
        shape = prototype.shape
        new_offset = offset + prototype.numel()
        value = flat_values[offset:new_offset].reshape(shape)
        return value, new_offset
    else:
        result = []
        for value in prototype:
            value, offset = _unflatten_torch_tensor(flat_values, value, offset)
            result.append(value)
        return result, offset

def unflatten_torch_tensor(flat_values, prototype):
    # unflatten np.ndarray to nested lists with structure of prototype
    result, offset = _unflatten_torch_tensor(flat_values, prototype, 0)
    assert(offset == len(flat_values))
    return result

def quantize_tensor(a):
    sign = torch.sign(a)
    abs_a = torch.abs(a)
    max_a = torch.max(abs_a)
    sampled = ber.Bernoulli(abs_a / max_a).sample()
    return torch.mul(sign*max_a, sampled)


class ToNumpy(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return np.array(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Groupwise_Sampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, dataset, replacement=True):
        self.dataset = dataset
        self.replacement = replacement
        self.group_indicator = np.zeros( (len(self.dataset), ) )
        self.importance = np.ones( (len(self.dataset), ) )
        self.cur_sample_index = 0
        self.group_index = 0
        self.last_update_iteration = -1

    def update_importance(self, iteration, update_batchsize, model, device='cuda'):

        if iteration > self.last_update_iteration:
            self.group_index += 1
            self.last_update_iteration = iteration

        start_index = self.cur_sample_index
        end_index = min(self.cur_sample_index + update_batchsize, len(self.dataset) )

        data, label = self.dataset.get_slice(start_index, end_index)

        ## compute sample importances
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        presam_losses = F.cross_entropy(output, label, reduction='none')


        self.importance[start_index:end_index] = presam_losses.detach().cpu().numpy()
        self.group_indicator[start_index:end_index] = self.group_index

        if end_index == len(self.dataset):
            self.cur_sample_index = 0
        else:
            self.cur_sample_index = end_index

    def __iter__(self):
        counter = 0

        while True:
            group_member_location = self.group_indicator==self.group_index
            group_importances = self.importance[group_member_location]
            group_importances = group_importances +  np.mean(group_importances)
            group_importances = group_importances / np.sum(group_importances)

            # this is just the group index, need to convert back to global index
            index_list = torch.multinomial(torch.Tensor(group_importances),
                                           1, self.replacement).tolist()
            group_member_index = group_member_location.nonzero()[0]
            for i in group_member_index[np.array(index_list)]:
                yield i
                counter += 1
                if counter >= len(self.dataset):
                    return

    def __len__(self):
        return self.num_samples

class SampleImageFolder(ImageFolder):


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (index, sample, target) where target is class_index of the target class.
                    index in the location of the sample in the whole dataset
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, sample, target

class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)

class EMAverage(object):

    def __init__(self, alpha=0.9):
        self.first_update = True
        self.value = 0
        self.alpha = alpha

    def update(self, value):

        if self.first_update:
            self.value = value
            self.first_update = False
        else:
            self.value = self.alpha*self.value + (1 - self.alpha) * value


    def __str__(self):
        return '{:.6f}'.format(self.value)


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

class My_CIFAR10(torchvision.datasets.CIFAR10):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def get_slice(self, start, end):
        imgs = []
        targets = []

        for i in range(start, end):
            index, img, target = self[i]
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs), torch.LongTensor(targets)






def allreduce(t):
    """ Implementation of a ring-reduce. """
    rank = dist.get_rank()
    size = dist.get_world_size()

    tensors = torch.chunk(t, size)
    assert len(tensors) == size


    recv_buff_1 = torch.zeros(tensors[0].size())
    recv_buff_2 = torch.zeros(tensors[-1].size())

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        send_slice_idx = ((-i % size) + rank ) % size
        rec_slice_idx = (rank - i - 1) % size

        recv_buff = recv_buff_2 if rec_slice_idx == len(tensors) -1 else recv_buff_1

        send_req = dist.isend(tensors[send_slice_idx], right)

        dist.recv(recv_buff, left)
        tensors[rec_slice_idx][:] += recv_buff

        send_req.wait()


    for i in range(size - 1):
        send_slice_idx = (1 + rank - i ) % size
        rec_slice_idx = ( rank- i) % size


        recv_buff = recv_buff_2 if rec_slice_idx == len(tensors) -1 else recv_buff_1

        send_req = dist.isend(tensors[send_slice_idx], right)

        dist.recv(recv_buff, left)
        tensors[rec_slice_idx][:] = recv_buff

        send_req.wait()


    return torch.cat(tensors,0)




if __name__ == '__main__':
    weights = [0 if i>10 else 1 for i in range(1,101)]
    weights_new = [1-i for i in weights]
    sampler = WeightedRandomSampler_2(weights, len(weights))

    a = iter(sampler)

    for idx, a1 in enumerate(a):
        if idx > 10:
            sampler.update_weights(weights_new)
        print(idx, a1)
