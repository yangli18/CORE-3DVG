# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
import os
import pickle

import torch
import torch.distributed as dist

from collections import defaultdict, deque

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    # def log_every(self, iterable, print_freq, header=None):
    #     i = 0
    #     if not header:
    #         header = ''
    #     start_time = time.time()
    #     end = time.time()
    #     iter_time = SmoothedValue(fmt='{avg:.4f}')
    #     data_time = SmoothedValue(fmt='{avg:.4f}')
    #     space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
    #     if torch.cuda.is_available():
    #         log_msg = self.delimiter.join([
    #             header,
    #             '[{0' + space_fmt + '}/{1}]',
    #             'eta: {eta}',
    #             '{meters}',
    #             'time: {time}',
    #             'data: {data}',
    #             'max mem: {memory:.0f}'
    #         ])
    #     else:
    #         log_msg = self.delimiter.join([
    #             header,
    #             '[{0' + space_fmt + '}/{1}]',
    #             'eta: {eta}',
    #             '{meters}',
    #             'time: {time}',
    #             'data: {data}'
    #         ])
    #     MB = 1024.0 * 1024.0
    #     for obj in iterable:
    #         data_time.update(time.time() - end)
    #         yield obj
    #         iter_time.update(time.time() - end)
    #         if i % print_freq == 0 or i == len(iterable) - 1:
    #             eta_seconds = iter_time.global_avg * (len(iterable) - i)
    #             eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    #             if torch.cuda.is_available():
    #                 print(log_msg.format(
    #                     i, len(iterable), eta=eta_string,
    #                     meters=str(self),
    #                     time=str(iter_time), data=str(data_time),
    #                     memory=torch.cuda.max_memory_allocated() / MB))
    #             else:
    #                 print(log_msg.format(
    #                     i, len(iterable), eta=eta_string,
    #                     meters=str(self),
    #                     time=str(iter_time), data=str(data_time)))
    #         i += 1
    #         end = time.time()
    #     total_time = time.time() - start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     print('{} Total time: {} ({:.4f} s / it)'.format(
    #         header, total_time_str, total_time / len(iterable)))


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


from torch.utils.data._utils.collate import default_collate
def collate_fn_w_span_indices(batch):
    elem = batch[0]
    assert isinstance(elem, dict)

    default_keys = [key for key in elem if not key.endswith('indices')]
    new_keys = [key for key in elem if key.endswith('indices')]

    batch_dict = {key: default_collate([d[key] for d in batch]) for key in default_keys}

    # 'span_indices', 'expand_indices'
    batch_dict.update({key: torch.cat([torch.as_tensor(d[key]) for d in batch], dim=0)
                        for key in new_keys})

    key = 'span_indices' #new_keys[0]
    batch_dict.update({'span_batch_idx': 
            torch.cat([torch.as_tensor([batch_idx] * len(d[key]))
                        for batch_idx, d in enumerate(batch)], dim=0)
    }) #
    batch_dict.update({'span_batch_ord': 
            torch.cat([torch.arange(len(d[key]))
                        for d in batch], dim=0)
    }) #
    batch_dict.update({'span_num': 
            torch.cat([torch.as_tensor([len(d[key])])
                        for d in batch], dim=0)
    }) #

    key = 'noun_indices'
    if key in new_keys:
        batch_dict.update({
            'noun_batch_idx': torch.cat([torch.as_tensor([batch_idx] * len(d[key]))
                                for batch_idx, d in enumerate(batch)], dim=0),
            'noun_batch_ord': torch.cat([torch.arange(len(d[key]))
                                for d in batch], dim=0),
            'noun_num': torch.cat([torch.as_tensor([len(d[key])])
                                for d in batch], dim=0)
    }) #

    return batch_dict