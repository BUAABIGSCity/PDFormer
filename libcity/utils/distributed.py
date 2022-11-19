import torch
from torch import distributed as dist


def reduce_array(tensor, n, device):
    rt = torch.tensor(tensor).to(device)
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt
