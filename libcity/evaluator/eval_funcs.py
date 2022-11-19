import numpy as np
import torch


def mse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MSE'
    return np.mean(sum(pow(loc_pred - loc_true, 2)))


def mae(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAE'
    return np.mean(sum(loc_pred - loc_true))


def rmse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'RMSE'
    return np.sqrt(np.mean(sum(pow(loc_pred - loc_true, 2))))


def mape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAPE'
    assert 0 not in loc_true, "MAPE:"
    return np.mean(abs(loc_pred - loc_true) / loc_true)


def mare(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "MARE"
    assert np.sum(loc_true) != 0, "MARE"
    return np.sum(np.abs(loc_pred - loc_true)) / np.sum(loc_true)


def smape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'SMAPE'
    assert 0 in (loc_pred + loc_true), "SMAPE"
    return 2.0 * np.mean(np.abs(loc_pred - loc_true) / (np.abs(loc_pred) +
                                                        np.abs(loc_true)))


def acc(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "accuracy"
    loc_diff = loc_pred - loc_true
    loc_diff[loc_diff != 0] = 1
    return loc_diff, np.mean(loc_diff == 0)


def top_k(loc_pred, loc_true, topk):
    assert topk > 0, "top-k ACC"
    loc_pred = torch.FloatTensor(loc_pred)
    val, index = torch.topk(loc_pred, topk, 1)
    index = index.numpy()
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):
        target = loc_true[i]
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg
