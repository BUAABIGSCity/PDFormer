import json
from heapq import nlargest
import pandas as pd
from libcity.model.loss import *


def output(method, value, field):
    if method == 'ACC':
        if field == 'model':
            print('----  {}  avg_acc={:.3f} ----'.format(method,
                                                                  value))
        else:
            print('{} avg_acc={:.3f}'.format(method, value))
    elif method in ['MSE', 'RMSE', 'MAE', 'MAPE', 'MARE', 'SMAPE']:
        if field == 'model':
            print('----  {}  avg_loss={:.3f} ----'.format(method,
                                                                   value))
        else:
            print('{} avg_loss={:.3f}'.format(method, value))
    else:
        if field == 'model':
            print('----  {}  avg_acc={:.3f} ----'.format(method,
                                                                  value))
        else:
            print('{} avg_acc={:.3f}'.format(method, value))


def transfer_data(data, model, maxk):
    if type(data) == str:
        data = json.loads(data)
    assert type(data) == dict, "invalid"
    if model == 'DeepMove':
        user_idx = data.keys()
        for user_id in user_idx:
            trace_idx = data[user_id].keys()
            for trace_id in trace_idx:
                trace = data[user_id][trace_id]
                loc_pred = trace['loc_pred']
                new_loc_pred = []
                for t_list in loc_pred:
                    new_loc_pred.append(sort_confidence_ids(t_list, maxk))
                data[user_id][trace_id]['loc_pred'] = new_loc_pred
    return data


def sort_confidence_ids(confidence_list, threshold):
    max_score_with_id = nlargest(
        threshold, enumerate(confidence_list), lambda x: x[1])
    return list(map(lambda x: x[0], max_score_with_id))


def evaluate_model(y_pred, y_true, metrics, mode='single', path='metrics.csv'):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true.shape is not equal to y_pred.shape")
    len_timeslots = y_true.shape[1]
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.FloatTensor(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.FloatTensor(y_true)
    assert isinstance(y_pred, torch.Tensor)
    assert isinstance(y_true, torch.Tensor)

    df = []
    for i in range(1, len_timeslots + 1):
        line = {}
        for metric in metrics:
            if mode.lower() == 'single':
                if metric == 'masked_MAE':
                    line[metric] = masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
                elif metric == 'masked_MSE':
                    line[metric] = masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
                elif metric == 'masked_RMSE':
                    line[metric] = masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
                elif metric == 'masked_MAPE':
                    line[metric] = masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
                elif metric == 'MAE':
                    line[metric] = masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'MSE':
                    line[metric] = masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'RMSE':
                    line[metric] = masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'MAPE':
                    line[metric] = masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'R2':
                    line[metric] = r2_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'EVAR':
                    line[metric] = explained_variance_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                else:
                    raise ValueError('Error parameter mode={}, please set `single` or `average`.'.format(mode))
            elif mode.lower() == 'average':
                if metric == 'masked_MAE':
                    line[metric] = masked_mae_torch(y_pred[:, :i], y_true[:, :i], 0).item()
                elif metric == 'masked_MSE':
                    line[metric] = masked_mse_torch(y_pred[:, :i], y_true[:, :i], 0).item()
                elif metric == 'masked_RMSE':
                    line[metric] = masked_rmse_torch(y_pred[:, :i], y_true[:, :i], 0).item()
                elif metric == 'masked_MAPE':
                    line[metric] = masked_mape_torch(y_pred[:, :i], y_true[:, :i], 0).item()
                elif metric == 'MAE':
                    line[metric] = masked_mae_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'MSE':
                    line[metric] = masked_mse_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'RMSE':
                    line[metric] = masked_rmse_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'MAPE':
                    line[metric] = masked_mape_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'R2':
                    line[metric] = r2_score_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'EVAR':
                    line[metric] = explained_variance_score_torch(y_pred[:, :i], y_true[:, :i]).item()
                else:
                    raise ValueError('Error parameter metric={}!'.format(metric))
            else:
                raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(mode))
        df.append(line)
    df = pd.DataFrame(df, columns=metrics)
    print(df)
    df.to_csv(path)
    return df
