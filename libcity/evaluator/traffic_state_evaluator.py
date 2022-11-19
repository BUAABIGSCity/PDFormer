import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
from logging import getLogger
from libcity.evaluator.abstract_evaluator import AbstractEvaluator


class TrafficStateEvaluator(AbstractEvaluator):

    def __init__(self, config):
        self.metrics = config.get('metrics', ['MAE'])
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE',
                                'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR']
        self.save_modes = config.get('save_modes', ['csv', 'json'])
        self.mode = config.get('mode', 'single')
        self.config = config
        self.len_timeslots = 0
        self.result = {}
        self.intermediate_result = {}
        self._check_config()
        self._logger = getLogger()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficStateEvaluator'.format(str(metric)))

    def collect(self, batch):
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']
        y_pred = batch['y_pred']
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        self.len_timeslots = y_true.shape[1]
        for i in range(1, self.len_timeslots+1):
            for metric in self.metrics:
                if metric+'@'+str(i) not in self.intermediate_result:
                    self.intermediate_result[metric+'@'+str(i)] = []
        if self.mode.lower() == 'average':
            for i in range(1, self.len_timeslots+1):
                for metric in self.metrics:
                    if metric == 'masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                    elif metric == 'masked_MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                    elif metric == 'masked_RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                    elif metric == 'masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, :i], y_true[:, :i], 0).item())
                    elif metric == 'MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.r2_score_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.explained_variance_score_torch(y_pred[:, :i], y_true[:, :i]).item())
        elif self.mode.lower() == 'single':
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, i-1], y_true[:, i-1], 0).item())
                    elif metric == 'masked_MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, i-1], y_true[:, i-1], 0).item())
                    elif metric == 'masked_RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, i-1], y_true[:, i-1], 0).item())
                    elif metric == 'masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, i-1], y_true[:, i-1], 0).item())
                    elif metric == 'MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, i-1], y_true[:, i-1]).item())
                    elif metric == 'MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, i-1], y_true[:, i-1]).item())
                    elif metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, i-1], y_true[:, i-1]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, i-1], y_true[:, i-1]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.r2_score_torch(y_pred[:, i-1], y_true[:, i-1]).item())
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.explained_variance_score_torch(y_pred[:, i-1], y_true[:, i-1]).item())
        else:
            raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))

    def evaluate(self):
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                self.result[metric+'@'+str(i)] = sum(self.intermediate_result[metric+'@'+str(i)]) / \
                                                 len(self.intermediate_result[metric+'@'+str(i)])
        return self.result

    def save_result(self, save_path, filename=None):
        self._logger.info('Note that you select the {} mode to evaluate!'.format(self.mode))
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset'] + '_' + self.mode

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.json'.format(filename)))

        dataframe = {}
        if 'csv' in self.save_modes:
            for metric in self.metrics:
                dataframe[metric] = []
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    dataframe[metric].append(self.result[metric+'@'+str(i)])
            dataframe = pd.DataFrame(dataframe, index=range(1, self.len_timeslots + 1))
            dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.csv'.format(filename)))
            self._logger.info("\n" + str(dataframe))
        return dataframe

    def clear(self):
        self.result = {}
        self.intermediate_result = {}
