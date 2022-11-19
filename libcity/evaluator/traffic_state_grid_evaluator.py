import os
import json
import datetime
import pandas as pd
from libcity.utils import ensure_dir
from libcity.model import loss
from libcity.evaluator.traffic_state_evaluator import TrafficStateEvaluator


class TrafficStateGridEvaluator(TrafficStateEvaluator):

    def __init__(self, config):
        super().__init__(config)
        self.output_dim = self.config.get('output_dim', 1)
        self.mask_val = self.config.get('mask_val', 10)

    def collect(self, batch):
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']
        y_pred = batch['y_pred']
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        self.len_timeslots = y_true.shape[1]
        for j in range(self.output_dim):
            for i in range(1, self.len_timeslots+1):
                for metric in self.metrics:
                    if str(j)+'-'+metric+'@'+str(i) not in self.intermediate_result:
                        self.intermediate_result[str(j)+'-'+metric+'@'+str(i)] = []
        if self.mode.lower() == 'average':
            for j in range(self.output_dim):
                for i in range(1, self.len_timeslots+1):
                    for metric in self.metrics:
                        if metric == 'masked_MAE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mae_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_MSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mse_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_RMSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_rmse_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_MAPE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mape_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j], 0, self.mask_val).item())
                        elif metric == 'MAE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mae_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'MSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mse_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'RMSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_rmse_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'MAPE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mape_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'R2':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.r2_score_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
                        elif metric == 'EVAR':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.explained_variance_score_torch(y_pred[:, :i, ..., j], y_true[:, :i, ..., j]).item())
        elif self.mode.lower() == 'single':
            for j in range(self.output_dim):
                for i in range(1, self.len_timeslots+1):
                    for metric in self.metrics:
                        if metric == 'masked_MAE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mae_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_MSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mse_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_RMSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_rmse_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j], 0, self.mask_val).item())
                        elif metric == 'masked_MAPE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mape_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j], 0, self.mask_val).item())
                        elif metric == 'MAE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mae_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'MSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mse_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'RMSE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_rmse_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'MAPE':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.masked_mape_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'R2':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.r2_score_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
                        elif metric == 'EVAR':
                            self.intermediate_result[str(j) + '-' + metric + '@' + str(i)].append(
                                loss.explained_variance_score_torch(y_pred[:, i-1, ..., j], y_true[:, i-1, ..., j]).item())
        else:
            raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))

    def evaluate(self):
        for j in range(self.output_dim):
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    self.result[str(j)+'-'+metric+'@'+str(i)] = sum(self.intermediate_result[str(j)+'-'+metric+'@'+str(i)]) / \
                                                                len(self.intermediate_result[str(j)+'-'+metric+'@'+str(i)])
        return self.result

    def save_result(self, save_path, filename=None):
        self._logger.info('Note that you select the {} mode to evaluate!'.format(self.mode))
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.json'.format(filename)))

        dataframe = {}
        if 'csv' in self.save_modes:
            for j in range(self.output_dim):
                for metric in self.metrics:
                    dataframe[str(j)+"-"+metric] = []
                for i in range(1, self.len_timeslots + 1):
                    for metric in self.metrics:
                        dataframe[str(j)+"-"+metric].append(self.result[str(j)+'-'+metric+'@'+str(i)])
            dataframe = pd.DataFrame(dataframe, index=range(1, self.len_timeslots + 1))
            dataframe.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), index=False)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.csv'.format(filename)))
            self._logger.info("\n" + str(dataframe))
        return dataframe
