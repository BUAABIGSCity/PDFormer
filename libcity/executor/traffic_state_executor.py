import os
import time
import numpy as np
import torch

from ray import tune
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from libcity.executor.abstract_executor import AbstractExecutor
from libcity.utils import get_evaluator, ensure_dir, reduce_array
from libcity.model import loss
from functools import partial


class TrafficStateExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.device = self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self.exp_id = self.config.get('exp_id', None)

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './libcity/cache/{}'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._scaler = self.model.get_data_feature().get('scaler')
        self.rank = self.config.get('rank', 0)
        self.distributed = self.config.get('distributed', False)
        if self.distributed:
            self.world_size = self.config.get('world_size', 1)
            self._logger.info("Using native Torch DistributedDataParallel.")
            local_rank = self.config.get('local_rank', 0)
            self.model = NativeDDP(self.model, device_ids=[local_rank])
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.epochs = self.config.get('max_epoch', 100)
        self.train_loss = self.config.get('train_loss', 'none')
        self.train_loss = 'none'
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.hyper_tune = self.config.get('hyper_tune', False)

        self.output_dim = self.config.get('output_dim', 1)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)
        self.loss_func = self._build_train_loss()

        self.initial_ckpt = self.config.get("initial_ckpt", None)
        if self.initial_ckpt:
            self.load_model_with_initial_ckpt(self.initial_ckpt)
        self.grad_accmu_steps = config.get('grad_accmu_steps', 1)
        self.optimizer.zero_grad()

    def load_model_with_initial_ckpt(self, initial_ckpt):
        assert os.path.exists(initial_ckpt), 'Weights at %s not found' % initial_ckpt
        model_state, optimizer_state = torch.load(initial_ckpt, map_location=torch.device('cpu'))
        model_keys = self.model.state_dict()
        state_dict_load = {}
        unexpect_keys = []
        for k, v in model_state.items():
            if k not in model_keys.keys() or v.shape != model_keys[k].shape:
                unexpect_keys.append(k)
            else:
                state_dict_load[k] = v
        for k, v in model_keys.items():
            if k not in model_state.keys():
                unexpect_keys.append(k)
        self._logger.info("unexpected keys: {}".format(unexpect_keys))
        self.model.load_state_dict(state_dict_load, strict=False)
        self._logger.info("Initialize model from {}".format(initial_ckpt))

    def save_model(self, cache_name):
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model_with_epoch(self, epoch):
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def _build_train_loss(self):
        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the loss func defined in the model.')
            return None
        if self.train_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info('You select `{}` as train loss function.'.format(self.train_loss.lower()))

        def func(batch):
            y_true = batch['y']
            y_predicted = self.model.predict(batch)
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
            if self.train_loss.lower() == 'mae':
                lf = loss.masked_mae_torch
            elif self.train_loss.lower() == 'mse':
                lf = loss.masked_mse_torch
            elif self.train_loss.lower() == 'rmse':
                lf = loss.masked_rmse_torch
            elif self.train_loss.lower() == 'mape':
                lf = loss.masked_mape_torch
            elif self.train_loss.lower() == 'logcosh':
                lf = loss.log_cosh_loss
            elif self.train_loss.lower() == 'huber':
                lf = loss.huber_loss
            elif self.train_loss.lower() == 'quantile':
                lf = loss.quantile_loss
            elif self.train_loss.lower() == 'masked_mae':
                lf = partial(loss.masked_mae_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mse':
                lf = partial(loss.masked_mse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_rmse':
                lf = partial(loss.masked_rmse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mape':
                lf = partial(loss.masked_mape_torch, null_val=0)
            elif self.train_loss.lower() == 'r2':
                lf = loss.r2_score_torch
            elif self.train_loss.lower() == 'evar':
                lf = loss.explained_variance_score_torch
            else:
                lf = loss.masked_mae_torch
            return lf(y_predicted, y_true)
        return func

    def evaluate(self, test_dataloader):
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model(batch) if self.distributed else self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result

    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses, batches_seen = self._train_epoch(train_dataloader, epoch_idx, batches_seen, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss = np.mean(losses)
            if self.distributed:
                train_loss = reduce_array(train_loss, self.world_size, self.device)
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, batches_seen, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            epoch_time = end_time - start_time
            if self.distributed:
                epoch_time = reduce_array(np.array(epoch_time), self.world_size, self.device)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, batches_seen, np.mean(losses), val_loss,
                           log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune:
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            average_train_time = sum(train_time) / len(train_time)
            average_eval_time = sum(eval_time) / len(eval_time)
            if self.distributed:
                average_train_time = reduce_array(average_train_time, self.world_size, self.device)
                average_eval_time = reduce_array(average_eval_time, self.world_size, self.device)
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), average_train_time, average_eval_time))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            batches_seen += 1
            loss = loss / self.grad_accmu_steps
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return losses, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            if self.distributed:
                mean_loss = reduce_array(mean_loss, self.world_size, self.device)
            self._writer.add_scalar('eval loss', mean_loss, batches_seen)
            return mean_loss
