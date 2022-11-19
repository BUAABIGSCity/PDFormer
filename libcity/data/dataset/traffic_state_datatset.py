import os
import pandas as pd
import numpy as np
import datetime
from logging import getLogger

from libcity.data.dataset import AbstractDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler, ensure_dir


class TrafficStateDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.num_workers = self.config.get('num_workers', 0)
        self.pad_with_last_sample = self.config.get('pad_with_last_sample', True)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.part_train_rate = self.config.get("part_train_rate", 1)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        self.ext_scaler_type = self.config.get('ext_scaler', 'none')
        self.load_external = self.config.get('load_external', False)
        self.normal_external = self.config.get('normal_external', False)
        self.add_time_in_day = self.config.get('add_time_in_day', False)
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.bidir = self.config.get('bidir', False)
        self.data_col = self.config.get('data_col', '')
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.part_train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample) + '_' + str("".join(self.data_col))
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'traffic_state_{}.npz'.format(self.parameters_str))
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        self.weight_col = self.config.get('weight_col', '')
        self.ext_col = self.config.get('ext_col', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.data_files = self.config.get('data_files', self.dataset)
        self.ext_file = self.config.get('ext_file', self.dataset)
        self.output_dim = self.config.get('output_dim', 1)
        self.time_intervals = self.config.get('time_intervals', 300)
        self.init_weight_inf_or_zero = self.config.get('init_weight_inf_or_zero', 'inf')
        self.set_weight_link_or_dist = self.config.get('set_weight_link_or_dist', 'dist')
        self.calculate_weight_adj = self.config.get('calculate_weight_adj', False)
        self.weight_adj_epsilon = self.config.get('weight_adj_epsilon', 0.1)
        self.data = None
        self.feature_name = {'X': 'float', 'y': 'float'}
        self.adj_mx = None
        self.scaler = None
        self.ext_scaler = None
        self.feature_dim = 0
        self.ext_dim = 0
        self.num_nodes = 0
        self.num_batches = 0
        self._logger = getLogger()
        self.rank = self.config.get('rank', 0)
        self.distributed = self.config.get('distributed', False)
        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(self.data_path + self.rel_file + '.rel'):
            self._load_rel()
        else:
            self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)

    def _load_geo(self):
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))

    def _load_grid_geo(self):
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.geo_to_rc = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
        for i in range(geofile.shape[0]):
            self.geo_to_rc[geofile['geo_id'][i]] = [geofile['row_id'][i], geofile['column_id'][i]]
        self.len_row = max(list(geofile['row_id'])) + 1
        self.len_column = max(list(geofile['column_id'])) + 1
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_grids=' + str(len(self.geo_ids))
                          + ', grid_size=' + str((self.len_row, self.len_column)))

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self._logger.info('set_weight_link_or_dist: {}'.format(self.set_weight_link_or_dist))
        self._logger.info('init_weight_inf_or_zero: {}'.format(self.init_weight_inf_or_zero))
        if self.weight_col != '':
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError('`weight_col` parameter must be only one column!')
                self.weight_col = self.weight_col[0]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) != 5:
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            else:
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                    'origin_id', 'destination_id', self.weight_col]]
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape))
        if self.calculate_weight_adj:
            self._calculate_adjacency_matrix()

    def _load_grid_rel(self):
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        dirs = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for i in range(self.len_row):
            for j in range(self.len_column):
                index = i * self.len_column + j
                for d in dirs:
                    nei_i = i + d[0]
                    nei_j = j + d[1]
                    if nei_i >= 0 and nei_i < self.len_row and nei_j >= 0 and nei_j < self.len_column:
                        nei_index = nei_i * self.len_column + nei_j
                        self.adj_mx[index][nei_index] = 1
                        self.adj_mx[nei_index][index] = 1
        self._logger.info("Generate grid rel file, shape=" + str(self.adj_mx.shape))

    def _calculate_adjacency_matrix(self):
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0

    def _load_dyna(self, filename):
        raise NotImplementedError('Please implement the function `_load_dyna()`.')

    def _load_dyna_3d(self, filename):
        self._logger.info("Loading file " + filename + '.dyna')
        dynafile = pd.read_csv(self.data_path + filename + '.dyna')
        if self.data_col != '':
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'entity_id')
            dynafile = dynafile[data_col]
        else:
            dynafile = dynafile[dynafile.columns[2:]]
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not dynafile['time'].isna().any():
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i+len_time].values)
        data = np.array(data, dtype=np.float)
        data = data.swapaxes(0, 1)
        self._logger.info("Loaded file " + filename + '.dyna' + ', shape=' + str(data.shape))
        return data

    def _load_grid_3d(self, filename):
        self._logger.info("Loading file " + filename + '.grid')
        gridfile = pd.read_csv(self.data_path + filename + '.grid')
        if self.data_col != '':
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'row_id')
            data_col.insert(2, 'column_id')
            gridfile = gridfile[data_col]
        else:
            gridfile = gridfile[gridfile.columns[2:]]
        self.timesolts = list(gridfile['time'][:int(gridfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridfile['time'].isna().any():
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        feature_dim = len(gridfile.columns) - 3
        df = gridfile[gridfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=np.float)
        data = data.swapaxes(0, 1)
        self._logger.info("Loaded file " + filename + '.grid' + ', shape=' + str(data.shape))
        return data

    def _load_grid_4d(self, filename):
        self._logger.info("Loading file " + filename + '.grid')
        gridfile = pd.read_csv(self.data_path + filename + '.grid')
        if self.data_col != '':
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'row_id')
            data_col.insert(2, 'column_id')
            gridfile = gridfile[data_col]
        else:
            gridfile = gridfile[gridfile.columns[2:]]
        self.timesolts = list(gridfile['time'][:int(gridfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridfile['time'].isna().any():
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        feature_dim = len(gridfile.columns) - 3
        df = gridfile[gridfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(self.len_row):
            tmp = []
            for j in range(self.len_column):
                index = (i * self.len_column + j) * len_time
                tmp.append(df[index:index + len_time].values)
            data.append(tmp)
        data = np.array(data, dtype=np.float)
        data = data.swapaxes(2, 0).swapaxes(1, 2)
        self._logger.info("Loaded file " + filename + '.grid' + ', shape=' + str(data.shape))
        return data

    def _load_od_4d(self, filename):
        self._logger.info("Loading file " + filename + '.od')
        odfile = pd.read_csv(self.data_path + filename + '.od')
        if self.data_col != '':
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_id')
            data_col.insert(2, 'destination_id')
            odfile = odfile[data_col]
        else:
            odfile = odfile[odfile.columns[2:]]
        self.timesolts = list(odfile['time'][:int(odfile.shape[0] / self.num_nodes / self.num_nodes)])
        self.idx_of_timesolts = dict()
        if not odfile['time'].isna().any():
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx

        feature_dim = len(odfile.columns) - 3
        df = odfile[odfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((self.num_nodes, self.num_nodes, len_time, feature_dim))
        for i in range(self.num_nodes):
            origin_index = i * len_time * self.num_nodes
            for j in range(self.num_nodes):
                destination_index = j * len_time
                index = origin_index + destination_index
                data[i][j] = df[index:index + len_time].values
        data = data.transpose((2, 0, 1, 3))
        self._logger.info("Loaded file " + filename + '.od' + ', shape=' + str(data.shape))
        return data

    def _load_grid_od_4d(self, filename):
        self._logger.info("Loading file " + filename + '.gridod')
        gridodfile = pd.read_csv(self.data_path + filename + '.gridod')
        if self.data_col != '':
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_row_id')
            data_col.insert(2, 'origin_column_id')
            data_col.insert(3, 'destination_row_id')
            data_col.insert(4, 'destination_column_id')
            gridodfile = gridodfile[data_col]
        else:
            gridodfile = gridodfile[gridodfile.columns[2:]]
        self.timesolts = list(gridodfile['time'][:int(gridodfile.shape[0] / len(self.geo_ids) / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridodfile['time'].isna().any():
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        feature_dim = len(gridodfile.columns) - 5
        df = gridodfile[gridodfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((len(self.geo_ids), len(self.geo_ids), len_time, feature_dim))
        for oi in range(self.len_row):
            for oj in range(self.len_column):
                origin_index = (oi * self.len_column + oj) * len_time * len(self.geo_ids)
                for di in range(self.len_row):
                    for dj in range(self.len_column):
                        destination_index = (di * self.len_column + dj) * len_time
                        index = origin_index + destination_index
                        data[oi * self.len_column + oj][di * self.len_column + dj] = df[index:index + len_time].values
        data = data.transpose((2, 0, 1, 3))
        self._logger.info("Loaded file " + filename + '.gridod' + ', shape=' + str(data.shape))
        return data

    def _load_grid_od_6d(self, filename):
        self._logger.info("Loading file " + filename + '.gridod')
        gridodfile = pd.read_csv(self.data_path + filename + '.gridod')
        if self.data_col != '':
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'origin_row_id')
            data_col.insert(2, 'origin_column_id')
            data_col.insert(3, 'destination_row_id')
            data_col.insert(4, 'destination_column_id')
            gridodfile = gridodfile[data_col]
        else:
            gridodfile = gridodfile[gridodfile.columns[2:]]
        self.timesolts = list(gridodfile['time'][:int(gridodfile.shape[0] / len(self.geo_ids) / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridodfile['time'].isna().any():
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        feature_dim = len(gridodfile.columns) - 5
        df = gridodfile[gridodfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((self.len_row, self.len_column, self.len_row, self.len_column, len_time, feature_dim))
        for oi in range(self.len_row):
            for oj in range(self.len_column):
                origin_index = (oi * self.len_column + oj) * len_time * len(self.geo_ids)
                for di in range(self.len_row):
                    for dj in range(self.len_column):
                        destination_index = (di * self.len_column + dj) * len_time
                        index = origin_index + destination_index
                        data[oi][oj][di][dj] = df[index:index + len_time].values
        data = data.transpose((4, 0, 1, 2, 3, 5))
        self._logger.info("Loaded file " + filename + '.gridod' + ', shape=' + str(data.shape))
        return data

    def _load_ext(self):
        extfile = pd.read_csv(self.data_path + self.ext_file + '.ext')
        if self.ext_col != '':
            if isinstance(self.ext_col, list):
                ext_col = self.ext_col.copy()
            else:
                ext_col = [self.ext_col].copy()
            ext_col.insert(0, 'time')
            extfile = extfile[ext_col]
        else:
            extfile = extfile[extfile.columns[1:]]
        self.ext_timesolts = extfile['time']
        self.idx_of_ext_timesolts = dict()
        if not extfile['time'].isna().any():
            self.ext_timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.ext_timesolts))
            self.ext_timesolts = np.array(self.ext_timesolts, dtype='datetime64[ns]')
            for idx, _ts in enumerate(self.ext_timesolts):
                self.idx_of_ext_timesolts[_ts] = idx
        feature_dim = len(extfile.columns) - 1
        df = extfile[extfile.columns[-feature_dim:]].values
        self._logger.info("Loaded file " + self.ext_file + '.ext' + ', shape=' + str(df.shape))
        return df

    def _add_external_information(self, df, ext_data=None):
        raise NotImplementedError('Please implement the function `_add_external_information()`.')

    def _add_external_information_3d(self, df, ext_data=None):
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, dayofweek] = 1
            data_list.append(day_in_week)
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
            else:
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def _add_external_information_4d(self, df, ext_data=None):
        num_samples, len_row, len_column, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, len_row, len_column, 7))
            day_in_week[np.arange(num_samples), :, :, dayofweek] = 1
            data_list.append(day_in_week)
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
                    data_list.append(data_ind)
            else:
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, len_row, len_column, 1]).transpose((3, 1, 2, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def _add_external_information_6d(self, df, ext_data=None):
        num_samples, len_row, len_column, _, _, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, len_row, len_column, len_row, len_column, 1]).\
                transpose((5, 1, 2, 3, 4, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.zeros(shape=(num_samples, len_row, len_column, len_row, len_column, 7))
            day_in_week[np.arange(num_samples), :, :, :, :, dayofweek] = 1
            data_list.append(day_in_week)
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, len_row, len_column, len_row, len_column, 1]). \
                        transpose((5, 1, 2, 3, 4, 0))
                    data_list.append(data_ind)
            else:
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, len_row, len_column, len_row, len_column, 1]). \
                            transpose((5, 1, 2, 3, 4, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data

    def _generate_input_data(self, df):
        num_samples = df.shape[0]
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def _generate_data(self):
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):
            ext_data = self._load_ext()
        else:
            ext_data = None
        x_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna(filename)
            if self.load_external:
                df = self._add_external_information(df, ext_data)
            x, y = self._generate_input_data(df)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y

    def _split_train_val_test(self, x, y):
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        x_train, y_train = x[int(num_train*(1 - self.part_train_rate)):num_train], y[int(num_train*(1 - self.part_train_rate)):num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

        if self.rank == 0 and self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_train_val_test(self):
        x, y = self._generate_data()
        return self._split_train_val_test(x, y)

    def _load_cache_train_val_test(self):
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _get_scalar(self, scaler_type, x_train, y_train):
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            self._logger.info('NormalScaler max: ' + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            self._logger.info('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            self._logger.info('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info('LogScaler')
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        self.ext_scaler = self._get_scalar(self.ext_scaler_type,
                                           x_train[..., self.output_dim:], y_train[..., self.output_dim:])
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample,
                                distributed=self.distributed)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        raise NotImplementedError('Please implement the function `get_data_feature()`.')
