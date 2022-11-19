import os

from libcity.data.dataset import TrafficStateDataset


class TrafficStateGridDataset(TrafficStateDataset):

    def __init__(self, config):
        super().__init__(config)
        self.use_row_column = self.config.get('use_row_column', True)
        self.parameters_str = self.parameters_str + '_' + str(self.use_row_column)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'grid_based_{}.npz'.format(self.parameters_str))
        self._load_rel()

    def _load_geo(self):
        super()._load_grid_geo()

    def _load_rel(self):
        if os.path.exists(self.data_path + self.rel_file + '.rel'):
            super()._load_rel()
        else:
            super()._load_grid_rel()

    def _load_dyna(self, filename):
        if self.use_row_column:
            return super()._load_grid_4d(filename)
        else:
            return super()._load_grid_3d(filename)

    def _add_external_information(self, df, ext_data=None):
        if self.use_row_column:
            return super()._add_external_information_4d(df, ext_data)
        else:
            return super()._add_external_information_3d(df, ext_data)

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "ext_dim": self.ext_dim,
                "output_dim": self.output_dim, "len_row": self.len_row, "len_column": self.len_column,
                "num_batches": self.num_batches}
