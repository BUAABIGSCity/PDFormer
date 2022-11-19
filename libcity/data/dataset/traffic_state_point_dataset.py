import os

from libcity.data.dataset import TrafficStateDataset


class TrafficStatePointDataset(TrafficStateDataset):

    def __init__(self, config):
        super().__init__(config)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'point_based_{}.npz'.format(self.parameters_str))

    def _load_geo(self):
        super()._load_geo()

    def _load_rel(self):
        super()._load_rel()

    def _load_dyna(self, filename):
        return super()._load_dyna_3d(filename)

    def _add_external_information(self, df, ext_data=None):
        return super()._add_external_information_3d(df, ext_data)

    def get_data_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}
