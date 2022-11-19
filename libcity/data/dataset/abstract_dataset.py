class AbstractDataset(object):

    def __init__(self, config):
        raise NotImplementedError("Dataset not implemented")

    def get_data(self):
        raise NotImplementedError("get_data not implemented")

    def get_data_feature(self):
        raise NotImplementedError("get_data_feature not implemented")