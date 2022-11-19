import torch
import numpy as np


class Batch(object):

    def __init__(self, feature_name, pad_item=None, pad_max_len=None):
        self.data = {}
        self.pad_len = {}
        self.origin_len = {}
        self.pad_max_len = pad_max_len if pad_max_len is not None else {}
        self.pad_item = pad_item if pad_item is not None else {}
        self.feature_name = feature_name
        for key in feature_name:
            self.data[key] = []
            if key in self.pad_item:
                self.pad_len[key] = 0
                self.origin_len[key] = []

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def append(self, item):
        if len(item) != len(self.feature_name):
            raise KeyError(
                'when append a batch, item is not equal length with \
                    feature_name')
        for i, key in enumerate(self.feature_name):
            self.data[key].append(item[i])
            if key in self.pad_item:
                self.origin_len[key].append(len(item[i]))
                if self.pad_len[key] < len(item[i]):
                    self.pad_len[key] = len(item[i])

    def padding(self):
        for key in self.pad_item:
            if key not in self.data:
                raise KeyError('when pad a batch, raise this error!')
            max_len = self.pad_len[key]
            if key in self.pad_max_len:
                max_len = min(self.pad_max_len[key], max_len)
            for i in range(len(self.data[key])):
                if len(self.data[key][i]) < max_len:
                    self.data[key][i] += [self.pad_item[key]] * \
                        (max_len - len(self.data[key][i]))
                else:
                    self.data[key][i] = self.data[key][i][-max_len:]
                    self.origin_len[key][i] = max_len

    def get_origin_len(self, key):
        return self.origin_len[key]

    def to_tensor(self, device):
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = torch.LongTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'float':
                self.data[key] = torch.FloatTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'array of int':
                for i in range(len(self.data[key])):
                    for j in range(len(self.data[key][i])):
                        try:
                            self.data[key][i][j] = torch.LongTensor(np.array(self.data[key][i][j])).to(device)
                        except TypeError:
                            print('device is ', device)
                            exit()
            elif self.feature_name[key] == 'no_pad_int':
                for i in range(len(self.data[key])):
                    self.data[key][i] = torch.LongTensor(np.array(self.data[key][i])).to(device)
            elif self.feature_name[key] == 'no_pad_float':
                for i in range(len(self.data[key])):
                    self.data[key][i] = torch.FloatTensor(np.array(self.data[key][i])).to(device)
            elif self.feature_name[key] == 'no_tensor':
                pass
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float, array of int, no_pad_float.\
                    and you give {}'.format(self.feature_name[key]))

    def to_ndarray(self):
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = np.array(self.data[key])
            elif self.feature_name[key] == 'float':
                self.data[key] = np.array(self.data[key])
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float, array of int, no_pad_float.\
                    and you give {}'.format(self.feature_name[key]))