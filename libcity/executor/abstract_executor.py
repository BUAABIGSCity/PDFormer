class AbstractExecutor(object):

    def __init__(self, config, model):
        raise NotImplementedError("Executor not implemented")

    def train(self, train_dataloader, eval_dataloader):
        raise NotImplementedError("Executor train not implemented")

    def evaluate(self, test_dataloader):
        raise NotImplementedError("Executor evaluate not implemented")

    def load_model(self, cache_name):
        raise NotImplementedError("Executor load cache not implemented")

    def save_model(self, cache_name):
        raise NotImplementedError("Executor save cache not implemented")
