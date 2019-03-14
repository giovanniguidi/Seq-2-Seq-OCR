class BaseTrain(object):
    def __init__(self, config, model, train_generator, val_generator):
#        self.model = model
#        self.data = data
        self.config = config
        self.model = model.model
        self.train_generator = train_generator
        self.val_generator = val_generator
        
    def train(self):
        raise NotImplementedError

    def save_model(self, model, graph_path, weights_path):
        raise NotImplementedError
