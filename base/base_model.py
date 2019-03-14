from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.loss = config['network']['loss']
        self.optimizer = self.set_optimizer()

    def build_model(self):
        raise NotImplementedError

    def build_graph(self):
        raise NotImplementedError

    def set_optimizer(self):
        
        optimizer_name = self.config['train']['optimizer']
        lr = self.config['train']['learning_rate']
                
        if optimizer_name == 'adam':
            optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)        
        elif optimizer_name == 'sgd':
            optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
        elif optimizer_name == 'adagrad':
            optimizer = Adagrad(lr=lr, epsilon=None, decay=0.0)
        elif optimizer_name == 'adadelta':
            optimizer = Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
        else:
            raise Exception('Optimizer unknown')
            
        return optimizer
            
