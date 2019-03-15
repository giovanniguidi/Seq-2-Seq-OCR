from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

class BaseModel(object):
    """
    Base class for the keras model

    Attributes
    ----------
    config : dict
        configuration file 
    model : keras.models 
        keras model
    loss : keras.losses
        keras loss
    optimizer : keras.optimizers
        keras optimizer

    Methods
    -------
    build_model()
        build the model
    build_graph()
        create the computational graph
    set_optimizer()
        select the optimizer
    """
    
    def __init__(self, config):
        """
        Base constructor
        """
        self.config = config
        self.model = None
        self.loss = config['network']['loss']
        self.optimizer = self.set_optimizer(self.config['train']['optimizer'], self.config['train']['learning_rate'])

    def build_model(self):
        """Build the model

        Raises
        ------
        NotImplementedError
        """
        
        raise NotImplementedError

    def build_graph(self):
        """Create the computational graph

        Raises
        ------
        NotImplementedError
        """
        
        raise NotImplementedError

    def set_optimizer(self, optimizer_name, lr):
        """Select the optimizer

        Parameters
        ------
        optimizer_name: 
            name of the optimizer, either adam, sgd, rmsprop, adagrad, adadelta
        lr: fload
            learning rate
            
        Raises
        ------
        Exception
        """
                
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
            
