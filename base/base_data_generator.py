import keras

class BaseDataGenerator(keras.utils.Sequence):
    """
    Base class for data generation

    Attributes
    ----------
    config : dict
        configuration file 
    dataset : list 
        dataset list, each elemen is a dictionary {'filename':, 'label'}
    dataset_len : int
        number of elements in dataset
    images_folder : str
        folder containing the images
    batch_size : int
        size of the batch
    y_size : int
        image height (first dimension in keras)
    x_size : int
        image width (second dimension in keras)
    shuffle : bool
        shuffle the dataset at the end of the epoch (good for training)
    use_data_aug : int
        use data aumentation functions

    Methods
    -------
    __len__()
        returns the length of the dataset 
    __getitem__(index)
        returns a batch of data
    on_epoch_end()
        function called when finishing an epoch
    data_generation(dataset_temp)     
        read and normalize images of the batch 
    """
    
    def __init__(self, config, dataset, shuffle, use_data_augmentation):
        """
        Base constructor
        """
        self.config = config
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.images_folder = self.config['images_folder']
        self.batch_size = self.config['train']['batch_size']
        self.y_size = self.config['image']['image_size']['y_size']
        self.x_size = self.config['image']['image_size']['x_size']
        self.num_channels = self.config['image']['image_size']['num_channels']
        self.shuffle = shuffle
        self.use_data_aug = use_data_augmentation
        
    def __len__(self):
        """Gives the number of batches per epoch
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """Returns a batch of data

        Parameters
        ------
        index: int
            index of the batch
            
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Function called when finishing an epoch

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def data_generation(self, dataset_temp):      
        """Read and normalize images of the batch 

        Parameters
        ------
        dataset_temp: list
            list of IDs of the elements in the batch
            
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
