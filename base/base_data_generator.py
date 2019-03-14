import keras

class BaseDataGenerator(keras.utils.Sequence):
    def __init__(self, config, dataset, shuffle, use_data_augmentation):
        #super().__init__()
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
        "number of batches in epoch"
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError

    def data_generation(self, dataset_temp):        
        raise NotImplementedError
