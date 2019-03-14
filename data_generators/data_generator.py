import os
import numpy as np
import cv2
import io
import json
#import keras
import string

from base.base_data_generator import BaseDataGenerator
from data_generators.data_augmentation import data_aug_functions
from preprocessing.preproc_functions import read_image_BW, read_image_color, normalize_0_mean_1_variance_BW, normalize_0_mean_1_variance_color


class DataGenerator(BaseDataGenerator):
    def __init__(self, config, dataset, shuffle=True, use_data_augmentation=False):
        super().__init__(config, dataset, shuffle, use_data_augmentation) 
        
#        self.decoder_tokens = sorted(string.ascii_uppercase + string.digits + '[]' )    
        self.decoder_tokens = sorted(string.printable)  
        self.num_decoder_tokens = len(self.decoder_tokens)
#        self.max_decoder_seq_length = self.get_decoder_seq_length()
        self.max_seq_length = self.config['network']['max_seq_lenght']
        self.token_indices, self.reverse_token_indices = self.token_indices()
        self.indices = np.arange(self.dataset_len)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.dataset_len / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        dataset_temp = [self.dataset[k] for k in indices]
        
        # Generate data
        X, y1, y2 = self.data_generation(dataset_temp)

        return [X, y1], y2

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)
       #print('reshuffling data')

    def get_full_dataset(self):
        
        dataset_images = []
        dataset_labels = []
        
        for elem in self.dataset:
#            normalized_image = self.preprocess_image(elem['filename'])
        
            y_size = self.config['image']['image_size']['y_size']
            x_size = self.config['image']['image_size']['x_size']
            num_channels = self.config['image']['image_size']['num_channels']
            convert_to_grayscale = self.config['image']['convert_to_grayscale']
            
#            image = read_image_BW(images_folder, elem['filename'], y_size, x_size)
            
#            print(self.images_folder, elem['filename'])
            #read image
            if num_channels == 1 or (num_channels == 3 and convert_to_grayscale):
                image = read_image_BW(self.images_folder, elem['filename'], y_size, x_size)
                image = normalize_0_mean_1_variance_BW(image, y_size, x_size)
            else:
                image = read_image_color(self.images_folder, elem['filename'], y_size, x_size)
                image = normalize_0_mean_1_variance_color(image, y_size, x_size)

#            image = normalize_0_mean_1_variance_BW(image, y_size, x_size)

            label = elem['label']
            
            dataset_images.append(image)
            dataset_labels.append(label)
            
        dataset_images = np.asarray(dataset_images)
        
        return dataset_images, dataset_labels

        
    def data_generation(self, dataset_temp):        
        batch_x = []
        batch_y1 = []
        batch_y2 = []

        for elem in dataset_temp:            
            #image = self.read_image(elem['filename'])

            y_size = self.config['image']['image_size']['y_size']
            x_size = self.config['image']['image_size']['x_size']
            num_channels = self.config['image']['image_size']['num_channels']
            convert_to_grayscale = self.config['image']['convert_to_grayscale']
            
            #images_folder = self.config['images_folder']
            #print(os.path.join(self.images_folder, elem['filename']))
            #image = read_image_BW(self.images_folder, elem['filename'], y_size, x_size)
            #print(image.shape)
            
            #read image, apply data augmentation and normalize
            if num_channels == 1 or (num_channels == 3 and convert_to_grayscale):
                image = read_image_BW(self.images_folder, elem['filename'], y_size, x_size)
                if self.use_data_aug:
                    image = data_aug_functions(image, self.config)
                image = normalize_0_mean_1_variance_BW(image, y_size, x_size)
            else:
                image = read_image_color(self.images_folder, elem['filename'], y_size, x_size)                
                if self.use_data_aug:
                    image = data_aug_functions(image, self.config)                
                image = normalize_0_mean_1_variance_color(image, y_size, x_size)
 
            #print(image.shape)

            decoder_input_data, decoder_target_data = self.one_hot_labels(elem['label'], self.max_seq_length, 
                                                                           self.num_decoder_tokens, 
                                                                           self.token_indices)

            batch_x.append(image)
            batch_y1.append(decoder_input_data)
            batch_y2.append(decoder_target_data)
        
        batch_x = np.asarray(batch_x, dtype = np.float32)
        batch_y1 = np.asarray(batch_y1, dtype = np.float32)
        batch_y2 = np.asarray(batch_y2, dtype = np.float32)
        
        return batch_x, batch_y1, batch_y2
    

#    def get_decoder_seq_length(self):
#        max_decoder_seq_length = 0.
        #decoder_tokens = set()

#        for elem in self.dataset:
#            word = elem['label']

#            if len(word) > max_decoder_seq_length:
#                max_decoder_seq_length = len(word)

#        max_decoder_seq_length += 1
#        return max_decoder_seq_length
    
    def token_indices(self):

        target_token_index = dict((k, v) for v, k in enumerate(self.decoder_tokens))
        reverse_target_token_index = dict((i, char) for char, i in target_token_index.items())

        return target_token_index, reverse_target_token_index
    
    def one_hot_labels(self, label, max_seq_length, num_decoder_tokens, target_token_index):

        decoder_input_data = np.zeros((max_seq_length, num_decoder_tokens), dtype='float32')
        decoder_target_data = np.zeros((max_seq_length, num_decoder_tokens), dtype='float32')
                    
        #generate one hot label for input decoder
        for t, char in enumerate('[' + label):
            if t < max_seq_length:
                decoder_input_data[t, target_token_index[char]] = 1.
#            if t == 0:
#                decoder_input_data[t, target_token_index['[']] = 1.
#            elif t > 0 and t < max_seq_length:
#                decoder_input_data[t, target_token_index[char]] = 1.

        #generate one hot label for output decoder
        for t, char in enumerate(label + ']'):
            if t < max_seq_length:
                decoder_target_data[t, target_token_index[char]] = 1.           
#            if t < max_seq_length - 1:
#                decoder_target_data[t, target_token_index[char]] = 1.
#            elif t == max_seq_length - 1:
#                decoder_target_data[t, target_token_index[']']] = 1.
                                   
        return decoder_input_data, decoder_target_data    
    
