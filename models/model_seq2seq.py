import os
import numpy as np
import cv2
import random
import datetime
import io
import json
import keras
import string


from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Bidirectional, concatenate, add, Lambda, Permute
from keras.callbacks import EarlyStopping
#import keras.backend as K
from keras.optimizers import Adam

from base.base_model import BaseModel

class ModelSeq2Seq(BaseModel):
    def __init__(self, config, max_decoder_seq_length, num_decoder_tokens):
        super().__init__(config)
        self.y_size = config['image']['image_size']['y_size']
        self.x_size = config['image']['image_size']['x_size']
        self.num_channels = config['image']['image_size']['num_channels']

        self.latent_dim = config['network']['latent_dim']  # Latent dimensionality of the encoding space.
        self.max_decoder_seq_length = max_decoder_seq_length
        self.num_decoder_tokens = num_decoder_tokens

        self.model = self.build_model()

    def build_model(self):
        model = self.build_graph()        
        model.compile(optimizer = self.optimizer, loss = self.loss)

        return model
        
    def build_graph(self):
        ## Define an input sequence and process it.
        encoder_input = Input(shape=(self.y_size, self.x_size, self.num_channels), name='input_encoder')
        
        assert len(self.config['network']['num_filters']) == len(self.config['network']['conv_kernels'])
        
        encoder_graph = encoder_input

        #----------------convolutional network-----------------------
        for i in range(len(self.config['network']['num_filters'])):
            
            encoder_graph = Conv2D(self.config['network']['num_filters'][i], self.config['network']['conv_kernels'][i], padding='same', use_bias=False, name='conv_' + str(i+1))(encoder_graph)
            if self.config['network']['use_batch_norm'] == True:
                encoder_graph = BatchNormalization(name='batch_norm_' + str(i+1))(encoder_graph)
            encoder_graph = Activation('relu', name='activation_' + str(i+1))(encoder_graph)
            encoder_graph = MaxPooling2D(pool_size=(2, 2), padding='valid', name='maxpool_' + str(i+1))(encoder_graph)

        conv_shapes = encoder_graph.shape[1:]
        timesteps = int(conv_shapes[0]*conv_shapes[1])
        num_features = int(conv_shapes[2])

        encoder_graph = Reshape((-1, num_features), name='reshape')(encoder_graph)

        #-------------------------encoder---------------------------
        encoder = LSTM(self.latent_dim, return_state=True, name='lstm_encoder')
        _, state_h, state_c = encoder(encoder_graph)
        encoder_states = [state_h, state_c]

        #------------------------decoder------------------------------
        decoder_input = Input(shape=(self.max_decoder_seq_length, self.num_decoder_tokens), name='input_decoder_teacher_forcing')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='lstm_decoder')
        decoder_graph, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)

        #------------------------dense--------------------------------
        decoder_dense = Dense(self.num_decoder_tokens, activation=None, name='dense')
        decoder_graph = decoder_dense(decoder_graph)

        decoder_output = Activation('softmax', name='softmax')(decoder_graph)

        model = Model([encoder_input, decoder_input], decoder_output)
        model.summary()
        
        #model.load_weights('./snapshots/snapshot_last.h5')
        
        return model
    
    def save_graph(self, model, graph_path):
                   
        model_json = model.to_json()
        with open(graph_path, "w") as json_file:
            json_file.write(model_json)
