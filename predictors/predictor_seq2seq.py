import os
import numpy as np
import cv2
import random
#import datetime
import io
import json
import keras
import string


from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Bidirectional, concatenate, add, Lambda, Permute
from keras.callbacks import EarlyStopping
#import keras.backend as K
from keras.optimizers import Adam
from keras.models import model_from_json
from preprocessing.preproc_functions import read_image_BW, normalize_0_mean_1_variance_BW

from base.base_predictor import BasePredictor

class PredictorSeq2Seq(BasePredictor):
    def __init__(self, config, graph_path, weights_path, num_decoder_tokens, max_decoder_seq_length,
                token_indices, reverse_token_indices, batch_size = 64):
        super().__init__(config)
        self.num_decoder_tokens = num_decoder_tokens
        self.max_decoder_seq_length = max_decoder_seq_length
        self.token_indices = token_indices
        self.reverse_token_indices = reverse_token_indices
        self.batch_size = batch_size
        self.model = self.load_model(graph_path, weights_path)
        self.encoder_graph, self.decoder_graph = self.build_graphs()
    
 #       self.callbacks_list = self.callbacks()
 #       self.loss = []
 #       self.acc = []
 #       self.val_loss = []
 #       self.val_acc = []
  #      self.init_callbacks()

    def load_model(self, graph_path, weights_path):
        json_file = open(graph_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(weights_path)
        
        return model
    
    def build_graphs(self):        
        
        latent_dim = self.config['network']['latent_dim']
        
        # Define inference models 
        
        #encoder
        #encoder_inputs_inference = Input(shape=(x_size, y_size, 1), name='input_encoder_inference')
        encoder_graph = Model(self.model.get_input_at(0)[0], self.model.get_layer("lstm_encoder").output[1:])

        #decoder
        decoder_state_h = Input(shape=(latent_dim,))
        decoder_state_c = Input(shape=(latent_dim,))
        decoder_states_input = [decoder_state_h, decoder_state_c]

        decoder_input = Input(shape=(1, self.num_decoder_tokens), name='input_decoder_inference')

        #get layers
        decoder_lstm_layer = self.model.get_layer("lstm_decoder")
        decoder_dense_layer = self.model.get_layer('dense')

        #construct decoder graph
        decoder_output, state_h, state_c = decoder_lstm_layer(decoder_input, initial_state=decoder_states_input)
        decoder_states = [state_h, state_c]

        decoder_output = decoder_dense_layer(decoder_output)
        decoder_graph = Model([decoder_input] + decoder_states_input, [decoder_output] + decoder_states)
        
        return encoder_graph, decoder_graph
    
    def predict(self, images):
     
        batch_size = self.batch_size
        
        n_images = images.shape[0]
        y_size = images.shape[1]
        x_size = images.shape[2]

        n_batches = (n_images + batch_size - 1) // batch_size

        output_list = []

        for i in range(n_batches):
    #    for i in range(1):

            batch_in, batch_out = (batch_size)* i, (batch_size)* i + batch_size

            if batch_out >= n_images:
                batch_out = n_images

            input_seq = images[batch_in:batch_out, :, :, :]
            batch_dim = batch_out - batch_in
            decoded_sentences = self.decode_sequence(input_seq, batch_dim, self.num_decoder_tokens, 
                                              self.max_decoder_seq_length,
                                              self.token_indices,
                                              self.reverse_token_indices)

            output_list.append(decoded_sentences)

            #flatten list
        flattened_list = [item for sublist in output_list for item in sublist]

        return flattened_list
    
    def decode_sequence(self, input_seq, batch_dim, num_decoder_tokens, max_decoder_seq_length, token_indices, reverse_token_indices):
        # Encode the input as state vectors.
        states_value = self.encoder_graph.predict(input_seq, batch_size = batch_dim)

        full_seq = np.zeros((batch_dim, 1, num_decoder_tokens))

        target_seq = np.zeros((batch_dim, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[:, token_indices['[']] = 1.

        for i in range(max_decoder_seq_length):
            output_tokens, h, c = self.decoder_graph.predict([np.expand_dims(target_seq, axis=1)] + states_value, 
                                                            batch_size = batch_dim )

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=1)
            #        confidence.append(np.max(output_tokens[:, -1, :]))

            target_seq = np.zeros((batch_dim, num_decoder_tokens))    
            for j in range(batch_dim):
                target_seq[j, sampled_token_index[j]] = 1.

            states_value = [h, c]

            #concatenate with the full sequence array
            full_seq = np.concatenate((full_seq, np.expand_dims(target_seq, axis=1)), axis=1)

        #remove first time element (is empty)
        full_seq = full_seq[:, 1:, :]
        decoded_sentences = []

        for i in range(batch_dim):
            sentence = []
            for j in range(full_seq.shape[1]):
                sampled_token_index = np.argmax(full_seq[i, j, :])   
                sentence.append(reverse_token_indices[sampled_token_index])

            decoded_sentences.append(''.join(sentence).replace("]", ""))  

        return decoded_sentences
