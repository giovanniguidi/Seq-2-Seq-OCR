import numpy as np
import string
from sklearn.model_selection import train_test_split
import pickle
import os
import cv2

def score_prediction(pred_list):
    n_identified = 0
    n_characters_identified = 0
    n_char_tot = 0

    list_accuracy_characters = []
    
    for i in range(len(pred_list)):
        pred_row = pred_list[i] 
        
        #check if date are the same
        if pred_row[0] == pred_row[1]:
            n_identified += 1
            
        if len(pred_row[1]) < len(pred_row[0]):
            pred_row[1] += '-' * (len(pred_row[0]) - len(pred_row[1]))    
        elif len(pred_row[1]) > len(pred_row[1]):

            pred_row[1] = pred_row[1][0:len(pred_row[0])]

        #check the number of characters that are the same
        for k in range(len(pred_row[0])):
            if pred_row[0][k] == pred_row[1][k]:
                n_characters_identified += 1
            n_char_tot += 1
        
    accuracy = n_identified/len(pred_list) 
    array_accuracy_characters = np.asarray(list_accuracy_characters)
    accuracy_char = float(n_characters_identified / n_char_tot)
    
    return accuracy, accuracy_char

def generate_token_index():
    numbers = ''.join([str(x) for x in range(10)])

    numbers += ':[]'
    num_decoder_tokens = len(numbers)
    
    return dict((k, v) for v, k in enumerate(numbers)), num_decoder_tokens

def y_labels(ys, max_decoder_seq_length, num_decoder_tokens, target_token_index):

    decoder_input_data = np.zeros( (len(ys),  max_decoder_seq_length + 2, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros( (len(ys), max_decoder_seq_length + 2, num_decoder_tokens), dtype='float32')
    
    for i, target_text in enumerate(ys):
        target_text = '[' + target_text + ']'
        
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    return decoder_input_data, decoder_target_data

def generate_dataset(pickle_path, images_path, x_size, y_size, test_set_size = 0.1, val_set_size = 0.05, random_state = 1):

    dataset = pickle.load(open(pickle_path, 'rb'))

    imgs = []
    gt = []

    for ind, element in enumerate(dataset):
        fpath = os.path.join(images_path, element['name'])
#        file_names.append(element['name'])
        img = cv2.imread(fpath, 0)
        resized_image = cv2.resize(img, (y_size, x_size)) 
        # normalize
        (m, s) = cv2.meanStdDev(resized_image)
        m = m[0][0]
        s = s[0][0]
        resized_image = resized_image - m
        resized_image = resized_image / s if s > 0 else resized_image
        
        #resized_image = resized_image/255. 
        
        imgs.append(resized_image)
        gt.append(element['tag'])
            
    print("images read")

    array_images = np.asarray(imgs)
    array_images = np.reshape(array_images, (-1, x_size, y_size, 1))

    # train_val / test split
    array_images_train_val, array_images_test, gt_train_val, gt_test = train_test_split(
        array_images, gt, random_state=random_state, test_size=test_set_size)  

    # train / val split
    array_images_train, array_images_val, gt_train, gt_val = train_test_split(
        array_images_train_val, gt_train_val, random_state=random_state, test_size=val_set_size)  

    #create pickle
    dataset = { "train": {"input": array_images_train, "label": gt_train}, 
            "valid": {"input": array_images_val,  "label": gt_val}, 
            "test": {"input": array_images_test, "label": gt_test} } 

    return dataset
