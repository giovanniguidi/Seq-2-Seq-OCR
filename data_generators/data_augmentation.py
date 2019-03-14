import os
#import pandas as pd
import numpy as np
import cv2
#from PIL import Image
import random
#import tensorflow as tf
#import re
#import datetime
import io
#from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plt
#import pickle
#import string
#from utils import score_prediction, generate_token_index, y_labels, generate_dataset
#import json
#import keras
#import string


def data_aug_functions(img, config):
    #print('data aug')
    
    y_size = config['image']['image_size']['y_size']
    x_size = config['image']['image_size']['x_size']
    num_channels = config['image']['image_size']['num_channels'] 
    
    rotation_range = config['data_aug']['rotation_range']
    width_shift_range = config['data_aug']['width_shift_range']
    height_shift_range = config['data_aug']['height_shift_range']
    zoom_range = config['data_aug']['zoom_range']
    horizontal_flip = config['data_aug']['horizontal_flip']
    vertical_flip = config['data_aug']['vertical_flip']
    shear_range = config['data_aug']['shear_range']

    img_aug = img

    img_aug = flip(img_aug, horizontal_flip, vertical_flip, y_size, x_size, num_channels)
    img_aug = translation(img_aug, width_shift_range, height_shift_range, y_size, x_size, num_channels)
    img_aug = rotation(img_aug, rotation_range, y_size, x_size, num_channels)
    img_aug = zoom(img_aug, zoom_range, y_size, x_size, num_channels)
    img_aug = shear(img_aug, shear_range, y_size, x_size, num_channels)

    return img_aug

def flip(img, horizontal_flip, vertical_flip, y_size, x_size, num_channels):

    img_flipped = img

    if horizontal_flip:
        if np.random.randint(2) == 0:
            img_flipped = cv2.flip(img_flipped, 1)
    if vertical_flip:
        if np.random.randint(2) == 0:
            img_flipped = cv2.flip(img_flipped, 0)

    return img_flipped


def translation(img, width_shift_range, height_shift_range, y_size, x_size, num_channels):

  #  rows = y_size
  #  cols = x_size

    y_shift = np.random.uniform(-height_shift_range, height_shift_range) * y_size
    x_shift = np.random.uniform(-width_shift_range, width_shift_range) * x_size

    M = np.float32([[1,0, x_shift],[0,1, y_shift]])
    img_translated = cv2.warpAffine(img, M, (x_size, y_size))

    return img_translated


def rotation(img, rotation_range, y_size, x_size, num_channels):
    angle = np.random.uniform(-rotation_range, rotation_range)

    #print(img.shape)
#    rows = self.y_size
#    cols = self.x_size

    M = cv2.getRotationMatrix2D((x_size/2, y_size/2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (x_size, y_size))

    img_rot = np.reshape(img_rot, (y_size, x_size, num_channels))

    return img_rot



def zoom(img, zoom_range, y_size, x_size, num_channels):
#    rows = self.y_size
#    cols = self.x_size

    zoom = np.random.uniform(zoom_range[0], zoom_range[1]) 

    p1 = [5, 5] 
    p2 = [20, 5]
    p3 = [5, 20]

#        [5,5],[20,5],[5,20]

    pts1 = np.float32([p1, p2, p3])
    pts2 = np.float32([[x * zoom for x in p1], 
                       [x * zoom for x in p2], 
                       [x * zoom for x in p3] ])

    M = cv2.getAffineTransform(pts1,pts2) 
    zoomed_image = cv2.warpAffine(img,M,(x_size, y_size))

    return zoomed_image

def shear(img, shear_range, y_size, x_size, num_channels):
 #   rows = self.y_size
 #   cols = self.x_size

    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5 + shear_range*np.random.uniform() - shear_range/2
    pt2 = 20 + shear_range*np.random.uniform() - shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    M = cv2.getAffineTransform(pts1, pts2) 
    sheared_range = cv2.warpAffine(img, M, (x_size, y_size))

    return sheared_range
