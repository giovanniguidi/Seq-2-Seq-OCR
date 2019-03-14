import os
import cv2
import numpy as np

def read_image_BW(images_folder, filename, y_size, x_size):
  
    fpath = os.path.join(images_folder, filename)
    img = cv2.imread(fpath, 0)
    
    if img is not None: 
        (wt, ht) = x_size, y_size
        (h, w) = img.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(img, newSize)
        target = np.ones([ht, wt]) * 255
        target[0:newSize[1], 0:newSize[0]] = img
        img = target          
    else:
        img = np.zeros((x_size, y_size)) 
            
    return img 


def read_image_BW_old(images_folder, filename, y_size, x_size):
  
    fpath = os.path.join(images_folder, filename)
    img = cv2.imread(fpath, 0)
    
    if img is not None: 
        img = cv2.resize(img, (x_size, y_size)) 
    else:
        img = np.zeros((x_size, y_size)) 
            
    return img 

def read_image_color(images_folder, filename, y_size, x_size):
  
    fpath = os.path.join(images_folder, filename)
    img = cv2.imread(fpath)
    
    if img is not None: 
        (wt, ht) = x_size, y_size
        (h, w) = img.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 3)) # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(img, newSize)
        target = np.ones([ht, wt, 3]) * 255
        target[0:newSize[1], 0:newSize[0], :] = img
        img = target          
    else:
        img = np.zeros((x_size, y_size, 3)) 
            
    return img 


def read_image_color_old(images_folder, filename, y_size, x_size):
 
    fpath = os.path.join(images_folder, filename)
    img = cv2.imread(fpath)
    img = cv2.resize(img, (x_size, y_size, 3)) 

    return img 

def normalize_0_mean_1_variance_BW(img, y_size, x_size):

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    img = np.reshape(img, (y_size, x_size, 1))
    
    return img

def normalize_0_mean_1_variance_color(img, y_size, x_size):

    # normalize
    (m, s) = cv2.meanStdDev(img)
#    m = m[0][0]
#    s = s[0][0]    
    m = np.average(m)
    s = np.average(s)
    img = img - m
    img = img / s if s > 0 else img

    img = np.reshape(img, (y_size, x_size, 3))

    return img


def normalize_0_1(img):

    img /= 255. 
#    img = np.reshape(img, (y_size, x_size, num_channels))

    return img