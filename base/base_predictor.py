import os
import numpy as np
#import cv2
#import string

class BasePredictor(object):
    
    def __init__(self, config):
        self.config = config
        
    def predict(self):
        raise NotImplementedError
        
    def load_model(self, graph_path, weights_path):
        raise NotImplementedError