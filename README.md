## Seq-2-Seq-OCR

Handwritten text recognition using Seq-2-Seq modelling.

## Data

Download the dataset from 
http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

(you need first to register). 

Put the IAM dataset "words" data into a "data" folder. 

Folder structure should be data/a01 ... data/a02 and data/words.txt


![picture alt](http://www.brightlightpictures.com/assets/images/portfolio/thethaw_header.jpg "Title is optional")

# Project structure

base: base classes for data_generator, model, trainer and predictor

callbacks: custom callbacks

configs: config file

data_generators: data generator class and data augmentation functions

dataset: folder containing the dataset and labels

experiments: folder that contained saved snapshots. They can be used for restoring the training 

figures: contains plots and figures

models: keras model

notebooks: notebooks for testing code 

predictors: predictor class 

preprocessing: preprocessing functions (reading and normalizing the image)

snapshots: graph and weights of the trained model

tensorboard: tensorboard logs

test_images: images from the dataset that can be used for testing 

traines: trainer classes

utils: various utils including the script to generate ground truth labels


# Input

The input json can be created from utils/create_labels.py and follows this structure:

dataset['train'], ['val'], ['test']

Each split gives a list of dictionary: {'filename': FILENAME, 'label': LABEL}.


## Weights

The graph and trained weights can be found at:

https://drive.google.com/open?id=1Y_xJexxYcbU9eSd_poS_qKAW9eJg6Gbv



# Train

To train a model run:

python3 main.py -c configs/config.yml --train

If you set "weights_initialization" in config.yml you can use a pretrained model to inizialize the weights. 

During training the best and last snapshots can be stored, if thos options are set in "callbacks" inside config.yml.


# Inference 

To predict on full test set run: 

python3 main.py -c configs/config.yml --predict_on_test

To predict on a single image:

python3 main.py -c configs/config.yml --predict --filename FILENAME

In "/test_images" there are some images that can be used for testing. 


## Performance

On test set:

CER:  13.681 %

WER:  28.243 %


# To do

- [x] Document code following Docstring conventions


# References