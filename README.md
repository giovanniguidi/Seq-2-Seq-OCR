## Seq-2-Seq-OCR

Handwritten text recognition using Seq-2-Seq modelling with Keras.

This project is based on Sequence-To-Sequence modelling, initially introduced for machine translation by https://arxiv.org/pdf/1409.3215.pdf:

![picture alt](https://github.com/giovanniguidi/Seq-2-Seq-OCR/blob/master/figures/seq2seq.png "")

In our case the "sequences" are the images from the IAM dataset, that contain handwritten words. 

A convolutional neural network extracts the features from the images at different locations, depending on the receptive field of the final neurons. Those features are flattened and encoded by an LSTM. The decoder (another LSTM) predicts the labels (i.e. the words) using as initial states the output of the encoder.


## Data

Download the IAM dataset "words" (words/ and words.txt) from 
http://www.fki.inf.unibe.ch/databases/iam-handwriting-database (you need first to register). 

Put the "words" folder and words.txt file in "datasets". 

Folder structure should be ./datasets/words/a01, ./datasets/words/a02, ... and ./dataset/words.txt.

This is an example of images in the dataset:

![picture alt](https://github.com/giovanniguidi/Seq-2-Seq-OCR/blob/master/test_images/b01-049-01-00.png "")

## Project structure

The project has this structure:

base: base classes for data_generator, model, trainer and predictor that are extended

callbacks: custom callbacks (unused)

configs: configuration file

data_generators: data generator class and data augmentation functions

dataset: folder containing the dataset and the labels

experiments: contains snapshots, that can be used for restoring the training 

figures: plots and figures

models: neural network model

notebooks: notebooks for testing 

predictors: predictor class 

preprocessing: preprocessing functions (reading and normalizing the image)

snapshots: graph and weights of the trained model

tensorboard: tensorboard logs

test_images: images from the dataset that can be used for testing 

traines: trainer classes

utils: various utilities, including the one to generate the labels


## Input

The input json can be created from utils/create_labels.py and follows this structure:

dataset['train'], ['val'], ['test']

Each split gives a list of dictionary: {'filename': FILENAME, 'label': LABEL}.


## Weights

The graph and trained weights can be found at:

https://drive.google.com/open?id=1Y_xJexxYcbU9eSd_poS_qKAW9eJg6Gbv

If you want to use these weights be sure that you use the original labels in "datasets" folder, otherwise you may mix the train and test set, and you results will be unreliable.


## Train

To train a model run:

python3 main.py -c configs/config.yml --train

If you set "weights_initialization" in config.yml you can use a pretrained model to inizialize the weights. 

During training the best and last snapshots can be stored if you set those options in "callbacks" in config.yml.


## Inference 

To predict on the full test set run: 

python3 main.py -c configs/config.yml --predict_on_test

(you need a file labels.json in "dataset").

To predict on a single image run:

python3 main.py -c configs/config.yml --predict --filename FILENAME

In "./test_images/" there are some images that can be used for fast testing. 


## Performance

On the test set we get this performance (character error rate and word error rate):

CER:  13.681 %

WER:  28.243 %


## To do

- [x] Document code following Docstring conventions


## References


\[1\] [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)

\[2\] [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)