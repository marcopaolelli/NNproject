import numpy as np
import random
import string
import os
import sys
import argparse
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from data_preprocess import getDataset

batchSize = 16
learningRate = 0.001
nbEpoch = 100
"""
def cnn():
    convnet = input_data(shape=[None, 217, 64, 4], name='input')

    convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)
                         
    convnet = fully_connected(convnet, 1024, activation='elu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, 5, activation='softmax')
    convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

    model = tflearn.DNN(convnet)
    return model
"""

def cnn():
    convnet = input_data(shape=[None, 217, 64, 4], name='input')
    
    convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)
    """
    convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)
    """
    convnet = conv_2d(convnet, 160, 4, weights_init="Xavier")
    convnet = max_pool_2d(convnet, 4)
                    
    convnet = fully_connected(convnet, 512, activation='relu')
    
    convnet = fully_connected(convnet, 512, activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, 5, activation='softmax')
    convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

    model = tflearn.DNN(convnet)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test","slice"])
args = parser.parse_args()

model = cnn()

if "train" in args.mode:

    #Create or load new dataset
    train1_X, train1_y, train2_X, train2_y, validation_X, validation_y = getDataset(mode="train")
    print(train1_X.shape)
    print(train1_y.shape)
    print(train2_X.shape)
    print(train2_y.shape)


    train_X = np.concatenate((train1_X, train2_X), axis=0)
    train_y = np.concatenate((train1_y, train2_y), axis=0)

    #Define run id for graphs
    run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))

    #Train the model
    print("[+] Training the model...")
    model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_X, validation_y), snapshot_step=100, show_metric=True, run_id=run_id)
    print("    Model trained! ")

    #Save trained model
    print("[+] Saving the weights...")
    model.save('musicDNN.tflearn')
    print("[+] Weights saved!")

if "test" in args.mode:

    #Create or load new dataset
    test_X, test_y = getDataset(mode="test")

    #Load weights
    print("[+] Loading weights...")
    model.load('musicDNN.tflearn')
    print("    Weights loaded!")

    testAccuracy = model.evaluate(test_X, test_y)[0]
    print("[+] Test accuracy: {} ".format(testAccuracy))
