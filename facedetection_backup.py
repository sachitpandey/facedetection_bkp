#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:37:54 2020

@author: syntax
"""


from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from fr_utils import *
from inception_blocks_v2 import *

path = ".../images/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((96,96), Image.ANTIALIAS)
            imResize.save(f + e, 'JPEG', quality=90)

resize()


FRmodel = faceRecoModel(input_shape=(3, 96, 96))


def triplet_loss(y_true, y_pred, alpha = 0.2):
    #Returns: value of the loss
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
    
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
    
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}

new_face = input("Do you want to add a new face? (y/n): ")
if new_face.upper() == 'Y':
    name = input("Enter your name: ")
    #camera takes photo
    
    
elif new_face.upper() == 'N':
    print("OK")

database["{}".format(name)] = img_to_encoding("/home/syntax/images/zzz.jpg", FRmodel)    


def verify(image_path, identity, database, model):
    #Function that verifies if the person on the "image_path" image is "identity".

    encoding = img_to_encoding(image_path, model)
    
    dist = np.linalg.norm(encoding - database[identity])
    
    if (dist < 0.7):
        print("It's " + str(identity) + ", welcome in!")
        access_grant = True
    else:
        print("It's not " + str(identity) + ", please go away.)
        access_grant = False
        
    return dist, access_grant

verify("../images/uuu.jpg", "sachit", database, FRmodel)

