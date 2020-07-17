import numpy as np 
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.models import save_model


#import the model
vgg_model = VGG16()
vgg_model.summary()

vgg_model.save("orignal_vgg.h5")


