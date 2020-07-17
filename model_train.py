'''Used to train a VGG16 model using transfer learning''' 

import numpy as np 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import  Dense
from keras import optimizers, layers
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import save_model, load_model
from keras.models import model_from_json

# load features and labels
# feature is (224,224,3) RGB image
X = np.load('features.npy')
y = np.load('labels.npy')

# split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 42)

# preprocess according to VGG16 -> subtracting the mean RGB value, computed on the training set, from each pixel
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

#load the model 
vgg_model=load_model("orignal_vgg.h5",compile = False)
vgg_model.summary()

#change the model into sequential type and exclude the final softmax layer
model = Sequential()
for layer in vgg_model.layers[:-1]:
    model.add(layer)

# just to check the summary
model.summary()

#make all the layers untrainable
for layer in model.layers:
    layer.trainable = False

#Add final dense softmax layer 
model.add(Dense(3, activation='softmax'))
model.summary()

#intermediate model
#model.save("final_vgg16.h5")

# compile and train the model
model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=20, epochs=2)

#test
score = model.evaluate(X_test,y_test)
print("test_loss, test_acc:" ,score)

# to save
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 




