''' Makes a dataset using given pictures'''

import cv2
import matplotlib as plt
import numpy as np
import os
import random 


data_dir = input("enter the picture dir: ")   #the main directory that contains the picture 
categories = ['rock','paper','scissor']     # the categories that we have 
img_size = 224

train_data = []

#lbl_to_class = {'rock':0,'paper':1,'scissor':2}
for categ in categories:
    path = os.path.join(data_dir, categ) #path to each individial category
    label = categories.index(categ)      # rock =0, paper=1, scissor=2
    for img in os.listdir(path):         # for each image in path we perform functions and append to an array 
        img_arr = cv2.imread(os.path.join(path,img))
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_arr = cv2.resize(img_arr, (img_size,img_size))
        train_data.append([img_arr,label])
                
print(len(train_data))

# we want to randomly shuffle the data
random.shuffle(train_data)

X = []
y = []

#creating two sperate arrays for features and labels
for features,labels in train_data:
    X.append(features)
    y.append(labels)


# shape the X and y
X = np.array(X).reshape(-1, img_size,img_size,3)
y = np.array(y).reshape(-1,1)

print("X shape is " + str(X.shape))
print("y shape is " + str(y.shape))

#save the data sets
np.save("features.npy",X)
np.save("labels.npy",y)
