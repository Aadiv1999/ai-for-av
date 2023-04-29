#----------------------------------------------------------------------------------------------------------------
# Required Dependencies
#----------------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import os
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import cv2
from keras.callbacks import ModelCheckpoint

#----------------------------------------------------------------------------------------------------------------
# Helper Functions
#----------------------------------------------------------------------------------------------------------------

def load_images(folder,images,Id,target):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA))
            target.append(Id)
    return images, target



#----------------------------------------------------------------------------------------------------------------
# Traffic Sign Classification
#----------------------------------------------------------------------------------------------------------------

#Read in the different traffic signs and assign a target ID for the different signs
# IDs: stop sign = 0, speed 30 = 1, speed 60 = 2, speed 90 = 3, speed limit 30 = 4, speed limit 40 = 5, speed limit 60 = 6

batch_size = 128
num_classes = 7
epochs = 24

# Read image sets and assign classification number
train_path = "./traffic_signs/train"
val_path = "./traffic_signs/val"

image_set_paths = ['stop/', 'speed_30/', 'speed_60/', 'speed_90/', 'speed_limit_30/', 'speed_limit_40/', 'speed_limit_60/']

train_images = [] #set of images to train classification
train_target = [] #target classification for training images

val_images = [] #set of images to validate classification
val_target = [] #target classification for validating images
Id = 0
for img_set in image_set_paths:
    train_folder = os.path.join(train_path, img_set)
    train_images, train_target = load_images(train_folder,train_images,Id,train_target)

    val_folder = os.path.join(val_path, img_set)
    val_images, val_target = load_images(val_folder,val_images,Id,val_target)

    Id+=1

#The training dataset
x_train = np.array(train_images)
print(x_train.shape)
y_train = np.asarray(train_target).astype('float')

#The testing dataset
x_test = np.array(val_images)
y_test = np.asarray(val_target).astype('float')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape:', x_train[0].shape)
print("Training samples: {}".format(x_train.shape[0]))
print("Test samples: {}".format(x_test.shape[0]))

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=x_train[0].shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(7, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Fit data to model
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="traffic_sign_weights.hdf5", verbose=0, save_best_only=True) # save best model

model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpointer],verbose=1,epochs=100)

model.load_weights('traffic_sign_weights.hdf5') # load weights from best model

score = model.evaluate(x_test,y_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))

y_comp = np.argmax(y_test,axis=1)

pred2 = model.predict(x_test)
pred2 = np.argmax(pred2,axis=1)

submit_df = pd.DataFrame({'pred':pred2,'y':y_comp})

# Save a copy if you like
submit_df.to_csv('traffic_sign_classification_output.csv',index=False)
