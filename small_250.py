import pandas as pd
import math
import numpy as np
#import matplotlib as plt
import keras
import pickle
import tensorflow
from tensorflow.python.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import Model
from keras import optimizers
import os, cv2
import tensorflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

imgPATH = '/home/samcano/pyfiles/250x250/line_imgs/'
csvPATH = '/home/samcano/pyfiles/250x250/250x250_labels.csv'
data = pd.read_csv(csvPATH, skipinitialspace=True)
coord1 = [0] * 508
coord2 = [0] * 508
imlist = [0] * 508


def makeDF():

    listing = os.listdir(imgPATH)
    imarray = 0
    for i in range(508):
        b = data['External ID'][i] + ".png"
        img_location = imgPATH + b
        for img in listing:
            if(img == b):
                imarray = cv2.imread(img_location)
                break
        imlist[i] = imarray
        data['External ID'][i] = b
    data['Image Array'] = imlist
    
    
    return data, imlist

def handleCSV():
    #parse csv for x and y values
    for i in range(508):
        data2 = data['Label'][i].split(':')
        coord1[i] = [0.0 ,0.0]
        coord2[i] = [0.0 ,0.0]
        x1 = data2[3].split(",")
        y1 = data2[4].split("}")
        x2 = data2[7].split(",")
        y2 = data2[8].split("}")

        #orient labels so origin is bottom left corner
        x1= float(x1[0])
        y1= -1 * (float(y1[0]) - 250.0)
        #y1= float(y1[0])
        x2= float(x2[0])
        y2= -1 * (float(y2[0]) - 250.0)
        #y2= float(y2[0])

        #load and normalize
        coord1[i] = [x1/250.0, y1/250.0]
        #coord1[i] = [x1, y1]
        coord2[i] = [x2/250.0, y2/250.0]
        #coord2[i] = [x2, y2]
    data["Label"] = coord1
    data["Label_2"] = coord2

    for i in range(508):
        hold = [-1, -1]
        if(data['Label'][i][0] == -0.0): data['Label'][i][0] = 0.0
        if(data['Label'][i][1] == -0.0): data['Label'][i][1] = 0.0
        if(data['Label_2'][i][0] == -0.0): data['Label_2'][i][0] = 0.0
        if(data['Label_2'][i][1] == -0.0): data['Label_2'][i][1] = 0.0

        if(data['Label_2'][i][0] < data['Label'][i][0]):
            hold[0] = data['Label'][i][0]
            hold[1] = data['Label'][i][1]
            data['Label'][i][0] = data['Label_2'][i][0]
            data['Label'][i][1] = data['Label_2'][i][1]
            data['Label_2'][i][0] = hold[0]
            data['Label_2'][i][1] = hold[1]
    data.to_pickle('lines_df.pickle')

    return data


def split_mse(label,pred):

    if len(label)!=len(pred) : return -1,-1
    n = float(len(pred))
    resid = np.transpose(label-pred)
    theta_mse = map(np.square,resid[0])
    theta_mse = math.sqrt(reduce(np.add,theta_mse)/n)
    r_mse = map(np.square,resid[1])
    r_mse = math.sqrt(reduce(np.add,r_mse)/n)

    return theta_mse,r_mse




def train():
    print "\n\nTraining\n\n"
    a = np.mean(imlist, -1)
    input_imgs = np.reshape(imlist, (508,250,250,3))
    input_shape = input_imgs[0].shape

    x, y = input_imgs, coord1
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)


    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((5, 5)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((5, 5)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='relu'))

    model.compile(optimizer='adamax', loss='mse',metrics=['accuracy', 'mse'])
    model.summary()
    model.get_config()
    model.layers[0].get_config()
    model.layers[0].input_shape
    model.layers[0].output_shape
    model.layers[0].get_weights()
    np.shape(model.layers[0].get_weights()[0])
    model.layers[0].trainable
    
    #fit the model
    hist = model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(X_test, y_test))

    return 0
'''
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print df
'''



if __name__ == '__main__':    
    makeDF()
    handleCSV()
    #train()
