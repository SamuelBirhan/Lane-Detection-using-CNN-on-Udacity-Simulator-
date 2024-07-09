import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def getName(filePath):
    return filePath.split('/')[-1]

def importDataInfo(path):
    colomuns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = colomuns)
    data['Center'] = data['Center'].apply(getName)
    if "Steering" in data.columns:
        # Replace data outside desired range (-0.5 to 0.5) with 0
        data["Steering"] = data["Steering"].where((data["Steering"] >= -0.5) & (data["Steering"] <= 0.5), 0)
        print("Data outside -0.5 to 0.5 range replaced with 0 successfully!")
    else:
        print("Steering column not found in the CSV file.")
    # print(data.head())
    print('Total images imported', data.shape[0])
    return data

def balanceData(data, display = True):
    nBins = 31
    samplesPerBin = 1500
    hist, bins = np.histogram(data['Steering'], bins=nBins)
    # print(bins)
    if display:
        center = (bins[:-1] + bins[1:])*0.5
        # print(center)
        plt.bar(center, hist, width=0.06, align='center')
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.title('Distribution of Steering Angles')
        plt.xlabel('Steering Angle')
        plt.ylabel('Frequency')
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print('Removed Images', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print('Remaining Images', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        # print(center)
        plt.bar(center, hist, width=0.06, align='center')
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.title('Distribution of Balanced Steering Angles')
        plt.xlabel('Steering Angle')
        plt.ylabel('Frequency')
        plt.show()
    return data



def loadData(path, data):
    imagesPath = []
    steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        #print(indexedData)
        imagesPath.append(os.path.join(path, 'IMG', indexedData.iloc[0]))
        steering.append(float(indexedData.iloc[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

def augmentImages(imgPath, steering):
    img = mpimg.imread(imgPath)
    ##PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)
    ## ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    #Brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.6,1.2))
        img = brightness.augment_image(img)
    #Flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering

def preProcessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img, (200,66)) # because NVIDIA uses this size
    img = img/255 # normalization

    return img


def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImages(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield (np.asarray(imgBatch),
               np.asarray(steeringBatch))

def creatModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    # model.compile(Adam(learning_rate=0.0001), loss='mse')
    model.compile(Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

    return model






