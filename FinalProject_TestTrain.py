# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:52:51 2018

@author: Muhammad Shahbaz
"""

import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

imgDir = "C:/DataScience/Google Drive/MMAI/MMAI-894/Final Project/Data/indoorCVPR_09/Images"
imgCats = os.listdir(imgDir)
imgData = []
WIDTH = 200
HEIGHT = 200
    

for category in imgCats:
    imgPath = os.path.join(imgDir, category)
    classnum = imgCats.index(category)
    
    for img in os.listdir(imgPath):
        try:
            onehot = np.zeros((1,67),dtype=int)
            onehot[0,classnum]=1
            image = cv2.imread(os.path.join(imgPath, img),cv2.IMREAD_COLOR)
            imgData.append([cv2.resize(image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC),onehot])
        except Exception as e:
            pass
        
random.shuffle(imgData)

def create_train_test(imgData):
    trainData = []
    testData = []
    for i in range(0,int(.8*len(imgData))):
        trainData.append(imgData[i]) 
    for j in range(int(.8*len(imgData)),len(imgData)):
        testData.append(imgData[j])
    
    return trainData, testData

trData,ttData = create_train_test(imgData)
np.save("TestData.npy",ttData)
np.save("TrainData.npy",trData)

plt.imshow(ttData[1][0], cmap="gray")
imarr =  imgData[1][0]
plt.show()       
print(ttData[1][1])