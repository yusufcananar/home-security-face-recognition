# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:02:46 2020

@author: yusuf
"""

import os
import numpy as np
import cv2
from PIL import Image

#####################CAREFUL#####################
recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "Dataset"

def getUserPath(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        
        cv2.imshow("training",faceNp)
        cv2.waitKey(1)
        
    return np.array(IDs), faces

Ids, faces = getUserPath(path)
recognizer.train(faces,Ids)

recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()


