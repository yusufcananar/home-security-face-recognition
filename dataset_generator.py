# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:39:22 2020

@author: yusuf

Dataset Generator :
    Program wants user to input hers/his id to create a folder of user's
"""
import os
from datetime import date
import numpy as np
import cv2

def make_files(dirName):
# Create target directory & all intermediate directories if don't exists
    try:
        os.makedirs(dirName)    
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists") 


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

#Check If User DATA file is existing
txtPath = os.listdir()
for txts in txtPath:
    if "user_infos.txt" == txts:
        c = True
        break
    else:
        c = False

if c != True:
    user_infos = open("user_infos.txt",'a')
    user_infos.close()   

#Get user name from users and create a unique ID for each 
name = input("How should I recognize you, Boss?: ")
if name != None:
    ID = int(np.random.rand(1)[0]*(10**8))
    Date = date.today()
    
user_info =[]
user_data = open("user_infos.txt",'r')
read_data = user_data.read()
splitted = read_data.split("!")

#Make sure they have Unique IDs
for i in range(1,len(splitted)):
    split_data = splitted[i].split("#",3)
    Id_data = str(split_data[2]).split(":")[1]
    if ID != int(Id_data):
        continue
    else:
        print("I am trying to find the Best ID for you :)")
        ID = int(np.random.rand(1)[0]*(10**8))
        i=0
#Fill the text file with user infos  
user_info.append([name,ID,Date])
user_infos = open("user_infos.txt",'a')
user_infos.write("!#name:"+str(user_info[0][0])+"# ID number:"+str(user_info[0][1])+"# Date:"+str(user_info[0][2])+"#\n")
user_infos.close()

#Create your database folder for each user input
make_files("Dataset")

sampleCount = 0
detected = 0
while (1):
    ret,img = cam.read()
    sampleCount += 1
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        detected = 1
        cv2.waitKey(100)
    
    print("Please, move your head any kind of direction, for me :)\n I need all of your interesting photos")
    cv2.imshow("Facial Data Gathering",img)
    if detected == 1:
        cv2.imwrite("Dataset/"+str(name)+"."+str(ID)+"."+str(sampleCount)+".jpg",roi_gray)
    
    cv2.waitKey(1)
    if(sampleCount >= 500):
        break
cam.release()
cv2.destroyAllWindows()