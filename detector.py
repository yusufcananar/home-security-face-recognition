# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:42:30 2020

@author: yusuf
"""
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import numpy as np
import cv2
import os


g_login = GoogleAuth()
g_login.LocalWebserverAuth()
drive = GoogleDrive(g_login)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recg = cv2.face.LBPHFaceRecognizer_create()

recg.read("C:/Users/yusuf/Documents/PROJECTS/Home Security System/Face Recognition/recognizer/trainingData.yml")

cam = cv2.VideoCapture(0)

ID = 0

#Check If User DATA file is existing
txtPath = os.listdir()
for txts in txtPath:
    if "C:/Users/yusuf/Documents/PROJECTS/Home Security System/Face Recognition/user_infos.txt" == txts:
        c = True
        break
    else:
        c = False

if c != True:
    user_infos = open("C:/Users/yusuf/Documents/PROJECTS/Home Security System/Face Recognition/user_infos.txt",'a')
    user_infos.close() 
    
user_info =[]
user_data = open("C:\\Users\\yusuf\\Documents\\PROJECTS\\Home Security System\\Face Recognition/user_infos.txt",'r')
read_data = user_data.read()
splitted = read_data.split("!")

names = []
ids = []
###################################
for i in range(1,len(splitted)):
    split_data = splitted[i].split("#",3)
    Id_data = str(split_data[2]).split(":")[1]
    name_data = str(split_data[1]).split(":")[1]
    names.append(name_data) 
    ids.append(Id_data)
 
sendit = True
count = 0
while (1):
    ret,img = cam.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        ID , conf = recg.predict(roi_gray)
        print(conf)
        
        if str(ID) in ids and conf <= 50: #more confident if close to 0
            for i in range(len(ids)):
                if str(ID) == ids[i]:
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(15,100,255),1)
                    img = cv2.rectangle(img,(x,y+h),(int(x+w),int(y+h*1.1)),(0,50,255),-1)
                    cv2.putText(img,names[i]+" "+str(conf),(x,int(y+h*1.1)), cv2.FONT_HERSHEY_DUPLEX , 0.7, (255,255,255),1)
                    
        elif conf > 50:
            sendit = True
            count += 1
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,50,50),1)
            img = cv2.rectangle(img,(x,y+h),(x+w,int(y+h*1.1)),(255,50,50),-1)
            cv2.putText(img,"Unknown/?",(x,int(y+h*1.1)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255),1)
            
            #Send datas to the cloud server
            if sendit == True:
                file1 = drive.CreateFile({"title":"Unknown"+str(count)+".png"})
                cv2.imwrite("unknown"+str(count)+".jpg",img)
                file1.SetContentFile("unknown"+str(count)+".jpg")
                file1.Upload() # Upload the file.
                sendit = False
                    
        
        
    cv2.imshow("face",img)
    if(cv2.waitKey(1)== ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
