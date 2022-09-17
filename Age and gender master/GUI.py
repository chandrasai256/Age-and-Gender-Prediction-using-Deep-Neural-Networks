import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import math

import numpy
#load the trained model to classify sign
                         
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Age and Gender Detection')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def detect(filepath):
        print("hi")
        global label_packed
        image = Image.open(filepath)
        print(type(image))
        #image = image.resize((30,30))
        #image = numpy.expand_dims(image, axis=0)
        #image = numpy.array(image)
        #print(image)

        faceProto="opencv_face_detector.pbtxt"
        faceModel="opencv_face_detector_uint8.pb"
        ageProto="age_deploy.prototxt"
        ageModel="age_net.caffemodel"
        genderProto="gender_deploy.prototxt"
        genderModel="gender_net.caffemodel"

        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)','(20-25)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList=['Male','Female']

        faceNet=cv2.dnn.readNet(faceModel,faceProto)
        ageNet=cv2.dnn.readNet(ageModel,ageProto)
        genderNet=cv2.dnn.readNet(genderModel,genderProto)

        video=cv2.VideoCapture(image)
        print(video.read())
        print(video)
        padding=20
        while cv2.waitKey(1)<0 :
            hasFrame,frame=video.read()
            
            if not hasFrame:
                cv2.waitKey()
                break
            
            resultImg,faceBoxes=highlightFace(faceNet,frame)
            if not faceBoxes:
                print("No face detected")

            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                           min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                           :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')

                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')

                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Detecting age and gender", resultImg)
 
   
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        detect(file_path)
    except:
        pass

def highlightFace2(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


def detect2():
        faceProto="opencv_face_detector.pbtxt"
        faceModel="opencv_face_detector_uint8.pb"
        ageProto="age_deploy.prototxt"
        ageModel="age_net.caffemodel"
        genderProto="gender_deploy.prototxt"
        genderModel="gender_net.caffemodel"

        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)','(20-25)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList=['Male','Female']

        faceNet=cv2.dnn.readNet(faceModel,faceProto)
        ageNet=cv2.dnn.readNet(ageModel,ageProto)
        genderNet=cv2.dnn.readNet(genderModel,genderProto)

        video=cv2.VideoCapture(0)
        padding=20
        while cv2.waitKey(1)<0 :
            hasFrame,frame=video.read()
            if not hasFrame:
                cv2.waitKey()
                break
            
            resultImg,faceBoxes=highlightFace2(faceNet,frame)
            if not faceBoxes:
                print("No face detected")

            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                           min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                           :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')

                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')

                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Detecting age and gender", resultImg)



upload1=Button(top,text="Open Camera",command=detect2,padx=10,pady=5)
upload1.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload1.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Age and Gender Detection",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
