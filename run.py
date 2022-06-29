from tkinter.tix import Tree
import numpy as np 
import cv2
import tensorflow as tf
from playsound import playsound

my_model=tf.saved_model.load('G:\my_project\Real-time-hand-detecyion-\inference\saved_model')
print('Load Model')

def creatbondingBox(image,threshold=0.25):
    inputTensor=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    inputTensor=tf.convert_to_tensor(inputTensor)
    inputTensor=inputTensor[tf.newaxis,...]
    detection=my_model(inputTensor)
    bboxs=detection['detection_boxes'][0].numpy()
    classIndexes=detection['detection_classes'][0].numpy().astype(np.int32)
    classeScores=detection['detection_scores'][0].numpy()
    imH, imW, imC=image.shape
    bboxInd=tf.image.non_max_suppression(bboxs,classeScores,max_output_size=50,iou_threshold=threshold,
                                        score_threshold=threshold)
    
    if len(bboxInd)!=0:
        for i in bboxInd:
            bbox=tuple(bboxs[i].tolist())
            classConfidence=round(100*classeScores[i])
            ymin,xmin,ymax,xmax=bbox
            ymin, xmin, ymax, xmax=(ymin*imH,xmin*imW,ymax*imH,xmax*imW)
            ymin, xmin, ymax, xmax=int(ymin),int(xmin),int(ymax),int(xmax)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=(0,0,255),thickness=1)
            cv2.putText(image,'head',(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

cam=cv2.VideoCapture(0)
while True:
    _,frame=cam.read()
    creatbondingBox(frame)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)==27:
        break

