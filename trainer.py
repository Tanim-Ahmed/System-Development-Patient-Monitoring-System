import os
import cv2 as cv
import numpy as np
from PIL import Image
import pickle

recognizer=cv.face.LBPHFaceRecognizer_create()
path='Dataset'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        #cv.imshow("training ",faceNp)
        cv.waitKey(10)
    return np.array(IDs), faces

Ids,faces=getImagesWithID(path)
recognizer.train(faces,Ids)
recognizer.save('Recognizer/trianing.yml')
cv.destroyAllWindows()
