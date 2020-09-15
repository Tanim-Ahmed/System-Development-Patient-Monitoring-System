import cv2
import numpy as np
import pymysql as mdb
import sqlite3
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

# id=raw_input("Enter user ID: ")
sampleNum = 0;
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    db = mdb.connect('localhost', 'root', '', 'mydatabase')
    with db:
        cur = db.cursor()
        cur.execute("SELECT MAX(id) FROM data")
        res = cur.fetchone()
        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1;
            # cur.execute("UPDATE data SET image = img")
            cv2.imwrite("Dataset/User." + str(res[0] + 1) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.waitKey(100);
        cv2.imshow("Face", img)
    cv2.waitKey(1);
    if (sampleNum >= 20):
        break
cam.release()
cv2.destroyAllWindows()
