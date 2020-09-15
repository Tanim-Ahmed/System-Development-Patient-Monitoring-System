import datetime

import numpy as np
import cv2
import sys
import pymysql as mdb
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import QDialog, QApplication, QFormLayout, QLineEdit, QLabel, QWidget, QPushButton, QHBoxLayout, \
    QMessageBox, QMainWindow
from PyQt5.uic import loadUi
import sqlite3
import pickle
from PIL import Image
import os
from PyQt5 import QtCore, QtGui, QtWidgets

# from trainer import getImagesWithID


class Dboxing(QDialog):
    def __init__(self):
        super(Dboxing, self).__init__()
        loadUi('C:/Users/Manik/PycharmProjects/first/main.ui', self)
        self.image = None
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        self.setWindowTitle("Patient Monitoring")
        self.setGeometry(300, 100, 480, 280)
        # self.start_rec.clicked.connect(self.start_cam)
        # self.stop_rec.clicked.connect(self.stop_cam)
        self.detect_button.setCheckable(True)
        self.detect_button.toggled.connect(self.detector)
        self.face_Enabled = False
        self.input_button.clicked.connect(self.execute_input)
        self.updateButton.clicked.connect(self.update_info)
        # self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect_webcam_face(self, status):
        self.start_cam()
        self.stop_cam()
        self.start_cam()
        if status:
            self.detect_button.setText('Stop Detection')
            self.face_Enabled = True
        else:
            self.detect_button.setText('Detect')
            self.face_Enabled = False

    def getProfile(self, id):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cursor = db.cursor()
                cursor.execute("SELECT * FROM data WHERE ID=" + str(id))
            profile = None
            for row in cursor:
                profile = row
            db.close()
            return profile
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)

    # Function for drawing the rectangle

    def trainer(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path = 'C:/Users/Manik/PycharmProjects/first/Dataset'

        def getImagesWithID(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            IDs = []
            for imagePath in imagePaths:
                faceImg = Image.open(imagePath).convert('L');
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
                # cv2.imshow("training ",faceNp)
                cv2.waitKey(10)
            return np.array(IDs), faces

        Ids, faces = getImagesWithID(path)
        recognizer.train(faces, Ids)
        if os.path.isdir('C:/Users/Manik/PycharmProjects/first/Recognizer') is False:
            os.mkdir('C:/Users/Manik/PycharmProjects/first/Recognizer')
        recognizer.save('C:/Users/Manik/PycharmProjects/first/Recognizer/trianing.yml')
        cv2.destroyAllWindows()

    def detector(self):
        self.trainer()
        faceDetect = cv2.CascadeClassifier("C:/Users/Manik/PycharmProjects/first/haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(1)
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read("C:/Users/Manik/PycharmProjects/first/Recognizer/trianing.yml")
        id = 0
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h / 2), (0, 255, 0), 2)
                id, conf = rec.predict(gray[y:y + h, x:x + w])
                profile = self.getProfile(id)  # to send the face id got from trainer.py and get all details of that id
                if profile is not None:
                    cv2.putText(img, "ID: " + str(profile[0]), (x, y + h + 15), font, 1, (255, 0, 0), 1)
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), font, 1, (255, 0, 0), 1)
                    cv2.putText(img, "Ref Doctor: " + str(profile[2]), (x, y + h + 60), font, 1, (255, 0, 0), 1)
                    cv2.putText(img, "Age :" + str(profile[3]), (x, y + h + 90), font, 1, (255, 0, 0), 1)

                    cv2.putText(img, "Appointment Date: " + str(profile[4]), (x, y + h + 120), font, 1, (255, 0, 0), 1)
                    cv2.putText(img, "Disease: " + str(profile[5]), (x, y + h + 150), font, 1, (255, 0, 0), 1)
                    cv2.putText(img, "Medicines: " + str(profile[6]), (x, y + h + 180), font, 1, (255, 0, 0), 1)

            cv2.imshow("Face", img)
            if (cv2.waitKey(1)) == ord(' '):
                break
        cam.release()
        cv2.destroyAllWindows()

    def detect_face(self, img):
        rec = cv2.face.LBPHFaceRecognizer_create()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            id, conf = self.rec.predict(gray[y:y + h, x:x + w])
            profile = self.getProfile(id)
            if profile is not None:
                cv2.cv.putText(cv2.cv.fromarray(img), str(profile[1]), [x, y + h + 30], 255)
        return img

    '''update info with id'''

    def update_info(self):
        input_start = Window2()
        input_start.exec_()

    def execute_input(self):
        input_start = Window()
        input_start.exec_()
        self.trainer()

    def dsetCreator(self):
        faceDetect = cv2.CascadeClassifier("C:/Users/Manik/PycharmProjects/first/haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(1)

        sampleNum = 0;
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            try:
                db = mdb.connect('localhost', 'root', '', 'mydatabase')
                with db:
                    cur = db.cursor()
                    cur.execute("SELECT MAX(id) FROM data")
                    res = cur.fetchone()
                    for (x, y, w, h) in faces:
                        sampleNum = sampleNum + 1;
                        # cur.execute("UPDATE data SET image = img")
                        cv2.imwrite("C:/Users/Manik/PycharmProjects/first/Dataset/User." + str(res[0]) + "." + str(
                            sampleNum) + ".jpg",
                                    gray[y:y + h, x:x + w])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.waitKey(100);
                    cv2.imshow("Face", img)
                cv2.waitKey(1);
                if (sampleNum >= 20):
                    break
            except mdb.Error as e:
                QMessageBox.about(self, 'Connection', 'Failed to connect')
                sys.exit(1)
        cam.release()
        cv2.destroyAllWindows()

    def stop_cam(self):
        self.timer.stop()


class StartPage(QDialog):
    def __init__(self):
        super().__init__()

        self.title = "Start Page"
        self.setGeometry(300, 100, 480, 280)
        self.initWindow()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))

    def initWindow(self):
        self.setWindowTitle(self.title)

        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter Password')
        self.lineedit2.setEchoMode(QLineEdit.Password)
        self.lineedit2.setGeometry(150, 50, 200, 30)

        self.button = QPushButton("Enter", self)
        self.button.setGeometry(150, 100, 200, 30)
        self.button.clicked.connect(self.startMainPage)

        '''self.button4 = QPushButton("Change Password", self)
        self.button4.setGeometry(150, 150, 200, 30)
        self.button4.clicked.connect(self.updatePass)'''

    def updatePass(self):
        start = updateWindow()
        start.exec_()

    def startMainPage(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()
                sql = ("""SELECT id FROM doctor WHERE pass=%s""",
                       (self.lineedit2.text()))
                cur.execute(*sql)
                rows = cur.fetchone()
                if rows is not None:
                    start = doctorPage()
                    start.exec_()
                    self.close()
                else:
                    QMessageBox.about(self, 'Error', 'Password Wrong')
                    self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
                    sys.exit(1)
                self.close()
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)


class adminPage(QDialog):
    def __init__(self):
        super(adminPage, self).__init__()
        loadUi('C:/Users/Manik/PycharmProjects/first/adminUI.ui', self)
        self.title = "Admin Page"
        self.setGeometry(300, 100, 480, 280)
        self.initWindow()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))

    def initWindow(self):
        self.add.clicked.connect(self.execute_doctor)

    def execute_doctor(self):
        start = addingPage()
        start.exec_()
        sys.exit(1)


class addingPage(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "Add New Doctor"
        self.setGeometry(300, 100, 480, 280)
        self.initWindow()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))

    def initWindow(self):
        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setPlaceholderText('Enter ID')
        self.lineedit1.setGeometry(150, 10, 200, 30)

        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter Name')
        self.lineedit2.setGeometry(150, 60, 200, 30)

        self.lineedit3 = QLineEdit(self)
        self.lineedit3.setPlaceholderText('Enter Password')
        self.lineedit3.setGeometry(150, 110, 200, 30)

        self.lineedit4 = QLineEdit(self)
        self.lineedit4.setPlaceholderText('Enter Employee type')
        self.lineedit4.setGeometry(150, 160, 200, 30)

        self.addDoctor = QPushButton("Add", self)
        self.addDoctor.setGeometry(150, 210, 200, 30)
        self.addDoctor.clicked.connect(self.insert)

    def insert(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()
                if self.lineedit1.text() == "" or self.lineedit2.text() == "" or self.lineedit3.text() == "":
                    QMessageBox.about(self, 'Failed', 'Every Field is required')
                    self.close()
                else:
                    cur.execute("INSERT into data(eid, name, pass, type)"
                                "VALUES('%s', '%s', '%s', '%s')" % (''.join(self.lineedit1.text()),
                                                                    ''.join(self.lineedit2.text()),
                                                                    ''.join(self.lineedit3.text()),
                                                                    ''.join(self.lineedit4.text())))
                    QMessageBox.about(self, 'Connection', 'Inserted successfully')
                    self.close()
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)


class workerPage(QDialog):
    def __init__(self):
        super(workerPage, self).__init__()
        loadUi('C:/Users/Manik/PycharmProjects/first/mainW.ui', self)
        self.image = None
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        self.setWindowTitle("Patient Monitoring")
        self.setGeometry(300, 100, 480, 280)
        self.detect_button.setCheckable(True)
        self.detect_button.toggled.connect(self.detector)
        self.face_Enabled = False
        self.insert_button.clicked.connect(self.execute_input)
        self.view_button.clicked.connect(self.execute_view)
        self.updatedoctor.clicked.connect(self.updateDoctor)


    def detect_webcam_face(self, status):
        self.start_cam()
        self.stop_cam()
        self.start_cam()
        if status:
            self.detect_button.setText('Stop Detection')
            self.face_Enabled = True
        else:
            self.detect_button.setText('Detect')
            self.face_Enabled = False

    def getProfile(self, id):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cursor = db.cursor()
                cursor.execute("SELECT * FROM data WHERE ID=" + str(id))
            profile = None
            for row in cursor:
                profile = row
            db.close()
            return profile
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)

    # Function for drawing the rectangle

    def trainer(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path = 'C:/Users/Manik/PycharmProjects/first/Dataset'

        def getImagesWithID(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            IDs = []
            for imagePath in imagePaths:
                faceImg = Image.open(imagePath).convert('L');
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
                # cv2.imshow("training ",faceNp)
                cv2.waitKey(10)
            return np.array(IDs), faces

        Ids, faces = getImagesWithID(path)
        recognizer.train(faces, Ids)
        if os.path.isdir('C:/Users/Manik/PycharmProjects/first/Recognizer') is False:
            os.mkdir('C:/Users/Manik/PycharmProjects/first/Recognizer')
        recognizer.save('C:/Users/Manik/PycharmProjects/first/Recognizer/trianing.yml')
        cv2.destroyAllWindows()

    def detector(self):
        self.trainer()
        faceDetect = cv2.CascadeClassifier("C:/Users/Manik/PycharmProjects/first/haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(1)
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read("C:/Users/Manik/PycharmProjects/first/Recognizer/trianing.yml")
        id = 0
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h / 2), (0, 255, 0), 2)
                id, conf = rec.predict(gray[y:y + h, x:x + w])
                profile = self.getProfile(id)  # to send the face id got from trainer.py and get all details of that id
                if profile is not None:
                    cv2.putText(img, "ID: " + str(profile[0]), (x, y + h + 10), font, 1, (0, 0, 255), 1)
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), font, 1, (0, 0, 255), 1)
                    cv2.putText(img, "Ref Doctor: " + str(profile[2]), (x, y + h + 60), font, 1, (255, 255, 0), 1)
                    cv2.putText(img, "Age :" + str(profile[3]), (x, y + h + 90), font, 1, (255, 255, 0), 1)

                    cv2.putText(img, "Appointment Date: " + str(profile[4]), (x, y + h + 120), font, 1, (255, 255, 0), 1)
                    cv2.putText(img, "Disease: " + str(profile[5]), (x, y + h + 150), font, 1, (255, 255, 0), 1)
                    cv2.putText(img, "Medicines: " + str(profile[6]), (x, y + h + 180), font, 1, (255, 255, 0), 1)

            cv2.imshow("Face", img)
            if (cv2.waitKey(1)) == ord(' '):
                break
        cam.release()
        cv2.destroyAllWindows()



    '''update info with id'''

    def update_info(self):
        input_start = Window2()
        input_start.exec_()

    def execute_input(self):
        input_start = Window()
        input_start.exec_()
        self.trainer()

    def execute_view(self):
        start = Ui_Form()
        start.exec_()

    def dsetCreator(self):
        faceDetect = cv2.CascadeClassifier("C:/Users/Manik/PycharmProjects/first/haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(1)

        sampleNum = 0;
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            try:
                db = mdb.connect('localhost', 'root', '', 'mydatabase')
                with db:
                    cur = db.cursor()
                    cur.execute("SELECT MAX(id) FROM data")
                    res = cur.fetchone()
                    for (x, y, w, h) in faces:
                        sampleNum = sampleNum + 1;
                        # cur.execute("UPDATE data SET image = img")
                        cv2.imwrite("C:/Users/Manik/PycharmProjects/first/Dataset/User." + str(res[0]) + "." + str(
                            sampleNum) + ".jpg",
                                    gray[y:y + h, x:x + w])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.waitKey(100);
                    cv2.imshow("Face", img)
                cv2.waitKey(1);
                if (sampleNum >= 20):
                    break
            except mdb.Error as e:
                QMessageBox.about(self, 'Connection', 'Failed to connect')
                sys.exit(1)
        cam.release()
        cv2.destroyAllWindows()

    def updateDoctor(self):
        start = updateDoctorWindow()
        start.exec_()


    def stop_cam(self):
        self.timer.stop()


class updateDoctorWindow(QDialog):
    def __init__(self):
        super(updateDoctorWindow, self).__init__()
        self.initialWindow()
        self.setGeometry(300, 100, 480, 320)
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
    def initialWindow(self):
        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setPlaceholderText('Enter id')
        self.lineedit1.setGeometry(150, 100, 200, 30)

        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter Doctor Name')
        self.lineedit2.setGeometry(150, 150, 200, 30)

        self.button = QPushButton("Update", self)
        self.button.clicked.connect(self.updateDoctor)
        self.button.setGeometry(150,200,200,30)


    def updateDoctor(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()
                cur2 = db.cursor()
                if self.lineedit1.text() == "" or self.lineedit2.text() == "":
                    QMessageBox.about(self, 'Failed', 'Every Field is required')
                else:
                    sql = ("""UPDATE data SET dname=%s WHERE id=%s""",
                           (self.lineedit2.text(), self.lineedit1.text()))
                    cur.execute(*sql)
                    cur.execute("SELECT MAX(id) FROM data")
                    res = cur.fetchone()
                    cur.execute("""SELECT * FROM data WHERE id=%s""",
                                (self.lineedit1.text()))
                    res2 = cur.fetchone()
                    cur2.execute("INSERT INTO details(id,name,dname,age,diseasename,medicine,date)"
                                 "VALUES('%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (''.join(self.lineedit1.text()),
                                                                                      ''.join(str(res2[1])),
                                                                                      ''.join(str(res2[2])),
                                                                                      ''.join(str(res2[3])),
                                                                                      ''.join(str(res2[4])),
                                                                                      ''.join(str(res2[5])),
                                                                                      ''.join(str(res2[6]))))
                    QMessageBox.about(self, 'Success', 'Data updated successfully')
                    self.close()
        except mdb.Error as e:
            QMessageBox.about(self, 'Error', 'Failed to connect')
            sys.exit(1)






class viewDetailsWindow(QDialog):
    def __init__(self):
        super(viewDetailsWindow, self).__init__()
        self.initialWindow()
        self.setGeometry(300, 100, 480, 280)
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
    def initialWindow(self):
        start = Ui_Form()
        start.exec_()



    def viewDetails(self):
        start = Ui_Form()
        start.exec_()




class Ui_Form(QDialog,):
    def __init__(self):
        super(Ui_Form, self).__init__()
        loadUi('C:/Users/Manik/PycharmProjects/first/viewDetails.ui', self)
        self.image = None
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        self.setWindowTitle("View Details")
        self.setGeometry(300, 100, 680, 320)
        self.loadButton.clicked.connect(self.LoadData)
        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setPlaceholderText('Enter a ID Number to get the details')
        self.lineedit1.setGeometry(180, 230, 300, 30)
        self.text = self.lineedit1.text()
    def LoadData(self):

        try:
            conn = mdb.connect('localhost', 'root', '', 'mydatabase')

            with conn:
                cur = conn.cursor()
                cur.execute("""SELECT id,name,dname,age,date,diseasename,medicine from details WHERE id=%s""",
                       (self.lineedit1.text()))

                self.cursor = QTextCursor(self.textEdit.document())
                self.cursor.insertText(str("( ID |   Name |  Doctor | Age | Apointment | Diseases | Medicines ) ") + '\n')

                for i in range(cur.rowcount):
                    result = cur.fetchall()
                    for row in result:

                        #self.cursor = QTextCursor(self.textEdit.document())
                        self.cursor.insertText(str(row)+'\n')

        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)


class doctor_Login(QDialog):
    def __init__(self):
        super().__init__()

        self.title = "Doctor Page"
        self.setGeometry(300, 100, 480, 280)
        self.initWindow()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))

    def initWindow(self):

        self.setWindowTitle(self.title)

        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setPlaceholderText('Enter ID')
        self.lineedit1.setGeometry(150, 10, 200, 30)

        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter Password')
        self.lineedit2.setEchoMode(QLineEdit.Password)
        self.lineedit2.setGeometry(150, 60, 200, 30)

        self.button = QPushButton("Enter", self)
        self.button.setGeometry(150, 110, 200, 30)
        self.button.clicked.connect(self.startMainPage)

        self.button4 = QPushButton("Change Password", self)
        self.button4.setGeometry(150, 200, 200, 30)
        self.button4.clicked.connect(self.updatePass)

    def updatePass(self):
        start = updateWindow()
        start.exec_()

    def startMainPage(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()
                sql = ("""SELECT name from doctor WHERE pass=%s""",
                       (self.lineedit2.text()))
                cur.execute(*sql)
                rows = cur.fetchone()
                if rows is not None:
                    start = doctorPage()
                    start.exec_()
                    self.close()
                else:
                    QMessageBox.about(self, 'Error', 'Password Wrong')
                    self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
                    sys.exit(1)
                self.close()
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)


class doctorPage(QDialog):
    def __init__(self):
        super(doctorPage, self).__init__()
        loadUi('C:/Users/Manik/PycharmProjects/first/mainD.ui', self)
        self.image = None
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        self.setWindowTitle("Patient Monitoring")
        self.setGeometry(300, 100, 480, 280)
        # self.start_rec.clicked.connect(self.start_cam)
        # self.stop_rec.clicked.connect(self.stop_cam)
        self.detect_button.setCheckable(True)
        self.detect_button.toggled.connect(self.detector)
        self.face_Enabled = False
        self.updateButton.clicked.connect(self.update_info)
        # self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.viewDetails.clicked.connect(self.viewdetails)
    def detect_webcam_face(self, status):
        self.start_cam()
        self.stop_cam()
        self.start_cam()
        if status:
            self.detect_button.setText('Stop Detection')
            self.face_Enabled = True
        else:
            self.detect_button.setText('Detect')
            self.face_Enabled = False

    def getProfile(self, id):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cursor = db.cursor()
                cursor.execute("SELECT * FROM data WHERE ID=" + str(id))
            profile = None
            for row in cursor:
                profile = row
            db.close()
            return profile
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)

    # Function for drawing the rectangle

    def trainer(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path = 'C:/Users/Manik/PycharmProjects/first/Dataset'

        def getImagesWithID(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            IDs = []
            for imagePath in imagePaths:
                faceImg = Image.open(imagePath).convert('L');
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
                # cv2.imshow("training ",faceNp)
                cv2.waitKey(10)
            return np.array(IDs), faces

        Ids, faces = getImagesWithID(path)
        recognizer.train(faces, Ids)
        if os.path.isdir('C:/Users/Manik/PycharmProjects/first/Recognizer') is False:
            os.mkdir('C:/Users/Manik/PycharmProjects/first/Recognizer')
        recognizer.save('C:/Users/Manik/PycharmProjects/first/Recognizer/trianing.yml')
        cv2.destroyAllWindows()

    def detector(self):
        self.trainer()
        faceDetect = cv2.CascadeClassifier("C:/Users/Manik/PycharmProjects/first/haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(1)
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read("C:/Users/Manik/PycharmProjects/first/Recognizer/trianing.yml")
        id = 0
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h / 2), (0, 255, 0), 2)
                id, conf = rec.predict(gray[y:y + h, x:x + w])
                profile = self.getProfile(id)  # to send the face id got from trainer.py and get all details of that id
                if profile is not None:
                    cv2.putText(img, "ID: " + str(profile[0]), (x, y + h + 10), font, 1, (0, 0, 255), 1)
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), font, 1, (0, 0, 255), 1)
                    cv2.putText(img, "Ref Doctor: " + str(profile[2]), (x, y + h + 60), font, 1, (255, 255, 0), 1)
                    cv2.putText(img, "Age :" + str(profile[3]), (x, y + h + 90), font, 1, (255, 255, 0), 1)

                    cv2.putText(img, "Appointment Date: " + str(profile[4]), (x, y + h + 120), font, 1, (255, 255, 0),
                                1)
                    cv2.putText(img, "Disease: " + str(profile[5]), (x, y + h + 150), font, 1, (255, 255, 0), 1)
                    cv2.putText(img, "Medicines: " + str(profile[6]), (x, y + h + 180), font, 1, (255, 255, 0), 1)

            cv2.imshow("Face", img)
            if (cv2.waitKey(1)) == ord(' '):
                break
        cam.release()
        cv2.destroyAllWindows()

    def detect_face(self, img):
        rec = cv2.face.LBPHFaceRecognizer_create()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            id, conf = self.rec.predict(gray[y:y + h, x:x + w])
            profile = self.getProfile(id)
            if profile is not None:
                cv2.cv.putText(cv2.cv.fromarray(img), str(profile[1]), [x, y + h + 30], 255)
        return img

    '''update info with id'''

    def update_info(self):
        input_start = Window2()
        input_start.exec_()

    def execute_input(self):
        input_start = Window()
        input_start.exec_()
        self.trainer()

    def dsetCreator(self):
        faceDetect = cv2.CascadeClassifier("C:/Users/Manik/PycharmProjects/first/haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(1)

        sampleNum = 0;
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            try:
                db = mdb.connect('localhost', 'root', '', 'mydatabase')
                with db:
                    cur = db.cursor()
                    cur.execute("SELECT MAX(id) FROM data")
                    res = cur.fetchone()
                    for (x, y, w, h) in faces:
                        sampleNum = sampleNum + 1;
                        # cur.execute("UPDATE data SET image = img")
                        cv2.imwrite("C:/Users/Manik/PycharmProjects/first/Dataset/User." + str(res[0]) + "." + str(
                            sampleNum) + ".jpg",
                                    gray[y:y + h, x:x + w])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.waitKey(100);
                    cv2.imshow("Face", img)
                cv2.waitKey(1);
                if (sampleNum >= 20):
                    break
            except mdb.Error as e:
                QMessageBox.about(self, 'Connection', 'Failed to connect')
                sys.exit(1)
        cam.release()
        cv2.destroyAllWindows()

    def viewdetails(self):
        start = Ui_Form()
        start.exec_()

    def stop_cam(self):
        self.timer.stop()


class updateWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "Update Password"
        self.setGeometry(300, 100, 480, 280)
        self.initWindow()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
    def initWindow(self):
        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setPlaceholderText('Enter old password')
        self.lineedit1.setEchoMode(QLineEdit.Password)
        self.lineedit1.setGeometry(150, 50, 200, 30)

        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter new password')
        self.lineedit2.setEchoMode(QLineEdit.Password)
        self.lineedit2.setGeometry(150, 150, 200, 30)

        self.button = QPushButton("Update", self)
        self.button.clicked.connect(self.updatePass)

    def updatePass(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()
                sql = ("""SELECT name FROM doctor WHERE pass=%s""",
                       (self.lineedit1.text()))
                cur.execute(*sql)
                rows = cur.fetchone()
                if rows is not None:
                    sql = ("""UPDATE doctor SET pass=%s WHERE name=%s""",
                           (self.lineedit2.text(), "admin"))
                    cur.execute(*sql)
                    QMessageBox.about(self, 'Connection', 'Password updated successfully')
                    self.close()
                else:
                    QMessageBox.about(self, 'Error', 'Password Wrong')
                    self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
                    sys.exit(1)
                self.close()
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)


#for new insert
class Window(QDialog):
    def __init__(self):
        super().__init__()

        self.title = "Insert data"
        self.top = 100
        self.left = 100
        self.width = 480
        self.height = 300
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        self.initWindow()

    def initWindow(self):

        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setPlaceholderText('Enter patient name')
        self.lineedit1.setGeometry(100, 50, 200, 30)

        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter doctor name')
        self.lineedit2.setGeometry(100, 100, 200, 30)

        self.lineedit3 = QLineEdit(self)
        self.lineedit3.setPlaceholderText('Enter Age')
        self.lineedit3.setGeometry(100, 150, 200, 30)

        self.lineedit4 = QLineEdit(self)
        self.lineedit4.setPlaceholderText('Enter Date')
        self.lineedit4.setGeometry(100, 200, 200, 30)

        self.button = QPushButton("Insert Data", self)
        self.button.setGeometry(100, 250, 100, 30)
        self.button.clicked.connect(self.InsertData)
        self.setWindowTitle(self.title)
        self.setGeometry(300, 100, 480, 320)

        # self.exec_()

    def InsertData(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()
                cur2 = db.cursor()
                if self.lineedit1.text() == "" or self.lineedit2.text() == "" or self.lineedit3.text() == "":
                    QMessageBox.about(self, 'Failed', 'Every Field is required')
                    self.close()
                else:
                    cur.execute("INSERT INTO data(name, dname, age, date)"
                                 "VALUES('%s', '%s', '%s','%s')" % (''.join(self.lineedit1.text()),
                                                                    ''.join(self.lineedit2.text()),
                                                                    ''.join(self.lineedit3.text()),
                                                                    ''.join(self.lineedit4.text())))
                    cur.execute("SELECT MAX(id) FROM data")
                    res = cur.fetchone()
                    '''cur2.execute("INSERT INTO details(id, name, dname, age, date)"
                                "VALUES('%s', '%s', '%s', '%s','%s')" % (''.join(str(res[0])),
                                                              ''.join(self.lineedit1.text()),
                                                              ''.join(self.lineedit2.text()),
                                                              ''.join(self.lineedit3.text()),
                                                              ''.join(self.lineedit3.text()),
                                                              ''.join(self.lineedit4.text())))'''
                    QMessageBox.about(self, 'Connection', 'Data inserted successfully')
                    self.close()
                    self.datasetCreator()
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)

    def datasetCreator(self):
        faceDetect = cv2.CascadeClassifier("C:/Users/Manik/PycharmProjects/first/haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(1)

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
                    cv2.imwrite("C:/Users/Manik/PycharmProjects/first/Dataset/User." + str(res[0] + 1) + "." + str(
                        sampleNum) + ".jpg",
                                gray[y:y + h, x:x + w])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.waitKey(100);
                cv2.imshow("Face", img)
            cv2.waitKey(1);
            if (sampleNum >= 20):
                break
        cam.release()
        cv2.destroyAllWindows()

#for updating patient's info
class Window2(QDialog):
    def __init__(self):
        super().__init__()

        self.title = "Update data"
        self.top = 100
        self.left = 100
        self.width = 680
        self.height = 500
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
        self.initWindow()

    def initWindow(self):
        self.lineedit0 = QLineEdit(self)
        self.lineedit0.setPlaceholderText('Enter ID')
        self.lineedit0.setGeometry(100, 50, 200, 30)

        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setPlaceholderText('Enter disease name')
        self.lineedit1.setGeometry(100, 100, 200, 30)

        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter Medicines')
        self.lineedit2.setGeometry(100, 150, 200, 30)

        self.lineedit3 = QLineEdit(self)
        self.lineedit3.setPlaceholderText('Next Date')
        self.lineedit3.setGeometry(100, 200, 200, 30)

        self.button = QPushButton("Update Data", self)
        self.button.setGeometry(100, 250, 100, 30)
        self.button.clicked.connect(self.UpdateData)
        self.setWindowTitle(self.title)
        self.setGeometry(300, 100, 480, 320)
        # self.show()

    def UpdateData(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()
                cur2 = db.cursor()
                if self.lineedit1.text() == "" or self.lineedit2.text() == "" or self.lineedit0.text() == "" or self.lineedit3.text() == "":
                    QMessageBox.about(self, 'Failed', 'Every Field is required')
                else:
                    sql = ("""UPDATE data SET diseasename=%s, medicine=%s, date=%s WHERE id=%s""",
                           (self.lineedit1.text(), self.lineedit2.text(), self.lineedit3.text(), self.lineedit0.text()))
                    cur.execute(*sql)
                    cur.execute("SELECT MAX(id) FROM data")
                    res = cur.fetchone()
                    cur.execute("""SELECT * FROM data WHERE id=%s""",
                                (str(res[0])))
                    res2 = cur.fetchone()
                    cur2.execute("INSERT INTO details(id,name,dname,age,diseasename,medicine,date)"
                                 "VALUES('%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (''.join(str(res[0])),
                                                                                      ''.join(str(res2[1])),
                                                                                      ''.join(str(res2[2])),
                                                                                      ''.join(str(res2[3])),
                                                                                      ''.join(self.lineedit1.text()),
                                                                                      ''.join(self.lineedit2.text()),
                                                                                       ''.join(self.lineedit3.text())))
                    QMessageBox.about(self, 'Success', 'Data updated successfully')
                    self.close()
        except mdb.Error as e:
            QMessageBox.about(self, 'Error', 'Failed to connect')
            sys.exit(1)


class initialPage(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Initial Page")
        self.setGeometry(300, 100, 480, 280)
        self.initWindow()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))

        self.initWindow()

    def initWindow(self):
        self.button = QPushButton("Start Scanning", self)
        self.button.setGeometry(150, 50, 150, 30)
        self.button.clicked.connect(self.workerPage)

        self.button2 = QPushButton("Monitor as a Doctor",self)
        self.button2.setGeometry(150,100,150,30)
        self.button2.clicked.connect(self.doctorPage)

        self.button3 = QPushButton("Set Password For The First Time",self)
        self.button3.setGeometry(100,200,250,30)
        self.button3.clicked.connect(self.setPass)

    def setPass(self):
        start = setPassWindow()
        start.exec_()

    def workerPage(self):
        start = workerPage()
        start.exec_()


    def doctorPage(self):
        start = StartPage()
        start.exec_()



class setPassWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "Set Password"
        self.setGeometry(300, 100, 480, 280)
        self.initWindow()
        self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))

    def initWindow(self):
        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter a new password')
        self.lineedit2.setEchoMode(QLineEdit.Password)
        self.lineedit2.setGeometry(150, 150, 200, 30)

        self.button = QPushButton("Set Password", self)
        self.button.setGeometry(150,200,200,30)
        self.button.clicked.connect(self.setPass)

    def setPass(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()
                sql = ("""SELECT COUNT(name) FROM doctor WHERE pass=%s""",
                       (self.lineedit2.text()))
                cur.execute(*sql)
                rows = cur.fetchone()
                if rows is None:
                    cur.execute("INSERT into doctor(name,pass)"
                                "VALUES('%s','%s')" % (''.join('admin'),
                                                       ''.join(self.lineedit2.text())))
                    QMessageBox.about(self, 'Connection', 'Password added successfully')
                    self.close()
                else:
                    QMessageBox.about(self, 'Error', 'You have a password')
                    self.setWindowIcon(QtGui.QIcon("C:/Users/Manik/PycharmProjects/first/1.ico"))
                    sys.exit(1)
                self.close()
        except mdb.Error as e:
            QMessageBox.about(self, 'Connection', 'Failed to connect')
            sys.exit(1)


app = QApplication(sys.argv)

initwindow = initialPage()
initwindow.show()
# initwindow.exec_()

'''window = Dboxing()
window.setWindowTitle('Patient Monitoring')
window.setGeometry(100, 100, 580, 580)
window.show()
'''
sys.exit(app.exec_())