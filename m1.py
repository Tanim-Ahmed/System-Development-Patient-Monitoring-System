import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit, QPushButton, QMessageBox
from qtpy import QtGui
import MySQLdb as mdb


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Insert data"
        self.top = 100
        self.left = 100
        self.width = 680
        self.height = 500

        self.initWindow()

    def initWindow(self):
        self.lineedit1 = QLineEdit(self)
        self.lineedit1.setPlaceholderText('Enter your name')
        self.lineedit1.setGeometry(200, 100, 200, 30)

        self.lineedit2 = QLineEdit(self)
        self.lineedit2.setPlaceholderText('Enter your Phone')
        self.lineedit2.setGeometry(200, 150, 200, 30)

        self.lineedit3 = QLineEdit(self)
        self.lineedit3.setPlaceholderText('Enter your Email')
        self.lineedit3.setGeometry(200, 200, 200, 30)

        self.button = QPushButton("Insert Data", self)
        self.button.setGeometry(200, 250, 100, 30)
        self.button.clicked.connect(self.InsertData)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def connectDB(self):
        try:
            db = mdb.connect('localhost','root','','mydatabase')
            QMessageBox.about(self,'Connection ','Connected')
        except mdb.Error as e:
            QMessageBox.about(self,'Connection','Failed to connect')
            sys.exit(1)


    def InsertData(self):
        try:
            db = mdb.connect('localhost', 'root', '', 'mydatabase')
            with db:
                cur = db.cursor()

                cur.execute("INSERT INTO data(name, email, phone)"
                            "VALUES('%s', '%s', '%s')" % (''.join(self.lineedit1.text()),
                                                        ''.join(self.lineedit3.text()),
                                                        ''.join(self.lineedit2.text())))
                QMessageBox.about(self, 'Connection', 'Data inserted successfully')
                self.close()
        except mdb.Error as e:
            QMessageBox.about(self,'Connection','Failed to connect')
            sys.exit(1)

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec_())
