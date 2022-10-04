from lib2to3.pytree import convert
import os
from os import environ
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"
if __name__ == "__main__":
    suppress_qt_warnings()


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5 import uic
import sys
import firebase_admin
from firebase_admin import credentials, storage
cred = credentials.Certificate("C:\\Users\\PRAKHAR\\Desktop\\App(Python)\\automated-road-audit-ddd9bc3b3257.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'automated-road-audit.appspot.com'}) # connecting to firebase


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        uic.loadUi("MainWindow.ui", self)

        self.button = self.findChild(QPushButton, "pushButton")
        self.button1 = self.findChild(QPushButton, "pushButton_2")

        self.button.clicked.connect(self.clicker)
        #self.button1.clicked.connect(self.analyze)
        self.button1.clicked.connect(self.upload)
        self.button1.clicked.connect(self.popup)
        self.show()

    def clicker(self):
        global file_name
        file_name = QFileDialog.getOpenFileName(self ,'Open image file', "", "")

    #def analyze(self):
        #os.system("python TrafficSign.py")
    
 

    def upload(self):
        file_path = str(file_name[0])
        bucket = storage.bucket() # storage bucket
        blob = bucket.blob(file_path)
        blob.upload_from_filename(file_path)


    def popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Message")
        msg.setText("Image uploaded")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()


#Initialise the App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
