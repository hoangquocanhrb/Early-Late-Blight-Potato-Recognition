import sys
from PyQt5.QtWidgets import QApplication,QMainWindow, QLabel, QDesktopWidget, QPushButton, QSizeGrip
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation
from PyQt5 import QtCore, QtGui, QtWidgets
import os 
import time 
import inspect
from datetime import datetime

import torch 
import torch.nn as nn 
from torchvision import transforms

from PIL import Image

# from Classify.model.VGG16 import VGG16
# from Segment.model.unet_model import UNet

import pyrebase

from utils import get_firebase_data, config, tools
import numpy as np
# import os 
# os.system('Pyrcc5 Photos/icons.qrc -o icons_rc.py')

from potato_main_window import Ui_MainWindow
from robot_control import Ui_Form

progressBarValue = 0

WINDOW_SIZE = 0

class MainWindow():
    def __init__(self, storage=None, database=None, segment_model=None, classify_model=None, device='cpu'):
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)
        self.main_win.clickPosition = QtCore.QPoint()

        # set up show img labels
        self.uic.origin_img_label.setScaledContents(True)
        self.uic.segment_img_label.setScaledContents(True)
        # remove window title
        self.main_win.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        # set main background to transparent
        self.main_win.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # put window to center sceen
        qtRectangle = self.main_win.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.main_win.move(qtRectangle.topLeft())

        # set timer to delay progressbar
        self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.appProgress)
        #timer interval in mili seconds
        # self.timer.start(100)

        # Button click 
        # Minimize window
        self.uic.minimizeButton.clicked.connect(lambda: self.main_win.showMinimized())
        # Restore or maximize window
        self.uic.restoreButton.clicked.connect(lambda: self.restore_or_maximize_window())
        # Close window
        self.uic.closeButton.clicked.connect(lambda: self.main_win.close())


        # Add click event to the top header to move window
        self.uic.main_header_frame.mouseMoveEvent = self.moveWindow

        #Left menu toggle button
        self.uic.left_menu_toggle_button.clicked.connect(lambda: self.slideLeftMenu())

        # Set the page that will be visible by defaul when app is opened
        self.uic.stackedWidget.setCurrentWidget(self.uic.home_page)

        #navigate to home page
        self.uic.home_button.clicked.connect(lambda: self.uic.stackedWidget.setCurrentWidget(self.uic.home_page))
        #navigate to control page
        # self.uic.control_button.clicked.connect(lambda: self.uic.stackedWidget.setCurrentWidget(self.uic.control_page))
        self.uic.control_button.clicked.connect(self.open_control_window)
        #navigate to setting page
        self.uic.setting_button.clicked.connect(lambda: self.uic.stackedWidget.setCurrentWidget(self.uic.setting_page))
        #navigate to record page
        self.uic.log_button.clicked.connect(lambda: self.uic.stackedWidget.setCurrentWidget(self.uic.history_page))
        self.datalog = []
        # self.load_log_data()

        # start menu button styling
        for w in self.uic.left_side_menu.findChildren(QPushButton):
            w.clicked.connect(self.applyButtonStyle)
        
        QSizeGrip(self.uic.size_grip)

        #set up firebase
        self.storage = storage
        self.database = database
        self.firebase_receiver = get_firebase_data.firebase_receiver(storage=self.storage, database=self.database)

        # set start call back click button
        self.uic.start_button.clicked.connect(self.controlTimer)
        #set timer callback func
        self.timer.timeout.connect(self.viewData)

        #Disease list
        self.diseases = {0: 'Early blight', 1: 'Late blight', 2: 'Healthy'}

        #Define model
        self.segment_model = segment_model
        self.segment_model.eval()
        self.classify_model = classify_model
        self.classify_model.eval()
        self.device = device
        #temp background
        self.back_ground = Image.open('utils/back_ground.JPG')
        self.back_ground = self.back_ground.resize((256, 256))
        self.back_ground = np.array(self.back_ground)

        #Send setting
        self.uic.done_set.clicked.connect(self.setting_send)

        self.uic.auto_mode.clicked.connect(self.set_auto)
        self.auto_mode = False 

    def set_auto(self):
        if self.auto_mode == False:
            self.firebase_receiver.send_control_signal('auto')
            self.auto_mode = True
            self.uic.auto_mode.setText('StopAuto')
        else:
            self.capture()
            self.auto_mode = False 
            self.uic.auto_mode.setText('Auto')

    def setting_send(self):
        speed_set = self.uic.speed_set.toPlainText()
        cap_duration = self.uic.cap_duration_set.toPlainText()
        print('Robot speed: ', speed_set)
        print('Cap duration: ', cap_duration)
        setting = {}
        setting['robot_speed'] = speed_set
        setting['capture_duration'] = cap_duration
        self.firebase_receiver.send_robot_setting(setting)

    def load_log_data(self):
        self.uic.tableLog.setRowCount(len(self.datalog))
        self.uic.tableLog.setItem(len(self.datalog)-1, 0, QtWidgets.QTableWidgetItem(self.datalog[-1]['Time']))
        self.uic.tableLog.setItem(len(self.datalog)-1, 1, QtWidgets.QTableWidgetItem(self.datalog[-1]['Disease']))
        self.uic.tableLog.setItem(len(self.datalog)-1, 2, QtWidgets.QTableWidgetItem(self.datalog[-1]['Accuracy']))
        self.uic.tableLog.setItem(len(self.datalog)-1, 3, QtWidgets.QTableWidgetItem(self.datalog[-1]['Note']))

    def open_control_window(self):
        self.sub_window = QMainWindow()
        self.sub_uic = Ui_Form()
        self.sub_uic.setupUi(self.sub_window)
        self.sub_window.show()
        self.sub_uic.forward.clicked.connect(self.di_thang)
        self.sub_uic.back.clicked.connect(self.di_lui)
        self.sub_uic.left.clicked.connect(self.quay_trai)
        self.sub_uic.right.clicked.connect(self.quay_phai)
        self.sub_uic.stop.clicked.connect(self.capture)

    def di_thang(self):
        self.firebase_receiver.send_control_signal('di_thang')
    def di_lui(self):
        self.firebase_receiver.send_control_signal('di_lui')
    def quay_trai(self):
        self.firebase_receiver.send_control_signal('quay_trai')
    def quay_phai(self):
        self.firebase_receiver.send_control_signal('quay_phai')
    def capture(self):
        self.firebase_receiver.send_control_signal('capture')

    #Use image from firebase
    def viewData(self):
        
        user = self.database.child('Data').get().val()
        if user['Image'] == 'sent':
            self.origin_img = self.firebase_receiver.getData()
            if self.origin_img != None:
                
                image = self.origin_img.copy()
                image = np.array(image)
                
                # self.origin_img = Image.fromarray(image)
                t = time.time()
                output_segment, pred, probs = tools.predict(
                    self.segment_model,
                    self.classify_model, 
                    self.origin_img,
                    self.back_ground,
                    self.device
                )
                print(probs)
                acc = round(max(probs).item(), 2)
                print('Infer time: ', time.time() - t)
                if pred is not None:
                    data = {'Time': str(datetime.now()), 'Disease': str(self.diseases[pred]), 'Accuracy': str(acc), 'Note': ''}
                    self.datalog.append(data)
                    self.load_log_data()

                height, width, channel = image.shape 
                step = channel * width

                qOrigin = QImage(image.data, width, height, step, QImage.Format_RGB888)
                qSegment = QImage(output_segment.data, width, height, step, QImage.Format_RGB888)
                
                self.uic.origin_img_label.setPixmap(QPixmap.fromImage(qOrigin))
                self.uic.segment_img_label.setPixmap(QPixmap.fromImage(qSegment))
                self.uic.result_label.setText(str(self.diseases[pred]))

    # Add mouse event to move window
    def mousePressEvent(self, event):
        self.main_win.clickPosition = event.globalPos()

    def moveWindow(self, e):
        if self.main_win.isMaximized() == False:
            #move window only when window is normal size
            if e.buttons() == Qt.LeftButton:
                self.main_win.move(self.main_win.pos() + e.globalPos() - self.main_win.clickPosition)
                self.main_win.clickPosition = e.globalPos()
                e.accept()

    def restore_or_maximize_window(self):
        global WINDOW_SIZE

        win_status = WINDOW_SIZE
        if win_status == 0:
            WINDOW_SIZE = 1
            self.main_win.showMaximized()
            #Update icon when maximize
            self.uic.restoreButton.setIcon(QIcon('Photos/Icon/full-screen-icon-11786.png'))
        else:
            WINDOW_SIZE = 0
            self.main_win.showNormal()
            #Update icon when minimize
            self.uic.restoreButton.setIcon(QIcon('Photos/Icon/full-screen-icon-11769.png'))

    def appProgress(self):
        global progressBarValue
        self.uic.loading_progressBar.setValue(progressBarValue)
        if progressBarValue > 100:
            self.timer.stop()
            # self.main_win.close()
            QtCore.QTimer.singleShot(0, lambda: self.uic.loading_label.setText('Loading completed'))
    
        progressBarValue += 1 

    def show(self):
        self.main_win.show()

    def slideLeftMenu(self):
        width = self.uic.left_side_menu.width()
        if width == 50:
            newWidth = 170
        else:
            newWidth = 50

        self.annimation = QPropertyAnimation(self.uic.left_side_menu, b"minimumWidth")
        self.annimation.setDuration(250)
        self.annimation.setStartValue(width)
        self.annimation.setEndValue(newWidth)
        self.annimation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.annimation.start()
    
    def applyButtonStyle(self):
        for w in self.uic.left_side_menu.findChildren(QPushButton):
            if w.objectName() != self.main_win.sender().objectName():
                defaultStyle = w.styleSheet().replace('border-left: 2px solid rgb(6, 70, 53);', "")
                w.setStyleSheet(defaultStyle)
        
        newStyle = self.main_win.sender().styleSheet() + ('border-left: 2px solid rgb(6, 70, 53);')
        self.main_win.sender().setStyleSheet(newStyle)

        return 

    def controlTimer(self):
        if not self.timer.isActive():
            self.timer.start(20)
            self.uic.start_button.setText('Stop')
        else:
            self.timer.stop()
            self.uic.start_button.setText('Start')

# if __name__ == "__main__":

#     firebase_storage = pyrebase.initialize_app(config.FIREBASE_CONFIG)
#     storage = firebase_storage.storage()
#     database = firebase_storage.database()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     data_transforms = transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     classify_model = VGG16(num_classes=3)
#     segment_model = UNet(n_channels=3, n_classes=1)
#     print(sys.path)
#     app = QApplication(sys.argv)
    
#     main_win = MainWindow(storage, database, device=device)
#     main_win.show()
#     sys.exit(app.exec())