# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'potato_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(784, 472)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.main_header_frame = QtWidgets.QFrame(self.centralwidget)
        self.main_header_frame.setMaximumSize(QtCore.QSize(16777215, 45))
        self.main_header_frame.setStyleSheet("background-color: rgb(240, 187, 98);\n"
"QFrame{\n"
"    border-bottom: 1px solid #000;\n"
"}")
        self.main_header_frame.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.main_header_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_header_frame.setObjectName("main_header_frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.main_header_frame)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.title_bar_container_frame = QtWidgets.QFrame(self.main_header_frame)
        self.title_bar_container_frame.setStyleSheet("")
        self.title_bar_container_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.title_bar_container_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.title_bar_container_frame.setObjectName("title_bar_container_frame")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.title_bar_container_frame)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.left_menu_toggle_frame = QtWidgets.QFrame(self.title_bar_container_frame)
        self.left_menu_toggle_frame.setMinimumSize(QtCore.QSize(50, 0))
        self.left_menu_toggle_frame.setMaximumSize(QtCore.QSize(50, 16777215))
        self.left_menu_toggle_frame.setStyleSheet("QFrame{\n"
"    background-color: rgb(240, 187, 98);\n"
"}\n"
"QPushButton{\n"
"    padding: 5px 10px;\n"
"    border: none;\n"
"    border-radius: 5px;\n"
"    background-color: #000;\n"
"    color: #fff\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(237, 212, 0);\n"
"}")
        self.left_menu_toggle_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.left_menu_toggle_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.left_menu_toggle_frame.setObjectName("left_menu_toggle_frame")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.left_menu_toggle_frame)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.left_menu_toggle_button = QtWidgets.QPushButton(self.left_menu_toggle_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_menu_toggle_button.sizePolicy().hasHeightForWidth())
        self.left_menu_toggle_button.setSizePolicy(sizePolicy)
        self.left_menu_toggle_button.setMinimumSize(QtCore.QSize(0, 0))
        self.left_menu_toggle_button.setMaximumSize(QtCore.QSize(50, 16777215))
        self.left_menu_toggle_button.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.left_menu_toggle_button.setAutoFillBackground(False)
        self.left_menu_toggle_button.setStyleSheet("background-color: rgb(240, 187, 98);\n"
"padding-left: 5px;\n"
"")
        self.left_menu_toggle_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/Icon/menu-icon-19349.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.left_menu_toggle_button.setIcon(icon)
        self.left_menu_toggle_button.setIconSize(QtCore.QSize(35, 35))
        self.left_menu_toggle_button.setObjectName("left_menu_toggle_button")
        self.horizontalLayout_4.addWidget(self.left_menu_toggle_button)
        self.horizontalLayout_5.addWidget(self.left_menu_toggle_frame)
        self.title_bar_frame = QtWidgets.QFrame(self.title_bar_container_frame)
        self.title_bar_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.title_bar_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.title_bar_frame.setObjectName("title_bar_frame")
        self.horizontalLayout_5.addWidget(self.title_bar_frame)
        self.horizontalLayout_2.addWidget(self.title_bar_container_frame)
        self.top_right_btns_frame = QtWidgets.QFrame(self.main_header_frame)
        self.top_right_btns_frame.setMaximumSize(QtCore.QSize(100, 16777215))
        self.top_right_btns_frame.setStyleSheet("QPushButton{\n"
"    border-radius: 5px;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(237, 212, 0);\n"
"}")
        self.top_right_btns_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.top_right_btns_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.top_right_btns_frame.setObjectName("top_right_btns_frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.top_right_btns_frame)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.restoreButton = QtWidgets.QPushButton(self.top_right_btns_frame)
        self.restoreButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.restoreButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/Icon/full-screen-icon-11769.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.restoreButton.setIcon(icon1)
        self.restoreButton.setIconSize(QtCore.QSize(28, 28))
        self.restoreButton.setObjectName("restoreButton")
        self.horizontalLayout_3.addWidget(self.restoreButton)
        self.minimizeButton = QtWidgets.QPushButton(self.top_right_btns_frame)
        self.minimizeButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.minimizeButton.setStyleSheet("image: url(:/icon/Icon/minimize-icon-23774.png);")
        self.minimizeButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icon/Icon/minimize-icon-23774.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.minimizeButton.setIcon(icon2)
        self.minimizeButton.setIconSize(QtCore.QSize(24, 24))
        self.minimizeButton.setObjectName("minimizeButton")
        self.horizontalLayout_3.addWidget(self.minimizeButton)
        self.closeButton = QtWidgets.QPushButton(self.top_right_btns_frame)
        self.closeButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.closeButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icon/Icon/close-button-png-30241.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.closeButton.setIcon(icon3)
        self.closeButton.setIconSize(QtCore.QSize(24, 24))
        self.closeButton.setObjectName("closeButton")
        self.horizontalLayout_3.addWidget(self.closeButton)
        self.horizontalLayout_2.addWidget(self.top_right_btns_frame)
        self.verticalLayout.addWidget(self.main_header_frame)
        self.main_body_frame = QtWidgets.QFrame(self.centralwidget)
        self.main_body_frame.setStyleSheet("background-color: rgb(92, 53, 102);")
        self.main_body_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.main_body_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_body_frame.setObjectName("main_body_frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.main_body_frame)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.left_side_menu = QtWidgets.QFrame(self.main_body_frame)
        self.left_side_menu.setMaximumSize(QtCore.QSize(50, 16777215))
        self.left_side_menu.setStyleSheet("QFrame{\n"
"    background-color:  rgb(240, 187, 98);\n"
"}\n"
"QPushButton{\n"
"    padding: 20px 10px;\n"
"    border: none;\n"
"    border-radius: 10px;\n"
"    background-color: rgb(240, 187, 98);\n"
"    color: #fff;\n"
"    border-left: 2px solid transparent;\n"
"border-bottom: 2px solid transparent;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    background-color: rgb(237, 212, 0);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"background-color: rgb(0,92,157);\n"
"border-bottom: 2px solid rgb(255,165,72);\n"
"}")
        self.left_side_menu.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.left_side_menu.setFrameShadow(QtWidgets.QFrame.Raised)
        self.left_side_menu.setObjectName("left_side_menu")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.left_side_menu)
        self.verticalLayout_2.setContentsMargins(7, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.left_menu_top_buttons_frame = QtWidgets.QFrame(self.left_side_menu)
        self.left_menu_top_buttons_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.left_menu_top_buttons_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.left_menu_top_buttons_frame.setObjectName("left_menu_top_buttons_frame")
        self.formLayout = QtWidgets.QFormLayout(self.left_menu_top_buttons_frame)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setSpacing(0)
        self.formLayout.setObjectName("formLayout")
        self.home_button = QtWidgets.QPushButton(self.left_menu_top_buttons_frame)
        self.home_button.setMinimumSize(QtCore.QSize(100, 0))
        self.home_button.setAutoFillBackground(False)
        self.home_button.setStyleSheet("background-color: rgb(240, 187, 98);\n"
"background-repeat: none;\n"
"padding-left: 0px;\n"
"background-position: center left;\n"
"")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icon/Icon/homepage-icon-png-2574.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.home_button.setIcon(icon4)
        self.home_button.setIconSize(QtCore.QSize(30, 30))
        self.home_button.setAutoRepeatDelay(300)
        self.home_button.setObjectName("home_button")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.home_button)
        self.control_button = QtWidgets.QPushButton(self.left_menu_top_buttons_frame)
        self.control_button.setMinimumSize(QtCore.QSize(100, 0))
        self.control_button.setAutoFillBackground(False)
        self.control_button.setStyleSheet("background-repeat: none;\n"
"padding-left: 0px;\n"
"background-position: center left;\n"
"background-color: rgb(240, 187, 98);")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icon/Icon/controller-icon-32409.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.control_button.setIcon(icon5)
        self.control_button.setIconSize(QtCore.QSize(30, 30))
        self.control_button.setObjectName("control_button")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.control_button)
        self.log_button = QtWidgets.QPushButton(self.left_menu_top_buttons_frame)
        self.log_button.setMinimumSize(QtCore.QSize(100, 0))
        self.log_button.setStyleSheet("background-repeat: none;\n"
"padding-left: 0px;\n"
"background-position: center left;\n"
"background-color: rgb(240, 187, 98);")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icon/Icon/history-icon-png-4665.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.log_button.setIcon(icon6)
        self.log_button.setIconSize(QtCore.QSize(30, 30))
        self.log_button.setObjectName("log_button")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.log_button)
        self.verticalLayout_2.addWidget(self.left_menu_top_buttons_frame)
        self.setting_button = QtWidgets.QPushButton(self.left_side_menu)
        self.setting_button.setMinimumSize(QtCore.QSize(100, 0))
        self.setting_button.setStyleSheet("background-repeat: none;\n"
"padding-left: 15px;\n"
"background-position: center left;\n"
"background-color: rgb(240, 187, 98);")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icon/Icon/settings-icon-14973.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setting_button.setIcon(icon7)
        self.setting_button.setIconSize(QtCore.QSize(30, 30))
        self.setting_button.setObjectName("setting_button")
        self.verticalLayout_2.addWidget(self.setting_button)
        self.horizontalLayout.addWidget(self.left_side_menu)
        self.center_main_items = QtWidgets.QFrame(self.main_body_frame)
        self.center_main_items.setStyleSheet("background-color: rgb(244, 238, 169);")
        self.center_main_items.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.center_main_items.setFrameShadow(QtWidgets.QFrame.Raised)
        self.center_main_items.setObjectName("center_main_items")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.center_main_items)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.stackedWidget = QtWidgets.QStackedWidget(self.center_main_items)
        self.stackedWidget.setObjectName("stackedWidget")
        self.home_page = QtWidgets.QWidget()
        self.home_page.setStyleSheet("")
        self.home_page.setObjectName("home_page")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.home_page)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.home_frame = QtWidgets.QFrame(self.home_page)
        self.home_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.home_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.home_frame.setObjectName("home_frame")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.home_frame)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.segment_img_widget = QtWidgets.QWidget(self.home_frame)
        self.segment_img_widget.setMinimumSize(QtCore.QSize(220, 220))
        self.segment_img_widget.setMaximumSize(QtCore.QSize(220, 220))
        self.segment_img_widget.setStyleSheet("background-color: rgb(81, 170, 89);\n"
"border-radius: 10px;")
        self.segment_img_widget.setObjectName("segment_img_widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.segment_img_widget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.segment_img_label = QtWidgets.QLabel(self.segment_img_widget)
        self.segment_img_label.setMinimumSize(QtCore.QSize(200, 200))
        self.segment_img_label.setMaximumSize(QtCore.QSize(200, 200))
        self.segment_img_label.setStyleSheet("background-color: rgb(81, 120, 89);\n"
"")
        self.segment_img_label.setObjectName("segment_img_label")
        self.gridLayout_2.addWidget(self.segment_img_label, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.segment_img_widget, 0, 1, 1, 1)
        self.segment_name = QtWidgets.QLabel(self.home_frame)
        self.segment_name.setMinimumSize(QtCore.QSize(120, 30))
        self.segment_name.setMaximumSize(QtCore.QSize(120, 30))
        self.segment_name.setStyleSheet("border-color: rgb(115, 210, 22);\n"
"border-radius: 10px;\n"
"background-color: rgb(81, 170, 89);\n"
"font: 63 11pt \"URW Bookman\";\n"
"padding-left: 2px;\n"
"")
        self.segment_name.setObjectName("segment_name")
        self.gridLayout_3.addWidget(self.segment_name, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.origin_name = QtWidgets.QLabel(self.home_frame)
        self.origin_name.setMinimumSize(QtCore.QSize(120, 30))
        self.origin_name.setMaximumSize(QtCore.QSize(120, 30))
        self.origin_name.setStyleSheet("border-color: rgb(115, 210, 22);\n"
"border-radius: 10px;\n"
"background-color: rgb(81, 170, 89);\n"
"font: 63 11pt \"URW Bookman\";\n"
"padding-left: 10px;\n"
"")
        self.origin_name.setObjectName("origin_name")
        self.gridLayout_3.addWidget(self.origin_name, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.origin_img_widget = QtWidgets.QWidget(self.home_frame)
        self.origin_img_widget.setMinimumSize(QtCore.QSize(220, 220))
        self.origin_img_widget.setMaximumSize(QtCore.QSize(220, 220))
        self.origin_img_widget.setStyleSheet("background-color: rgb(81, 170, 89);\n"
"border-radius: 10px;")
        self.origin_img_widget.setObjectName("origin_img_widget")
        self.gridLayout = QtWidgets.QGridLayout(self.origin_img_widget)
        self.gridLayout.setObjectName("gridLayout")
        self.origin_img_label = QtWidgets.QLabel(self.origin_img_widget)
        self.origin_img_label.setMinimumSize(QtCore.QSize(200, 200))
        self.origin_img_label.setMaximumSize(QtCore.QSize(200, 200))
        self.origin_img_label.setStyleSheet("background-color: rgb(81, 120, 89);\n"
"")
        self.origin_img_label.setObjectName("origin_img_label")
        self.gridLayout.addWidget(self.origin_img_label, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.origin_img_widget, 0, 0, 1, 1)
        self.verticalLayout_7.addWidget(self.home_frame)
        self.stackedWidget.addWidget(self.home_page)
        self.history_page = QtWidgets.QWidget()
        self.history_page.setStyleSheet("")
        self.history_page.setObjectName("history_page")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.history_page)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame = QtWidgets.QFrame(self.history_page)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.tableLog = QtWidgets.QTableWidget(self.frame)
        self.tableLog.setGeometry(QtCore.QRect(110, 10, 401, 341))
        self.tableLog.setObjectName("tableLog")
        self.tableLog.setColumnCount(4)
        self.tableLog.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableLog.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableLog.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableLog.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableLog.setHorizontalHeaderItem(3, item)
        self.verticalLayout_5.addWidget(self.frame)
        self.stackedWidget.addWidget(self.history_page)
        self.setting_page = QtWidgets.QWidget()
        self.setting_page.setStyleSheet("")
        self.setting_page.setObjectName("setting_page")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.setting_page)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_2 = QtWidgets.QFrame(self.setting_page)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.done_set = QtWidgets.QPushButton(self.frame_2)
        self.done_set.setGeometry(QtCore.QRect(430, 270, 89, 25))
        self.done_set.setObjectName("done_set")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(80, 80, 67, 17))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setGeometry(QtCore.QRect(26, 160, 121, 20))
        self.label_3.setObjectName("label_3")
        self.speed_set = QtWidgets.QTextEdit(self.frame_2)
        self.speed_set.setGeometry(QtCore.QRect(190, 70, 171, 31))
        self.speed_set.setObjectName("speed_set")
        self.cap_duration_set = QtWidgets.QTextEdit(self.frame_2)
        self.cap_duration_set.setGeometry(QtCore.QRect(190, 160, 171, 31))
        self.cap_duration_set.setObjectName("cap_duration_set")
        self.verticalLayout_6.addWidget(self.frame_2)
        self.stackedWidget.addWidget(self.setting_page)
        self.verticalLayout_3.addWidget(self.stackedWidget)
        self.horizontalLayout.addWidget(self.center_main_items)
        self.right_side_menu = QtWidgets.QFrame(self.main_body_frame)
        self.right_side_menu.setMaximumSize(QtCore.QSize(100, 16777215))
        self.right_side_menu.setStyleSheet("background-color: rgb(186, 189, 182);")
        self.right_side_menu.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.right_side_menu.setFrameShadow(QtWidgets.QFrame.Raised)
        self.right_side_menu.setObjectName("right_side_menu")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.right_side_menu)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.start_button = QtWidgets.QPushButton(self.right_side_menu)
        self.start_button.setObjectName("start_button")
        self.verticalLayout_4.addWidget(self.start_button)
        self.auto_mode = QtWidgets.QPushButton(self.right_side_menu)
        self.auto_mode.setObjectName("auto_mode")
        self.verticalLayout_4.addWidget(self.auto_mode)
        self.result_label = QtWidgets.QLabel(self.right_side_menu)
        self.result_label.setMinimumSize(QtCore.QSize(50, 100))
        self.result_label.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.result_label.setFont(font)
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("result_label")
        self.verticalLayout_4.addWidget(self.result_label)
        self.label = QtWidgets.QLabel(self.right_side_menu)
        self.label.setMinimumSize(QtCore.QSize(50, 100))
        self.label.setMaximumSize(QtCore.QSize(16777215, 100))
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout_4.addWidget(self.label)
        self.horizontalLayout.addWidget(self.right_side_menu)
        self.verticalLayout.addWidget(self.main_body_frame)
        self.main_footer_frame = QtWidgets.QFrame(self.centralwidget)
        self.main_footer_frame.setMaximumSize(QtCore.QSize(16777215, 30))
        self.main_footer_frame.setStyleSheet("QFrame{\n"
"    border-top-color: rgb(0,0,0);\n"
"    background-color:  rgb(240, 187, 98);\n"
"}")
        self.main_footer_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.main_footer_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_footer_frame.setObjectName("main_footer_frame")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.main_footer_frame)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_5 = QtWidgets.QLabel(self.main_footer_frame)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_6.addWidget(self.label_5)
        self.size_grip = QtWidgets.QFrame(self.main_footer_frame)
        self.size_grip.setMinimumSize(QtCore.QSize(20, 20))
        self.size_grip.setMaximumSize(QtCore.QSize(20, 20))
        self.size_grip.setStyleSheet("background-color:  rgb(240, 187, 98);")
        self.size_grip.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.size_grip.setFrameShadow(QtWidgets.QFrame.Raised)
        self.size_grip.setObjectName("size_grip")
        self.horizontalLayout_6.addWidget(self.size_grip, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignBottom)
        self.verticalLayout.addWidget(self.main_footer_frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.home_button.setText(_translate("MainWindow", "    HOME"))
        self.control_button.setText(_translate("MainWindow", "CONTROL"))
        self.log_button.setText(_translate("MainWindow", "   History"))
        self.setting_button.setText(_translate("MainWindow", "    SETTING"))
        self.segment_img_label.setText(_translate("MainWindow", "segment img"))
        self.segment_name.setText(_translate("MainWindow", "Segment Image"))
        self.origin_name.setText(_translate("MainWindow", "Origin Image"))
        self.origin_img_label.setText(_translate("MainWindow", "origin img"))
        item = self.tableLog.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Time"))
        item = self.tableLog.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Disease"))
        item = self.tableLog.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Accuracy"))
        item = self.tableLog.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Note"))
        self.done_set.setText(_translate("MainWindow", "Done"))
        self.label_2.setText(_translate("MainWindow", "Speed"))
        self.label_3.setText(_translate("MainWindow", "Capture duration"))
        self.start_button.setText(_translate("MainWindow", "start"))
        self.auto_mode.setText(_translate("MainWindow", "Auto"))
        self.result_label.setText(_translate("MainWindow", "Result"))
        self.label_5.setText(_translate("MainWindow", "v1.0"))
import icons_rc
