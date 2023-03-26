import argparse

import pyrebase
import os
import cv2 
import numpy as np 
import random

config = {

  'apiKey': "AIzaSyDoZr1AUBui1PFsK3PGEcrH3nD--isFGAs",

  'authDomain': "test-pi4.firebaseapp.com",

  'databaseURL': "https://test-pi4-default-rtdb.firebaseio.com",

  'projectId': "transfer_pi4",

  'storageBucket': "test-pi4.appspot.com",

  'messagingSenderId': "565845853592",

  'appId': "1:565845853592:web:3f086d24f35102968597ed",

  'measurementId': "G-9TCQH7TZF0",

  'serviceAccount': "transfer_pi4.json"
}


firebase_storage = pyrebase.initialize_app(config)
storage = firebase_storage.storage()
database = firebase_storage.database()

# back = cv2.imread('back_ground.JPG')
# back = cv2.resize(back, (224,224))
# cv2.imwrite('back_ground.JPG', back)

storage.child("hello.png").put('../back_ground.JPG')

database.child('Data').update({'Image': 'sent'})
