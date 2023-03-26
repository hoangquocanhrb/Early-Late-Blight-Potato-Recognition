import argparse

import pyrebase
import os
import cv2 
import numpy as np 
import random

config = {
    'apiKey': "AIzaSyDWg4Bccz0FoyqpRCAB9UMMCsjwQ3yxeSk",
  'databaseURL': "https://thesis-a13d9-default-rtdb.firebaseio.com",
  'authDomain': "thesis-a13d9.firebaseapp.com",

  'projectId': "thesis-a13d9",

  'storageBucket': "thesis-a13d9.appspot.com",

  'messagingSenderId': "664589897175",

  'appId': "1:664589897175:web:ae6824baa21aa89fd70f92",

  'measurementId': "G-WNJXXWFVK2",

  'serviceAccount': "Firebase/thesis_firebase.json"
    }


firebase_storage = pyrebase.initialize_app(config)
storage = firebase_storage.storage()
database = firebase_storage.database()

# back = cv2.imread('back_ground.JPG')
# back = cv2.resize(back, (224,224))
# cv2.imwrite('back_ground.JPG', back)
im_path = '/home/hqanh/Potato/Dataset/image_from_internet/images/early2.jpeg'
storage.child("leaf.png").put(im_path)

database.child('Data').update({'Image': 'sent'})
database.child('Control_motion').update({'Status': 'capture'})
