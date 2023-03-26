import os 
import pyrebase
from PIL import Image 
import numpy as np 
# import cv2 

class firebase_receiver:
    def __init__(self, storage=None, database=None):
        self.storage = storage
        self.database = database
        
    #Get image from firebase and then send 'received' to confirm
    def getData(self):
        origin_img = None
        if self.storage != None:
            files = self.storage.list_files()
            for f in files:
                f.download_to_filename(f.name)
                origin_img = Image.open(f.name)
                #for image from pi
                origin_img = np.array(origin_img)
                if origin_img.shape[2] > 3:
                    origin_img = np.delete(origin_img, 3, axis=2)
                origin_img = Image.fromarray(origin_img)
                origin_img = origin_img.resize((256, 256))

        if self.database != None:
            self.database.child('Data').update({'Image': 'received'})
        
        return origin_img

    def send_control_signal(self, signal):
        self.database.child('Control_motion').update({'Status': signal})

    def send_robot_setting(self, setting):
        self.database.child('Robot_setting').update(setting)
        self.database.child('Data').update({'Confirm_setting': 'sent'})
    