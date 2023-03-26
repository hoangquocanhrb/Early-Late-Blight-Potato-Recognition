import pyrebase
import os

config = {

  'apiKey': "AIzaSyDoZr1AUBui1PFsK3PGEcrH3nD--isFGAs",

  'authDomain': "test-pi4.firebaseapp.com",

  'databaseURL': "https://test-pi4-default-rtdb.firebaseio.com",

  'projectId': "test-pi4",

  'storageBucket': "test-pi4.appspot.com",

  'messagingSenderId': "565845853592",

  'appId': "1:565845853592:web:3f086d24f35102968597ed",

  'measurementId': "G-9TCQH7TZF0",

  'serviceAccount': "test_pi4.json"
}


firebase_storage = pyrebase.initialize_app(config)
storage = firebase_storage.storage()

files = storage.list_files()

# print(files)
for f in files:
  # f.download_to_filename('img/' + f.name)
  print(type(f.name))