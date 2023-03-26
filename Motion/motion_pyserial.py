import serial
import time
import pyrebase
from picamera import PiCamera
FIREBASE_CONFIG = {
    'apiKey': "AIzaSyDWg4Bccz0FoyqpRCAB9UMMCsjwQ3yxeSk",
  'databaseURL': "https://thesis-a13d9-default-rtdb.firebaseio.com",
  'authDomain': "thesis-a13d9.firebaseapp.com",

  'projectId': "thesis-a13d9",

  'storageBucket': "thesis-a13d9.appspot.com",

  'messagingSenderId': "664589897175",

  'appId': "1:664589897175:web:ae6824baa21aa89fd70f92",

  'measurementId': "G-WNJXXWFVK2",

  'serviceAccount': "../Firebase/thesis_firebase.json"
    }

firebase_storage = pyrebase.initialize_app(FIREBASE_CONFIG)
database = firebase_storage.database()
storage = firebase_storage.storage()

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)
def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

status = ''
camera = PiCamera()
camera.resolution = (224,224)

robot_speed = 0
cap_duration = 3

old_time = time.time()
sleep_time = 1

check_di_thang = False

while True:
    # num = input("Enter a number: ") # Taking input from user
    # value = write_read(num)

    temp_time = time.time()

    status = database.child("Control_motion").get().val()["Status"]
    confirm_setting = database.child('Data').get().val()['Confirm_setting']
    
    if confirm_setting == 'sent':
        robot_speed = database.child("Robot_setting").get().val()['robot_speed']
        cap_duration = database.child('Robot_setting').get().val()['capture_duration']
        robot_speed = int(robot_speed)
        cap_duration = int(cap_duration)
        database.child('Data').update({'Confirm_setting': 'received'})
    
    print('Robot speed: ', robot_speed)
    print('Cap duration: ', cap_duration)
    print('------------------------')
    firebase_time = time.time() - temp_time

    # print(time.time() - old_time)
    confirm_img = database.child("Data").get().val()['Image']
    if status == 'auto':

        if time.time() - old_time >= cap_duration + firebase_time:
            value = write_read('stop-' + str(robot_speed))
            print('auto capture')
            time.sleep(sleep_time)
            camera.capture('leaf.png')
            storage.child("leaf.png").put('leaf.png')
            database.child('Data').update({'Image': 'sent'})
            old_time = time.time()
            check_di_thang = True
        else:
            if check_di_thang == True:
                value = write_read('di_thang-' + str(robot_speed))
                print('auto di_thang')
                check_di_thang = False

    elif status != 'received' and status != 'auto':
        if status == 'capture' and confirm_img != 'sent':
            status = 'stop'
            value = write_read(status + '-' + str(robot_speed))
            database.child('Control_motion').update({'Status': 'received'})
            time.sleep(sleep_time)
            camera.capture('leaf.png')
            storage.child("leaf.png").put('leaf.png')
            database.child('Data').update({'Image': 'sent'})
        else:
            value = write_read(status + '-' + str(robot_speed))
            database.child('Control_motion').update({'Status': 'received'})
        print(status)