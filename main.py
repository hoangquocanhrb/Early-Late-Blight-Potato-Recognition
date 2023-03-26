import pyrebase
import numpy as np 

from utils import get_firebase_data, config, tools

from PyQt5.QtWidgets import QApplication

import sys 

sys.path.append('Segment')
sys.path.append('Classify')
sys.path.append('UI')
sys.path.append('Firebase')

import torch 
from torchvision import transforms

from Classify.model.VGG16 import VGG16
from Segment.model.unet_model import UNet
from Segment.U_2_Net.model import U2NET
from UI.mainWindow import MainWindow

if __name__ == "__main__":
    firebase_storage = pyrebase.initialize_app(config.FIREBASE_CONFIG)
    storage = firebase_storage.storage()
    database = firebase_storage.database()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classify_model = VGG16(num_classes=3)
    classify_model.load_state_dict(torch.load(config.CLASSIFY_MODEL_NAME, map_location=device))
    segment_model = UNet(n_channels=3, n_classes=1)
    segment_model.outc = torch.nn.Conv2d(64, 1, kernel_size=1) 
    
    segment_model.load_state_dict(torch.load(config.SEGMENT_MODEL_NAME, map_location=device))
    # segment_model = U2NET(3,1)
    # segment_model.load_state_dict(torch.load(config.SEGMENT_MODEL_NAME, map_location=device))
    app = QApplication(sys.argv)
    
    main_win = MainWindow(storage, database, segment_model=segment_model, classify_model=classify_model, device=device)
    main_win.show()
    sys.exit(app.exec())
