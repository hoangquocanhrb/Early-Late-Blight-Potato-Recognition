import torch 
import torchvision 
from torchvision import transforms
from utils import get_firebase_data, config, tools
from PIL import Image
from Classify.model.VGG16 import VGG16

import numpy as np 
import cv2 

import os 

def classify_predict(net, img, device, transform):
    img = transform(img)
    img = img.to(device)
    with torch.no_grad():
        output = net(img)
    return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
classify_model = VGG16(num_classes=3)
classify_model.load_state_dict(torch.load(config.CLASSIFY_MODEL_NAME, map_location=device))
classify_model.eval()

img_path = '/home/hqanh/Potato/Dataset/Classify/test/'
dis = os.listdir(img_path)
print(dis)
confu_matrix = np.zeros((3,3))
i = 0
for d in dis:
    img_list = os.listdir(img_path + d)
    for im in img_list:
        img = Image.open(img_path + d + '/' + im)

        out = classify_predict(classify_model, img, device, data_transforms)
        _, pred = torch.max(out, 0)
        confu_matrix[i][pred] += 1
    i += 1

print(confu_matrix)
