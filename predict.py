import cv2 
import numpy as np
import sys 
from PIL import Image 
import os 
import time 

sys.path.insert(0, 'Segment')
sys.path.insert(0, 'Classify')

import torch 
import torch.nn.functional as F
import torchvision 
from torchvision import transforms

from Segment.model.unet_model import UNet
from Classify.model.VGG16 import VGG16

def segment_preprocess(pil_img):
        w, h = pil_img.size
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

        return img_ndarray

def classify_preprocess(img):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf(img)

def mask_to_image(mask):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def segment_predict(net, img, device, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(segment_preprocess(img))
    img = img.unsqueeze(0)
    img = img.to(device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]
        
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_shape[0],resize_shape[1])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
    
    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

def classify_predict(net, img, device):
    net.eval()
    img = classify_preprocess(img)
    img = img.to(device)
    output = net(img)
    return output

if __name__ == '__main__':

    diseases = {0: 'Early blight', 1: 'Late blight', 2: 'Healthy'}
    resize_shape = (256,256)

    back_ground = cv2.imread('back_ground.JPG')
    back_ground = cv2.resize(back_ground, (resize_shape))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    segment_net = UNet(n_channels=3, n_classes=1)
    # segment_net.outc = torch.nn.Conv2d(64, 1, kernel_size=1)
    segment_net.load_state_dict(torch.load('/home/hqanh/Potato/weights/weightscheckpoint_segment.pth', map_location=device))
    
    classify_net = VGG16(num_classes=3)
    classify_net.load_state_dict(torch.load('/home/hqanh/Potato/weights/VGG16_thesis.pth', map_location=device))

    img_path = '/home/hqanh/Potato/Dataset/real_img/'
    list_img = os.listdir(img_path)
    
    error_count = 0
    for img in list_img:
        
        origin_img = cv2.imread(img_path + img)
        
        origin_img = cv2.resize(origin_img, resize_shape)
        
        # img = Image.open(img_path + img)
        img = Image.fromarray(origin_img)
        mask = segment_predict(segment_net, img, device)
        pred = np.array(mask_to_image(mask))
        
        segment_img = origin_img.copy()
        segment_img[np.where(pred != 255)] = [0,0,0]
        cv2.imshow('', segment_img)
        cv2.waitKey()
        segment_img[np.where((segment_img == [0,0,0]).all(axis=2))] = back_ground[np.where((segment_img == [0,0,0]).all(axis=2))]

        segment_img = Image.fromarray(segment_img)

        output = classify_predict(classify_net, segment_img, device)
        print(output)
        _, pred = torch.max(output, 0)

        disease = diseases[pred.item()]
        print(disease)

    print(error_count)