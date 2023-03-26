import torch
import numpy as np 
from PIL import Image, ImageFilter
import torch.nn.functional as F
from torchvision import transforms
import copy 
from torch.autograd import Variable

def mask_to_image(mask):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

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

def segment_predict(img, net, device, out_threshold=0.5):
    
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
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
    
    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def segment_predict_u2net(img, net, device, out_threshold=0.5):
    image = np.array(img)
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image/np.max(image)

    tmpImg[:,:,0] = (image[:,:,2]-0.406)/0.225
    tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
    tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = tmpImg[np.newaxis,:,:,:]
    tmpImg = torch.from_numpy(tmpImg)

    tmpImg = tmpImg.type(torch.FloatTensor)

    if torch.cuda.is_available():
        tmpImg = Variable(tmpImg.cuda())
    else:
        tmpImg = Variable(tmpImg)

    d1,d2,d3,d4,d5,d6,d7= net(tmpImg)
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    pred = pred.cpu().detach().numpy()
    return pred[0] 

def classify_predict(net, img, device):
    img = classify_preprocess(img)
    img = img.to(device)
    output = net(img)
    return output

def predict(segment_net, classify_net, pil_img, back_ground, device, segment_thr=0.6):
    origin_img = np.array(pil_img).copy()
    mask = segment_predict(pil_img, segment_net, device, segment_thr)
    mask = mask_to_image(mask)
    mask.save('output_test/out_real/early2.jpg')
    pred = np.array(mask)
    
    # pred = Image.fromarray(pred)
    # pred = pred.filter(ImageFilter.MinFilter(3))
    # pred = np.asarray(pred)
    # mask = segment_predict_u2net(pil_img, segment_net, device, segment_thr)
    # pred = np.array(mask_to_image(mask))
    
    segment_img = origin_img.copy()
    segment_img = np.array(segment_img)
    
    segment_img[np.where(pred == 0)] = [0,0,0]
    
    output_segment = segment_img.copy()

    segment_img[np.where((segment_img == [0,0,0]).all(axis=2))] = back_ground[np.where((segment_img == [0,0,0]).all(axis=2))]

    segment_img = Image.fromarray(segment_img)

    output = classify_predict(classify_net, segment_img, device)

    acc, pred = torch.max(output, 0)
    
    m = torch.nn.Softmax(dim=0)
    probs = m(output)
    
    return output_segment, pred.item(), probs