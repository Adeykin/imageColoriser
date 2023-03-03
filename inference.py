import torch
import dataloader
import cv2
import numpy as np
import os, sys
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
import torchvision


from UNet.unet import UNet

inputSize = 224
modelName = 'model_G_unet_OVERFITcow'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = UNet(n_channels=1, n_classes=2, bilinear=True)
#body = create_body(torchvision.models.resnet.resnet18(), pretrained=False, n_in=1, cut=-2)
#net = DynamicUnet(body, 2, (inputSize, inputSize))

resnet = torchvision.models.resnet.resnet18()

net.load_state_dict(torch.load('models/' + modelName + '.tar.gz', map_location=device))

outputPath = 'outputs/' + modelName
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

datasetSize = 32
#widerFaceLoader = dataloader.WiderFaceLoader('/home/adeykin/projects/visionlabs/WIDER_val/images', phase='test')
widerFaceLoader = dataloader.WiderFaceLoader('datasets/coco/coco_cow', phase='test')
dataSubset = torch.utils.data.Subset(widerFaceLoader, range(datasetSize))

for data,target,img,path in dataSubset:
    data = data.to(device=device)
    target = data.to(device=device)

    y_actual = net(data[None])
    svImg = y_actual.detach().numpy()[0, :].clip(0, 255).astype('uint8').transpose([1,2,0])
    imgConsruct = img.copy()
    imgConsruct[:, :, 1:] = svImg
    imgConsructRgb = cv2.cvtColor(imgConsruct, cv2.COLOR_Lab2BGR)
    imgRgb = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    imgGray = cv2.cvtColor(imgRgb, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)

    blank = np.zeros((inputSize,inputSize*3,3),dtype='uint8')
    blank[:, inputSize * 0:inputSize * 1, :] = imgGray
    blank[:, inputSize * 1:inputSize * 2, :] = imgConsructRgb
    blank[:, inputSize * 2:inputSize * 3,:] = imgRgb

    cv2.imwrite(outputPath + '/' + os.path.basename(path), blank)

    print('Done')


