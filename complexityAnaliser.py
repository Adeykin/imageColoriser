import torch
import torchsummary
import torchvision
from UNet.unet import UNet
import fastai
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from ptflops import get_model_complexity_info

inputSize = 224

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#net = UNet(n_channels=1, n_classes=2, bilinear=True)
body = create_body(torchvision.models.resnet.resnet18(), pretrained=True, n_in=1, cut=-2)
net = DynamicUnet(body, 2, (inputSize, inputSize))

#net.load_state_dict(torch.load('model_9.tar.gz', map_location=device))

#y = net(torch.zeros(1,1,224,224))
#print(y)
#quit()
torchsummary.summary(net, (1,224,224))

macs, params = get_model_complexity_info(net, (1, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))