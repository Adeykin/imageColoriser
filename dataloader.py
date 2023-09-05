import torch
import cv2
import os
import numpy as np
import transform
from torchvision import transforms

def augResize(img, inputSize):
    minSide = min(img.shape[:2])
    scaleFactor = minSide / inputSize
    targetSize = np.ceil(np.array(img.shape[:2]) / scaleFactor).astype(int)[::-1]
    img = cv2.resize(img, targetSize)
    return img


def augCrop(img, inputSize):
    return img[:inputSize, :inputSize]


def augNormalise(img):
    return img.astype('float32') / 127. - 1.


def recursiveReader(path):
    files = []
    for name in os.listdir(path):
        fullPath = path + '/' + name
        if os.path.isdir(fullPath):
            f = recursiveReader(fullPath)
            files += f
        else:
            files.append(fullPath)
    return files


class COCOLoader:
    def __init__(self, path, transform, phase='train', size=224, device=torch.device('cpu')):
        # self.path = path
        if os.path.isfile(path):
            listFile = path
            self.path = os.path.dirname(path)
        elif os.path.isdir(path):
            listFile = path + ('/coco_zoo_256_train.txt' if phase == 'train' else '/coco_zoo_159_test.txt')
            self.path = path
        else:
            print('[COCOLoader] error')
            quit()
        self.transform = transform

        self.images = [x.strip('\n') for x in open(listFile, 'r').readlines()]
        self.phase = phase
        self.inputSize = size
        self.device = device
        print("Read {} images".format(len(self.images)))

    def __getitem__(self, index):
        img = cv2.imread(self.path + '/' + self.images[index])[:, :, ::-1]
        tensor = self.transform(img).to(device=self.device)

        if self.phase == 'train':
            return tensor[0,:,:], tensor[1:,:,:]
        else:
            return tensor[0, :, :], tensor[1:, :, :], img, self.path + '/' + self.images[index]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    image_transform = transforms.Compose([
        transform.RandomResizeMinSide(224,224),
        transform.EdgeCrop(224),
        transform.Lab(),
        transforms.ToTensor(),
        transforms.Normalize(127,127)
    ])

    cocoLoader = COCOLoader('/home/adeykin/projects/coloriser/coloriser_own/datasets/coco/coco_zoo', image_transform)
    print('hello')

    train_loader = torch.utils.data.DataLoader(
        cocoLoader,
        batch_size=8,
        shuffle=True,
        pin_memory=True,  # ???
        drop_last=True,  # ???
    )

    for a in train_loader:
        print('a')

    img = cocoLoader[0]

    img = cv2.imread('/home/adeykin/projects/visionlabs/WIDER_val/images/46--Jockey/46_Jockey_Jockey_46_823.jpg')
    img = augResize(img)
    img = augCrop(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

