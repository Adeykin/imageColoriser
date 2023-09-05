import torch
import cv2
import os
import numpy as np


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
    def __init__(self, path, phase='train', size=224, device=torch.device('cpu')):
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

        self.images = [x.strip('\n') for x in open(listFile, 'r').readlines()]
        self.phase = phase
        self.inputSize = size
        self.device = device
        print("Read {} images".format(len(self.images)))

    def __getitem__(self, index):
        # Augmentation:
        #   Resize to 224 by min side
        #   ~Crop with random slide
        img = cv2.imread(self.path + '/' + self.images[index])[:, :, ::-1]
        img = augResize(img, self.inputSize)
        img = augCrop(img, self.inputSize)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        X = img[:, :, 0][:, :, None]
        Y = img[:, :, 1:].astype('float32')
        X = augNormalise(X)
        Y = augNormalise(Y)

        if self.phase == 'train':
            return torch.from_numpy(X).permute(2, 0, 1).to(device=self.device), \
                   torch.from_numpy(Y).permute(2, 0, 1).to(device=self.device)
        else:
            return torch.from_numpy(X).permute(2, 0, 1).to(device=self.device), \
                   torch.from_numpy(Y).permute(2, 0, 1).to(device=self.device), \
                   img, \
                   self.path + '/' + self.images[index]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    cocoLoader = COCOLoader('/home/adeykin/projects/visionlabs/WIDER_val/images')
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

