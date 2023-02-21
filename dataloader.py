import torch
import cv2
import os
import numpy as np

def augResize(img, inputSize):
    minSide = min(img.shape[:2])
    scaleFactor = minSide / inputSize
    #targetSize = int(np.ceil(img.shape[1]/scaleFactor)), int(np.ceil(img.shape[0]/scaleFactor))
    targetSize = np.ceil(np.array(img.shape[:2])/scaleFactor).astype(int)[::-1]
    img = cv2.resize(img, targetSize)
    return img

def augCrop(img, inputSize):
    return img[:inputSize,:inputSize]

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

class WiderFaceLoader:
    def __init__(self, path, phase='train', size=224):
        self.filePathes = recursiveReader(path)
        self.phase = phase
        self.inputSize = size
        print("Read {} images".format(len(self.filePathes)))

    def __getitem__(self, index):
        #Augmentation:
        #   Resize to 224 by min side
        #   ~Crop with random slide
        path = self.filePathes[index]
        img = cv2.imread(path)
        img = augResize(img, self.inputSize)
        img = augCrop(img, self.inputSize)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        X = img[:,:,0][:,:,None]
        Y = img[:,:,1:].astype('float32')
        X = augNormalise(X)

        if self.phase == 'train':
            return torch.from_numpy(X).permute(2, 0, 1), torch.from_numpy(Y).permute(2, 0, 1)
        else:
            return torch.from_numpy(X).permute(2, 0, 1), torch.from_numpy(Y).permute(2, 0, 1), img, self.filePathes[index]

    def __len__(self):
        return len(self.filePathes)


if __name__ == '__main__':
    widerFaceLoader = WiderFaceLoader('/home/adeykin/projects/visionlabs/WIDER_val/images')
    print('hello')

    train_loader = torch.utils.data.DataLoader(
        widerFaceLoader,
        batch_size=8,
        shuffle=True,
        pin_memory=True,  # ???
        drop_last=True,  # ???
    )

    for a in train_loader:
        print('a')

    img = widerFaceLoader[0]

    img = cv2.imread('/home/adeykin/projects/visionlabs/WIDER_val/images/46--Jockey/46_Jockey_Jockey_46_823.jpg')
    img = augResize(img)
    img = augCrop(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

