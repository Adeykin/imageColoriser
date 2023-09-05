import numpy as np
import random
import cv2
import torch
from PIL import Image
import collections


class RandomResizeMinSide(object):
    def __init__(self, minMinSize, maxMinSize):
        self.minMinSize = minMinSize
        self.maxMinSize = maxMinSize

    def __call__(self, img):
        inputSize = random.randrange(self.minMinSize, self.maxMinSize+1)
        minSide = min(img.shape[:2])
        scaleFactor = minSide / inputSize
        targetSize = np.ceil(np.array(img.shape[:2]) / scaleFactor).astype(int)[::-1]
        img = cv2.resize(img, targetSize)
        return img

class EdgeCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        assert(img.shape[0] >= self.size and img.shape[1] >= self.size )
        return img[:self.size, :self.size]

class RandomCrop(object):
    def __init__(self, size):
        self.size= size

    def __call__(self, img):
        assert (img.shape[0] >= self.size and img.shape[1] >= self.size)
        xoffset = random.randint(img.shape[1] - self.size)
        yoffset = random.randint(img.shape[0] - self.size)
        return img[yoffset:(yoffset+self.size), xoffset:(xoffset+self.size)]

class Lab(object):
    def __call__(self, img):
        return  cv2.cvtColor(img, cv2.COLOR_RGB2Lab)