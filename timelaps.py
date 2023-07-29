import cv2
import os

inputDir = '/home/adeykin/projects/coloriser/timelaps/OVERFITcow_unet_nonBilinear'
outputVideoPath = 'out.avi'
out = cv2.VideoWriter(outputVideoPath, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (224,224))


def readImages(path, index):
    i = 0
    while True:
        currentImgName = path + '/' + str(index) + '_' + str(i) + '.png'
        if not os.path.exists(currentImgName):
            return None
        yield cv2.imread(currentImgName)
        i += 10


print('hello')
for i in range(5):
    for img in readImages(inputDir, i):
        out.write(img)
out.release()