from UNet.unet import UNet, UNet2, UNetHalf, UNet2Half, PatchDiscriminator

import torch
import dataloader
import time
from tensorboardX import SummaryWriter

from UNet.unet import UNet, UNet2
import GAN
import cv2
import os

def train(epochsNumber = 600, learningRate = 1e-4, batchSize = 16):
    inputSize = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netG = UNet2(n_channels=1, n_classes=2)
    netD = UNet2Half(n_channels=3, n_classes=2)
    netG.to(device=device)
    netD.to(device=device)

    gan = GAN.GAN(netG, netD, device, lrG=learningRate, lrD=learningRate, patchLoss=True)

    outputPath = 'logs/cocozoo_unet2_patch_cos_lr{}_batch{}_epoch{}'.format(learningRate, batchSize, epochsNumber)

    writer = SummaryWriter(outputPath)

    if not os.path.exists(outputPath + '/vis'):
        os.mkdir(outputPath + '/vis')

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(gan.optimizerG, T_max=epochsNumber)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(gan.optimizerD, T_max=epochsNumber)

    datasetPath = '/home/adeykin/projects/coloriser/coloriser_own/datasets/coco/coco_zoo'
    cocoTrainLoader = dataloader.COCOLoader(datasetPath, phase='train', size=inputSize, device=device)
    cocoTestLoader = dataloader.COCOLoader(datasetPath, phase='test', size=inputSize, device=device)
    #dataSubset = torch.utils.data.Subset(widerFaceLoader, range(datasetSize))
    train_loader = torch.utils.data.DataLoader(
        cocoTrainLoader,
        batch_size=batchSize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        cocoTestLoader,
        batch_size=batchSize,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    trainIter = 0
    testIter = 0
    for epoch in range(epochsNumber):
        netG.train()
        netD.train()
        for I, ab in train_loader:
            #I = I.to(device=device)  # TODO: move it to dataloader
            #ab = ab.to(device=device)
            lossD = gan.trainD(I, ab)
            lossG = gan.trainG(I, ab)
            print("{}) {} {}".format(trainIter, lossD, lossG))
            writer.add_scalar('Train/LR', schedulerD.get_lr(), trainIter)
            writer.add_scalar('Train/LossIterG', lossG, trainIter)
            writer.add_scalar('Train/LossIterD', lossD, trainIter)
            trainIter += 1
        netG.eval()
        netD.eval()
        for iter, (I, ab) in enumerate(test_loader):
            #I = I.to(device=device)
            #ab = ab.to(device=device)
            lossD = gan.testD(I, ab)
            lossG = gan.testG(I, ab)
            writer.add_scalar('Test/LossIterG', lossG, testIter)
            writer.add_scalar('Test/LossIterD', lossD, testIter)
            testIter += 1

            if epoch % 10 == 0:
                abActual = netG(I)
                img = GAN.visualise(I, abActual, 0)
                #plt.imshow(img)
                #plt.show()
                print("Test/LossIterG={}; 'Test/LossIterD={}".format(lossG, lossD))
                cv2.imwrite(outputPath + '/vis/' + str(iter) + '_' + str(epoch) + '.png', img[:, :, ::-1])

        schedulerD.step()
        schedulerG.step()

if __name__ == '__main__':
    train(epochsNumber = 600, learningRate = 1e-4, batchSize = 4)