import torch
import dataloader

from UNet.unet import UNet, UNetHalf, PatchDiscriminator
import GAN

inputSize = 64
epochsNumber = 5
batchSize = 4
learningRateG = 1e-5
learningRateD = 1e-5

netG = UNet(n_channels=1, n_classes=2, bilinear=True)
netD = UNetHalf(n_channels=3, n_classes=2, bilinear=True, inputSize=inputSize)

datasetSize = batchSize
widerFaceLoader = dataloader.WiderFaceLoader('/home/adeykin/projects/visionlabs/WIDER_val/images', phase='train', size=inputSize)
dataSubset = torch.utils.data.Subset(widerFaceLoader, range(datasetSize))
train_loader = torch.utils.data.DataLoader(
    dataSubset,
    batch_size=batchSize,
    shuffle=True,
    pin_memory=True, drop_last=True,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG.to(device=device)
netD.to(device=device)

gan = GAN.GAN(netG, netD, device, lrG=learningRateG, lrD=learningRateD)

for epoch in range(epochsNumber):
    netG.train()
    netD.train()
    epochLoss = 0
    for I, ab in train_loader:
        I = I.to(device=device)  #TODO: move it to dataloader
        ab = ab.to(device=device)

        lossD = gan.trainD(I, ab)
        lossG = gan.trainG(I, ab)
        print("{} {}".format(lossD, lossG))
    #print('Epoch {}: loss={}'.format(epoch, epochLoss))

