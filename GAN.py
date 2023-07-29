import torch
import cv2

def denorm(x):
    return (x+1)*127

def normalise(x):
    return x / 127. - 1.

def visualise(I, ab, index):
    I = denorm(I).type(torch.uint8)
    #ab = ab.clip(0, 255).type(torch.uint8)
    ab = denorm(ab).type(torch.uint8)
    img = torch.cat((I, ab), dim=1)[index,:]
    img  = img.detach().cpu().numpy().transpose([1, 2, 0])
    imgRgb = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    return imgRgb

class GAN:
    def __init__(self, netG, netD, device, lrG=1e-3, lrD=1e-3, patchLoss=False):
        self.netG = netG
        self.netD = netD
        self.optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG)
        self.optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD)
        #self.criterium = torch.nn.CrossEntropyLoss()
        self.criterium2 = torch.nn.BCELoss()
        #self.criterium3 = torch.nn.BCEWithLogitsLoss()
        self.device = device
        self.patchLoss = patchLoss

    def trainG(self, I, abTarget):
        self.optimizerG.zero_grad()
        abActual = self.netG(I)

        y = self.netD(torch.cat((I, abActual), axis=1))
        ones = torch.tensor([1,]*y.shape[0], dtype=torch.int64, device=self.device)
        if self.patchLoss:
            target = torch.dstack([ones, ] * y.shape[1])[0, :]
            loss = self.criterium2(y, target.type(torch.float32))
        else:
            loss = self.criterium2(y[:,0,0,0], ones.type(torch.float32))

        loss.backward()
        self.optimizerG.step()
        return loss.item()

    def testG(self, I, abTarget):
        with torch.no_grad():
            abActual = self.netG(I)

            y = self.netD(torch.cat((I, abActual), axis=1))
            ones = torch.tensor([1, ] * y.shape[0], dtype=torch.int64, device=self.device)
            if self.patchLoss:
                target = torch.dstack([ones, ] * y.shape[1])[0, :]
                loss = self.criterium2(y, target.type(torch.float32))
            else:
                loss = self.criterium2(y[:,0,0,0], ones.type(torch.float32))

            return loss.item()

    def trainD(self, I, ab):
        batchSize = I.shape[0]
        halfSize = batchSize//2
        assert (batchSize % 2 == 0)

        self.optimizerD.zero_grad()
        #I_true = I[:halfSize, :]
        ab_true = ab[:halfSize, :]
        I_false = I[halfSize:, :]
        ab_false = self.netG(I_false)

        ab = torch.cat((ab_true, ab_false), axis=0)
        target = torch.tensor( [1,]*halfSize + [0,]*halfSize, dtype=torch.int64, device=self.device)
        actual = self.netD(torch.cat((I, ab), axis=1))
        if self.patchLoss:
            target = torch.dstack([target, ] * actual.shape[1])[0, :]
            loss = self.criterium2(actual, target.type(torch.float32))
        else:
            loss = self.criterium2(actual[:,0,0,0], target.type(torch.float32))

        loss.backward()
        self.optimizerD.step()
        return loss.item()

    def testD(self, I, ab):
        with torch.no_grad():
            batchSize = I.shape[0]
            halfSize = batchSize//2
            assert (batchSize % 2 == 0)

            ab_true = ab[:halfSize, :]
            I_false = I[halfSize:, :]
            ab_false = self.netG(I_false)

            ab = torch.cat((ab_true, ab_false), axis=0)
            target = torch.tensor( [1,]*halfSize + [0,]*halfSize, dtype=torch.int64, device=self.device)
            actual = self.netD(torch.cat((I, ab), axis=1))
            if self.patchLoss:
                target = torch.dstack([target, ] * actual.shape[1])[0, :]
                loss = self.criterium2(actual, target.type(torch.float32))
            else:
                loss = self.criterium2(actual[:,0,0,0], target.type(torch.float32))

            return loss.item()
