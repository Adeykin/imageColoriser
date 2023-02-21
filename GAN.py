import torch

def normalise(x):
    return x / 127. - 1.

class GAN:
    def __init__(self, netG, netD, device, lrG=1e-3, lrD=1e-3):
        self.netG = netG
        self.netD = netD
        self.optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG)
        self.optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD)
        self.criterium = torch.nn.CrossEntropyLoss()
        self.device = device

    def trainG(self, I, abTarget):
        self.optimizerG.zero_grad()
        abActual = self.netG(I)

        y = self.netD(torch.cat((I, normalise(abActual)), axis=1))
        ones = torch.tensor([1,]*y.shape[0], dtype=torch.int64, device=self.device)
        loss = self.criterium(y[:,:,0,0], ones)
        loss.backward()
        self.optimizerG.step()
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
        actual = self.netD(torch.cat((I, normalise(ab)), axis=1))
        loss = self.criterium(actual[:,:,0,0], target)
        loss.backward()
        self.optimizerD.step()
        return loss.item()
