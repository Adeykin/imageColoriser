import torch
import dataloader

from UNet.unet import UNet

inputSize = 224 #TODO make it one
epochsNumber = 5
batchSize = 1
learningRate = 1e-5

net = UNet(n_channels=1, n_classes=2, bilinear=True)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-5, weight_decay=1e-8, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score



widerFaceLoader = dataloader.WiderFaceLoader('/home/adeykin/projects/visionlabs/WIDER_val/images')
train_loader = torch.utils.data.DataLoader(
    widerFaceLoader,
    batch_size=batchSize,
    shuffle=True,
    pin_memory=True, drop_last=True,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net.to(device=device)
for epoch in range(epochsNumber):
    net.train()
    optimizer.zero_grad()
    epochLoss = 0
    for data,target in train_loader:
        data = data.to(device=device)
        target = data.to(device=device)

        y_actual = net(data)

        #loss = criterion(masks_pred, true_masks) + dice_loss(
        #    F.softmax(masks_pred, dim=1).float(),
        #    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
        #    multiclass=True)

        loss = criterion(y_actual, target)

        loss.backward()
        optimizer.step()

        epochLoss += loss.item()
        print(loss.item())
    print('Epoch {}: loss={}'.format(epoch, epochLoss))

