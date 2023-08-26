import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
batch_size = 96


path = '/home/rayhan/Resnet/R50/data/'

# Device
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Device:', DEVICE)

learning_rate = 0.00005
num_epochs = 30
num_classes = 4
IMG_SIZE = (32, 32)  # resize image

transforms = transforms.Compose({
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
    # transforms.Normalize(IMG_MEAN, IMG_STD)
})


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        with open(root + datatxt, 'r') as f:
            imgs = []
            for line in f:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
            self.root = root
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        f, label = self.imgs[index]
        img = Image.open(self.root + f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


train_data = MyDataset(path + 'train_data/', 'train_label.txt', transform=transform)
test_data = MyDataset(path + 'test_data/', 'test_label.txt', transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=1)

        self.layer1 = self._make_layer(ResBlock,
                                       layer_list[0],
                                       planes=64)
        self.layer2 = self._make_layer(ResBlock,
                                       layer_list[1],
                                       planes=128,
                                       stride=2)
        self.layer3 = self._make_layer(ResBlock,
                                       layer_list[2],
                                       planes=256,
                                       stride=2)
        self.layer4 = self._make_layer(ResBlock,
                                       layer_list[3],
                                       planes=512,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)

model = ResNet50(num_classes, channels=3)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
    start = time.perf_counter()
    # model.train()
    running_loss = 0.0
    correct_pred = 0
    for index, data in enumerate(train_loader):
        image, label = data
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        y_pred = model(image)

        _, pred = torch.max(y_pred, 1)
        correct_pred += (pred == label).sum()

        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
    end = time.perf_counter()
    print('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%'.
          format(epoch + 1, num_epochs, running_loss / (index + 1),
                 correct_pred.item() / (batch_size * (index + 1)) * 100))
    print('Time: {:.2f}s'.format(end - start))
print('Finished training!')






test_loss = 0.0
correct_pred = 0
model.eval()
for _, data in enumerate(test_loader):
    image, label = data
    image = image.to(DEVICE)
    lable = label.to(DEVICE)
    y_pred = model(image)

    _, pred = torch.max(y_pred, 1)
    correct_pred += (pred == label).sum()

    loss = criterion(y_pred, label)
    test_loss += float(loss.item())
print('Test loss: {:.4f}\tTest accuracy: {:.2f}%'.format(test_loss / 12, correct_pred.item() / 120 * 100))
