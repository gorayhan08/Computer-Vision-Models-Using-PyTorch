import time
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

path = '/home/rayhan/Successful/vgg16-pytorch-master/data/'

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

# hyperparameter
# random_seed = 1
learning_rate = 0.00005
num_epochs = 1
batch_size = 8

num_classes = 4
IMG_SIZE = (32, 32)  # resize image
# IMG_MEAN = [0.485, 0.456, 0.406]# IMG_STD = [0.229, 0.224, 0.225]

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


train_data = MyDataset(path + 'train_data/', 'train_label.txt', transform=transforms)
test_data = MyDataset(path + 'test_data/', 'test_label.txt', transform=transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

"""
vgg16
"""


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)

    return model


# torch.manual_seed(random_seed)
model = AlexNet(num_classes=num_classes)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

"""
train
"""
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

"""
test
"""
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
