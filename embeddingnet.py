from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms


class Vgg_Net(nn.Module):
    def __init__(self):
        super(Vgg_Net, self).__init__()
        model = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(*list(model.features.children()))
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1024)

    def forward(self, x):
        vgg_op = self.vgg(x)
        vgg_op = vgg_op.view(-1, 512*7*7)
        fc1 = self.fc1(vgg_op)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        return fc3