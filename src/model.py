from torchvision import models
from torch import nn

class FeaturesNet(nn.Module):
    ### 最后一层分类器，提取特征
    def __init__(self, pretrained_model):
        super(FeaturesNet, self).__init__()
        for name, module in list(pretrained_model._modules.items())[:-1]:
            self.add_module(name, module)

    def forward(self, x):
        self.eval()
        for name, module in list(self._modules.items()):
            x = module(x)
        return x

class DEMNet(nn.Module):
    def __init__(self, FeaturesNet, input_size, output_size):
        super(DEMNet, self).__init__()
        for param in FeaturesNet.parameters():
            param.requires_grad = False

        self.FeaturesNet = FeaturesNet
        self.transformer = nn.Sequential(*[
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        ])

    def forward(self, x):
        self.FeaturesNet.eval()
        x = self.FeaturesNet(x)
        x = x.view(x.size(0), -1) # 展平多维的卷积图进全连接
        x = self.transformer(x)
        return x


class ClassificationModelResenet18(nn.Module):
    def __init__(self, pretrained=False, NUM_CLASSES=190):
        super(ClassificationModelResenet18, self).__init__()
        resnet_model = models.resnet18(pretrained=pretrained)
        for name, module in list(resnet_model._modules.items())[:-2]:
            self.add_module(name, module)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        for name, module in list(self._modules.items())[:-1]:
            x = module(x)
        x = x.view(x.size(0), -1) # 展平多维的卷积图进全连接
        x = self.fc(x)
        return x