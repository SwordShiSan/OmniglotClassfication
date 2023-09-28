from torch import nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 卷积后，池化后尺寸计算公式：(图像尺寸-卷积核尺寸+2*填充值)/步长+1, (图像尺寸-池化窗尺寸+2*填充值)/步长+1
        # 图像1*105*105
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4086),
            nn.Linear(4086, 1623)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # 卷积后，池化后尺寸计算公式：(图像尺寸-卷积核尺寸+2*填充值)/步长+1, (图像尺寸-池化窗尺寸+2*填充值)/步长+1
        # 图像1*105*105
        model_conv = models.resnet50(pretrained=True)
        # resnet输入的是rgb，故改变第一层的输入通道为1
        model_conv.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        # 去除最后一个fc layer，使用自己的分类数1623
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.model = model_conv
        self.fc = nn.Linear(2048, 1623)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 2048)
        x = self.fc(x)
        return x
