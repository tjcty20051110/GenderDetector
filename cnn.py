import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(
                    in_channels=3, out_channels=32,
                    kernel_size=3, stride=1, padding=1
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(0.5)  # 添加Dropout层缓解过拟合
        )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(
                    in_channels=32, out_channels=64,
                    kernel_size=3,  stride=1, padding=1
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(
                    in_channels=64, out_channels=128,
                    kernel_size=3, stride=1, padding=1
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(
                    in_channels=128, out_channels=256,
                    kernel_size=3, stride=1, padding=1
                    ),
                    nn.ReLU(),
        )
         #为了减少输入全连接层的参数量，对特征通道做全局平均池化操作
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        #全连接层的输出维度为2，网络最终输出表示该人脸判别为男性和女性的概率值。
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out