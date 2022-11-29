import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MCKFFNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MCKFFNet, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1_1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.layer1_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.layer1_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2_1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.layer2_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )


        self.layer3_1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.layer3_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.layer3_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.w1 = nn.Parameter(torch.ones(3))
        self.w2 = nn.Parameter(torch.ones(3))
        self.w3 = nn.Parameter(torch.ones(3))

        self.w_l = nn.Parameter(torch.ones(3)*0.01)
        self.linear1 = nn.Linear(21504, 512)
        self.linear2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x1_1 = self.layer1_1(x)
        x1_2 = self.layer1_2(x)
        x1_3 = self.layer1_3(x)

        w11 = torch.exp(self.w1[0]) / torch.sum(torch.exp(self.w1))
        w12 = torch.exp(self.w1[1]) / torch.sum(torch.exp(self.w1))
        w13 = torch.exp(self.w1[2]) / torch.sum(torch.exp(self.w1))

        x_out1 = w11*x1_1 + w12*x1_2 + w13*x1_3
        x_out1_l = x_out1.view(x_out1.size(0), -1)

        x_out1 = F.max_pool2d(x_out1, 2)

        x2_1 = self.layer2_1(x_out1)
        x2_2 = self.layer2_2(x_out1)
        x2_3 = self.layer2_3(x_out1)

        w21 = torch.exp(self.w2[0]) / torch.sum(torch.exp(self.w2))
        w22 = torch.exp(self.w2[1]) / torch.sum(torch.exp(self.w2))
        w23 = torch.exp(self.w2[2]) / torch.sum(torch.exp(self.w2))

        x_out2 = w21 * x2_1 + w22 * x2_2 + w23 * x2_3
        x_out2_l = x_out2.view(x_out2.size(0), -1)

        x_out2 = F.max_pool2d(x_out2, 2)

        x3_1 = self.layer3_1(x_out2)
        x3_2 = self.layer3_2(x_out2)
        x3_3 = self.layer3_3(x_out2)

        w31 = torch.exp(self.w3[0]) / torch.sum(torch.exp(self.w3))
        w32 = torch.exp(self.w3[1]) / torch.sum(torch.exp(self.w3))
        w33 = torch.exp(self.w3[2]) / torch.sum(torch.exp(self.w3))

        x_out3 = w31 * x3_1 + w32 * x3_2 + w33 * x3_3
        x_out3_l = x_out3.view(x_out3.size(0), -1)

        x_linear = torch.cat((x_out1_l*self.w_l[0], x_out2_l*self.w_l[1], x_out3_l*self.w_l[2]), 1)
        # print(self.w_l)

        logits = self.linear2(self.linear1(x_linear))
        return logits


def test():
    net = MCKFFNet()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()
