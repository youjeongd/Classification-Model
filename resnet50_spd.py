'''
encoder를 resnet으로 사용함
'''

"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""


# 0218 WD 추가, momentum = 0.01 수정

import torch
import torch.nn as nn

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=0.01)  # Adjusted momentum to 0.01
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 2).reshape(x.shape[0] * x.shape[1], x.shape[2] * 4, int(x.shape[3]/2), int(x.shape[4]/2))
        return self.conv(x)
        # return self.conv(self.contract(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        layers = [
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False),
                  nn.BatchNorm2d(out_channels,momentum=0.01),  # Adjusted momentum to 0.01,
                  nn.ReLU(inplace=True),
        ]

        if stride ==2:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= 1, kernel_size=3, padding=1, bias= False),
            space_to_depth(),   # the output of this will result in 4*out_channels
            nn.BatchNorm2d(4*out_channels,momentum=0.01),  # Adjusted momentum to 0.01
            nn.ReLU(inplace=True),

            nn.Conv2d(4*out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion,momentum=0.01),  # Adjusted momentum to 0.01,                       
            ]

        else:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= stride, kernel_size=3, padding=1, bias= False),
            nn.BatchNorm2d(out_channels,momentum=0.01),  # Adjusted momentum to 0.01
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion,momentum=0.01),  # Adjusted momentum to 0.01                      
            ]

        layers.extend(layers2)

        self.residual_function = torch.nn.Sequential(*layers)

		
        # self.residual_function = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False),
		# 	space_to_depth(),   # the output of this will result in 4*out_channels
        #     nn.BatchNorm2d(4*out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(4*out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        # )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion,momentum=0.01)  # Adjusted momentum to 0.01
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=128):
        super().__init__()

        self.in_channels = 64

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))

        self.conv1 = Focus(1, 64, k=1,s=1)
        self.num_classes = num_classes

		
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)    # Here in_channels = 64, and num_block[0] = 64 and s = 1 
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        output = output.reshape(int(output.shape[0] / 8), 8, self.num_classes)

        return output


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



if __name__ =="__main__":
	net = resnet50()
	x = torch.empty((2,3,112,112)).normal_()
	print(net(x).shape)

# Weight Decay 적용
weight_decay = 1e-4  # Standard weight decay value for ResNet
parameters = [
    {"params": [p for n, p in net.named_parameters() if "bn" not in n], "weight_decay": weight_decay},  # BN 제외
    {"params": [p for n, p in net.named_parameters() if "bn" in n], "weight_decay": 0.0},  # BN에 Weight Decay 적용 X
]
optimizer = torch.optim.AdamW(parameters, lr=1e-3)

