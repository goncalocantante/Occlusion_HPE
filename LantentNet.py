import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class LantentNet(nn.Module):
    # LantentNet with 3 output layers for Euler angles: yaw, pitch and roll
    # Predicts Euler angles by bin classification + regression using expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(LantentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) #3 in_channels (channels in input image (RGB)) 64 out_channels(output by convolution, number of filters) stride is movement of kernel, addition of pixels to border of image
        self.bn1 = nn.BatchNorm2d(64) #minimizes internal covariate shift, a change of distribution regarding original true distribution 
        self.relu = nn.ReLU(inplace=True) #Applies the rectified linear unit function element-wise (inplace directly modify the tensor passed down, saves memory)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #finds max of kernel region
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7) #average pool of kernel region kernel size = 7
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 

        x = self.avgpool(x) 
        x = x.view(x.size(0), -1) #flatten tensor (latent vector)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return x, pre_yaw, pre_pitch, pre_roll
        


