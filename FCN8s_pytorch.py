import numpy as np
import torch
import torch.nn as nn

def get_upsample_filter(size):
    factor=(size+1)//2
    if size%2==1:
        center=factor-1
    else:
        center=factor-0.5
    og=np.ogrid[:size,:size]
    filter=(1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
    return torch.from_numpy(filter).float()

class FCN(nn.Module):

    def __init__(self, n_class=22):
        super(FCN, self).__init__()
        self.features_123 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )
        self.features_4 = nn.Sequential(
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )
        self.features_5 = nn.Sequential(
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )
        self.classifier = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # score_fr
            nn.Conv2d(4096, n_class, 1),
        )
        self.score_feat3 = nn.Conv2d(256, n_class, 1)
        self.score_feat4 = nn.Conv2d(512, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 16, stride=8,
                                              bias=False)
        self.upscore_4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False)
        self.upscore_5 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False)

    def forward(self, x):
        feat3 = self.features_123(x)  #1/8
        feat4 = self.features_4(feat3)  #1/16
        feat5 = self.features_5(feat4)  #1/32

        score5 = self.classifier(feat5)
        upscore5 = self.upscore_5(score5)
        score4 = self.score_feat4(feat4)
        score4 = score4[:, :, 5:5+upscore5.size()[2], 5:5+upscore5.size()[3]].contiguous()
        score4 += upscore5

        score3 = self.score_feat3(feat3)
        upscore4 = self.upscore_4(score4)
        score3 = score3[:, :, 9:9+upscore4.size()[2], 9:9+upscore4.size()[3]].contiguous()
        score3 += upscore4
        h = self.upscore(score3)
        h = h[:, :, 28:28+x.size()[2], 28:28+x.size()[3]].contiguous()

        return h

    def init_vgg16(self, vgg16, copy_fc8=True, init_upscore=True):
        for l1, l2 in zip(vgg16.features, [self.features_123,self.features_4,self.features_5]):
            if (isinstance(l1, nn.Conv2d) and
                    isinstance(l2, nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.data.size())
            l2.bias.data = l1.bias.data.view(l2.bias.data.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
        if init_upscore:
            # initialize upscore layer
            c1, c2, h, w = self.upscore.weight.data.size()
            assert c1 == c2 == n_class
            assert h == w
            weight = get_upsample_filter(h)
            self.upscore.weight.data = \
                weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)

            c1, c2, h, w = self.upscore_4.weight.data.size()
            assert c1 == c2 == n_class
            assert h == w
            weight = get_upsample_filter(h)
            self.upscore_4.weight.data = \
                weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)

            c1, c2, h, w = self.upscore_5.weight.data.size()
            assert c1 == c2 == n_class
            assert h == w
            weight = get_upsample_filter(h)
            self.upscore_5.weight.data = \
                weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
