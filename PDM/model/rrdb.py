import functools
import torch
from torch import nn
from torch.nn import functional as F


class RRDBNet(nn.Module):
    def __init__(self, input_channel: int,
                 num_feature: int,
                 output_channel: int, 
                 num_of_blocks: int,
                 scaling_factor: int,
                 gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=num_feature, gc=gc)

        self.conv_first = nn.Conv2d(input_channel, num_feature, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, num_of_blocks)
        self.trunk_conv = nn.Conv2d(num_feature, num_feature, 3, 1, 1, bias=True)

        num_down = int(scaling_factor ** 0.5)
        self.upconv1 = nn.Conv2d(num_feature, num_feature, 3, 1, 1, bias=True)

        # self.output_down = nn.Sequential(*[nn.LeakyReLU(negative_slope=0.2),
        #                                    nn.Conv2d(num_feature, num_feature, 3, 1, 1, bias=True), 
        #                                    nn.MaxPool2d(kernel_size=(2, 2))] * num_down)
        self.output_down = nn.Sequential(*[nn.LeakyReLU(negative_slope=0.2),
                                           nn.Conv2d(num_feature, num_feature, 3, 1, 1, bias=True)] * num_down)

        self.conv_last = nn.Conv2d(num_feature, output_channel, 3, 1, 1, bias=True)
        # self.HRconv = nn.Conv2d(num_feature, num_feature, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        feas = []
        x1 = (x + 1) / 2
        fea_first = fea = self.conv_first(x1)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)

        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea_hr = self.HRconv(fea)
        # out = self.conv_last(self.lrelu(fea_hr))
        fea = self.output_down(fea)
        out = self.conv_last(self.lrelu(fea))
        out = out.clamp(0, 1)
        out = out * 2 - 1

        return out, torch.cat(feas, dim=1)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    

def make_layer(block, n_layers, seq=False):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    if seq:
        return nn.Sequential(*layers)
    else:
        return nn.ModuleList(layers)