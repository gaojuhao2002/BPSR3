import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


def exists(x):
    return x is not None


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# added
class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, noise_level_emb_dim=None, use_affine_level=False):
        super(ConvBlock, self).__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, output_size, use_affine_level)  # added
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x, time_emb):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        out = self.noise_func(out, time_emb)  # added
        if self.activation is not None:
            return self.act(out)
        else:
            return out


# added
class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None, noise_level_emb_dim=None, use_affine_level=False):
        super(DeconvBlock, self).__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, output_size, use_affine_level)  # added
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x, time_emb):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)
        out = self.noise_func(out, time_emb)  # added
        if self.activation is not None:
            return self.act(out)
        else:
            return out


##added
class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None,
                 noise_level_emb_dim=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                    noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                  noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                    noise_level_emb_dim=noise_level_emb_dim, norm=None)

    def forward(self, x, time_emb):
        h0 = self.up_conv1(x, time_emb)
        l0 = self.up_conv2(h0, time_emb)
        h1 = self.up_conv3(l0 - x, time_emb)
        return h1 + h0


##added
class D_UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 noise_level_emb_dim=None, norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None,
                              noise_level_emb_dim=noise_level_emb_dim)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                    noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                  noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                    noise_level_emb_dim=noise_level_emb_dim, norm=None)

    def forward(self, x, time_emb):
        x = self.conv(x, time_emb)
        h0 = self.up_conv1(x, time_emb)
        l0 = self.up_conv2(h0, time_emb)
        h1 = self.up_conv3(l0 - x, time_emb)
        return h1 + h0


# added
class DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None,
                 noise_level_emb_dim=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                    noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                      noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                    noise_level_emb_dim=noise_level_emb_dim, norm=None)

    def forward(self, x, time_emb):
        l0 = self.down_conv1(x, time_emb)
        h0 = self.down_conv2(l0, time_emb)
        l1 = self.down_conv3(h0 - x, time_emb)
        return l1 + l0


# added
class D_DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True,
                 noise_level_emb_dim=None, activation='prelu',norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation,
                              noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                    noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                      noise_level_emb_dim=noise_level_emb_dim, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation,
                                    noise_level_emb_dim=noise_level_emb_dim, norm=None)

    def forward(self, x, time_emb):
        x = self.conv(x, time_emb)
        l0 = self.down_conv1(x, time_emb)
        h0 = self.down_conv2(l0, time_emb)
        l1 = self.down_conv3(h0 - x, time_emb)
        return l1 + l0
#out_size = （in_size - K + 2P）/ S +1



class Single_DBPN(nn.Module):
    def __init__(self, base_filter, noise_level_channel,scale_factor, num_stages=4):
        super(Single_DBPN, self).__init__()
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2


        self.down1 = DownBlock(base_filter, kernel, stride, padding, noise_level_emb_dim=noise_level_channel)
        self.up1 = UpBlock(base_filter, kernel, stride, padding, noise_level_emb_dim=noise_level_channel)
        self.down2 = DownBlock(base_filter, kernel, stride, padding, noise_level_emb_dim=noise_level_channel)

        self.up2 = D_UpBlock(base_filter, kernel, stride, padding, 2, noise_level_emb_dim=noise_level_channel)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 2, noise_level_emb_dim=noise_level_channel)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 3, noise_level_emb_dim=noise_level_channel)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 3, noise_level_emb_dim=noise_level_channel)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 4, noise_level_emb_dim=noise_level_channel)

        # 缓冲Change Scale Conv
        self.CS_conv = ConvBlock(num_stages * base_filter, base_filter, 3, 1, 1, activation=None, norm=None,
                                     noise_level_emb_dim=noise_level_channel)



    def forward(self, x, t):


        l1 = self.down1(x, t)
        h1 = self.up1(l1, t)
        l2 = self.down2(h1, t)

        concat_l = torch.cat((l2, l1), 1)
        h = self.up2(concat_l, t)

        concat_h = torch.cat((h, h1), 1)
        l = self.down3(concat_h, t)
        #=-----------------------------------------------------------这个模块可以拓展为stage的函数
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up3(concat_l, t)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h, t)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l, t)
        # =-----------------------------------------------------------这个模块可以拓展为stage的函数
        concat_h = torch.cat((h, concat_h), 1)
        out = self.CS_conv(concat_h, t)
        return out

class Mul_Scale_DBPN(nn.Module):
    def __init__(self,num_channels, base_filter,feat=256,with_noise_level_emb = True, inner_channel=64):

        # 可能的调节空间尺寸的方式：
        # 变化base_filter,当缩放规模小（图片大）时适当减少，反正处于较大水平，使得平衡
        # 串并连，释放中间图片资源
        # 改写Single_DBPN链长度num_stages随Scale调节，Scale小的链长，反之短
        super(Mul_Scale_DBPN, self).__init__()
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None


        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None,
                               noise_level_emb_dim=noise_level_channel)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None,
                               noise_level_emb_dim=noise_level_channel)
        self.Scale0d = ConvBlock(base_filter, base_filter//2, 1, 1, 0, activation='prelu', norm=None,
                               noise_level_emb_dim=noise_level_channel)

        self.Scale2d = Single_DBPN(base_filter,noise_level_channel,2)
        self.Scale4d = Single_DBPN(base_filter, noise_level_channel, 4)
        self.Scale8d = Single_DBPN(base_filter, noise_level_channel, 8)
        # 结束的缓冲
        num_stages_list=[4,4,4]#可以为了空间，调节numstages
        self.output_conv= ConvBlock(len(num_stages_list) * base_filter+base_filter//2, 3,3, 1, 1, activation=None, norm=None,
                                     noise_level_emb_dim=noise_level_channel)
        # 参数初始化==================================================================================================================!!!!!!!2023.1.1修改
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('Linear') != -1:
                torch.nn.init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)
    def forward(self, x,time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        x = self.feat0(x, t)
        x = self.feat1(x, t)
        sc0=self.Scale0d(x,t)
        sc2=self.Scale2d(x, t)
        sc4 = self.Scale4d(x, t)
        sc8 = self.Scale8d(x, t)
        # 空间换时间，，，，，时间换空间，，，，串联并联，先进先出，算完删除图片，只保存结果的特征，释放空间
        x=torch.cat([sc2,sc4,sc8,sc0],1)

        return self.output_conv(x,t)