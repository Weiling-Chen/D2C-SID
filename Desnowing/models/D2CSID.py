from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from models.network import *
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta

class BottleNect(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 63
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.dw_17 = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 7 // 2), stride=1, groups=dim)
        self.dw_71 = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(7 // 2, 0), stride=1, groups=dim)
        self.dw_111 = nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 11 // 2), stride=1, groups=dim)
        self.dw_1111 = nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(11 // 2, 0), stride=1, groups=dim)
        self.dw_121 = nn.Conv2d(dim, dim, kernel_size=(1, 21), padding=(0, 21 // 2), stride=1, groups=dim)
        self.dw_211 = nn.Conv2d(dim, dim, kernel_size=(21, 1), padding=(21 // 2, 0), stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)
        out17 = self.dw_71(self.dw_17(out))
        out11 = self.dw_111(self.dw_1111(out))
        out121 = self.dw_121(self.dw_211(out))

        # out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = x + out17 + out11 + out121 + x_sca
        out = self.act(out)
        return self.out_conv(out)

class BasicConv1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, gelu=False, bn=False, bias=True):
        super(BasicConv1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.gelu = nn.GELU() if gelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.gelu is not None:
            x = self.gelu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self, channel):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv1(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, gelu=False)
        self.dw1 = nn.Sequential(
            BasicConv1(channel, channel, 5, stride=1, dilation=2, padding=4, groups=channel),
            BasicConv1(channel, channel, 7, stride=1, dilation=3, padding=9, groups=channel)
        )
        self.dw2 = BasicConv1(channel, channel, kernel_size, stride=1, padding=1, groups=channel)

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)
        return out


class LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel

        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()

        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2,3), keepdim=True)
        return self.a*out*x + self.b*x

class ParamidAttention(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        pyramid = 1
        self.spatial_gate = SpatialGate(channel)
        layers = [LocalAttention(channel, p=i) for i in range(pyramid-1,-1,-1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))
    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a*out + self.b*x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
class EBlock1(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock1, self).__init__()

        self.layers = UNet(out_channel, out_channel, num_res)
    def forward(self, x):
        return self.layers(x)


class DBlock1(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock1, self).__init__()

        self.layers = UNet(channel, channel, num_res)
    def forward(self, x):
        return self.layers(x)

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(1, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

class FocalNet(nn.Module):
    def __init__(self, num_res=4):
        super(FocalNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock1(base_channel, num_res),
            # EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            # DBlock(base_channel, num_res),
            DBlock1(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.Priornet = Prponet(10)
        # VAE后验网络
        self.Posteriornet = Prponet(10)
        self.convs = nn.Conv2d(129, 128, kernel_size=3, padding=1)

        pyramid_attention = []
        for _ in range(1):
            pyramid_attention.append(ParamidAttention(base_channel * 4))
        self.pyramid_attentions = nn.Sequential(*pyramid_attention)
        self.spatial_axes = [2, 3]
        self.bottelneck = BottleNect(base_channel * 4)


    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)


    def forward(self, x, y, training=False):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        if training:
            smple_z_po, mu_po, logvar_po, dist_po = self.Priornet(x, y, po=True)
            smple_z_pr, mu_pr, logvar_pr, dist_pr = self.Priornet(x, x, po=False)

            # smple_z_po = torch.unsqueeze(smple_z_po, 2)
            # smple_z_po = self.tile(smple_z_po, 2, z.shape[self.spatial_axes[0]])
            # smple_z_po = torch.unsqueeze(smple_z_po, 3)
            # smple_z_po = self.tile(smple_z_po, 3, z.shape[self.spatial_axes[1]])

            z = torch.cat((z, smple_z_po), dim=1)
            z = self.convs(z)

            z = self.bottelneck(z)

            z = self.Decoder[0](z)
            z_ = self.ConvsOut[0](z)
            # 128
            z = self.feat_extract[3](z)
            outputs.append(z_+x_4)

            z = torch.cat([z, res2], dim=1)
            z = self.Convs[0](z)
            z = self.Decoder[1](z)
            z_ = self.ConvsOut[1](z)
            # 256
            z = self.feat_extract[4](z)
            outputs.append(z_+x_2)

            z = torch.cat([z, res1], dim=1)
            z = self.Convs[1](z)
            z = self.Decoder[2](z)
            z = self.feat_extract[5](z)
            outputs.append(z+x)

            return outputs, mu_po, logvar_po, mu_pr, logvar_pr, dist_po, dist_pr

        else:
            smple_z_pr, mu_pr, logvar_pr, dist_pr = self.Priornet(x, x, po=False)
            # --------------------------------------------------------------------------
            # smple_z_pr = torch.unsqueeze(smple_z_pr, 2)
            # smple_z_pr = self.tile(smple_z_pr, 2, z.shape[self.spatial_axes[0]])
            # smple_z_pr = torch.unsqueeze(smple_z_pr, 3)
            # smple_z_pr = self.tile(smple_z_pr, 3, z.shape[self.spatial_axes[1]])
            z = torch.cat((z, smple_z_pr), dim=1)

            z = self.convs(z)

            z = self.bottelneck(z)

            z = self.Decoder[0](z)
            z_ = self.ConvsOut[0](z)
            # 128
            z = self.feat_extract[3](z)
            outputs.append(z_ + x_4)

            z = torch.cat([z, res2], dim=1)
            z = self.Convs[0](z)
            z = self.Decoder[1](z)
            z_ = self.ConvsOut[1](z)
            # 256
            z = self.feat_extract[4](z)
            outputs.append(z_ + x_2)

            z = torch.cat([z, res1], dim=1)
            z = self.Convs[1](z)
            z = self.Decoder[2](z)
            z = self.feat_extract[5](z)
            outputs.append(z + x)

            return outputs


def build_net():
    return FocalNet()
if __name__ == '__main__':
    from thop import profile

    input_sen1 = torch.randn(1, 1, 256, 256).cuda()
    input_sen2 = torch.randn(1, 1, 256, 256).cuda()
    input = (input_sen1,input_sen2)
    fsnet = FocalNet().cuda()
    flops, params = profile(fsnet, inputs=(input[0],input[1],))

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("Params=", str(params / 1e6) + '{}'.format("M"))

    # print('# model_restoration parameters: %.2f M' % (sum(param.numel() for param in fsnet.parameters()) / 1e6))

