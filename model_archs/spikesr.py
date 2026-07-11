from model_archs import common
from spikingjelly.activation_based.neuron import (
    LIFNode, IFNode, ParametricLIFNode,
)
from spikingjelly.activation_based import neuron, functional, layer, surrogate
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
from thop import profile
# from model_archs.TTST_arc import Self_AttentiHon
from model_archs.func import GPA

def make_model(args, parent=False):
    return RCAN(args)
v_th = 0.15

alpha = 1 / (2 ** 0.5)

# from model_archs.mambair_arch import *
# from model_archs.mynet_1 import Attention


# Attention
class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, "b f c h w -> b c f h w")
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        out = rearrange(out, "b c f h w -> b f c h w")
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, "b f c h w -> b (f c) h w")
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = x.unsqueeze(1)
        return self.sigmoid(x)

class TCJA(nn.Module):
    def __init__(self, kernel_size_t: int = 1, kernel_size_c: int = 1, T: int = 4, channel: int = 96):
        super().__init__()
        '''
        Please refer to TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks
        '''

        # Excitation
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=T, out_channels=T,
                      kernel_size=kernel_size_t, padding='same', bias=False),
        )
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

        # self.sa = SpatialAttention()
        # self.self_att = cubic_attention(dim=channel)
        self.gpa = GPA(dim=channel, patch_size=16, qk_dim=32, mlp_dim=100)
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        x_seq = x_seq.transpose(0, 1)
        '''
        model10: reduce the model size of model9
        input: B T C H W; Returns: B T C H W
        '''
        # Temporal channel joint attention
        x = torch.mean(x_seq, dim=[3, 4])  # B T C
        x_c = x.permute(0, 2, 1)  # B C T

        conv_t_out = self.conv(x)  # B T C --> B T C， 对时间维度卷积

        conv_c_out = self.conv_c(x_c)  # B C T --> B C T, 对通道维度做卷积
        conv_c_out = conv_c_out.permute(0, 2, 1)  # B C T --> B T C

        att = conv_c_out * conv_t_out  # B T C
        att = self.sigmoid(att)  # B T C

        # max_out = self.con(torch.amax(x_seq, dim =[3,4]))

        y_seq1 = x_seq * att[:, :, :, None, None]  # B T C H W * B T C --> B T C H W

        # Self-attention
        y_seq2 = []
        for i in range(self.T):
            # temp = self.self_att(x_seq[:, i, :, :, :])
            temp = self.gpa(x_seq[:, i, :, :, :])
            y_seq2.append(temp)
        # att_s = self.sa(x_seq)  # 1 1 1 H W
        y_seq2 = torch.stack(y_seq2, dim=1)

        out = y_seq1 + y_seq2
        # ReLU and transpose
        out = self.gelu(out)
        out = out.transpose(0, 1)
        # print(789, out.size())

        return out

class Spiking_Residual_Block(nn.Module):
    def __init__(self, dim):
        super(Spiking_Residual_Block, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.residual = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m'),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),

            LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False),
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False,
                         step_mode='m'),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th * 0.2, affine=True),
        )
        self.shortcut = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                         bias=False, step_mode='m'),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha,
                                                v_th=v_th, affine=True),
        )
        self.attn = TCJA(channel=dim)  # Apply TJCA

    def forward(self, x):
        shortcut = torch.clone(x)
        out = self.residual(x) + self.shortcut(x)  # T B C H W
        out = self.attn(out) + shortcut
        return out

class Feature_Refinement_Block(nn.Module):
    def __init__(self, channel, reduction):
        super(Feature_Refinement_Block, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Conv2d(channel, channel // 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 3, 1, 1),
            nn.Sigmoid()
        )
        # self.self_att = Self_Attention(dim=channel)

    def forward(self, x):
        # a = self.self_att(x)
        a = self.ca(x)
        t = self.sa(x)
        s = torch.mul((1 - t), a) + torch.mul(t, x)
        return s

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [Spiking_Residual_Block(dim=n_feat) for _ in range(n_resblocks)]
        # modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

        self.reduce_chan_level2 = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False),
            layer.Conv2d(int(n_feat), int(n_feat), kernel_size=1, bias=True, step_mode='m'),
            layer.ThresholdDependentBatchNorm2d(num_features=int(n_feat), alpha=alpha, v_th=v_th),
        )

    def forward(self, x):
        res = self.body(x)
        res = self.reduce_chan_level2(res)
        res += x
        return res

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, spike_mode="lif", LayerNorm_type='WithBias', bias=False):
        super(OverlapPatchEmbed, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)



    def forward(self, x):
        x = self.proj(x)

        return x

## Residual Channel Attention Network (RCAN)
class SpikeSR(nn.Module):
    def __init__(self, n_feats=64, conv=common.default_conv):
        super(SpikeSR, self).__init__()
        inp_channels = 3
        n_resgroups = 4
        n_resblocks = 2
        n_feats = n_feats
        kernel_size = 3
        scale = 4
        act = nn.ReLU(True)

        # define head module
        modules_head = [OverlapPatchEmbed(in_c=inp_channels, embed_dim=n_feats)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        # modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        # modules_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, inp_channels, kernel_size)]

        # direct pixel-shuffle for lightweight sr
        # modules_tail = [
        #     nn.Conv2d(n_feats, (scale ** 2) * 3, 3, 1, 1),
        #     nn.PixelShuffle(scale)
        # ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        # self.tail = nn.Sequential(*modules_tail)
        self.tail = common.UpsampleOneStep(scale=scale, num_feat=n_feats, num_out_ch=3)

        self.refinement = Feature_Refinement_Block(channel=n_feats, reduction=16)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(4, 1, 1, 1, 1)  #  train 函数用

        x = self.head(x)
        # x = (x.unsqueeze(0)).repeat(4, 1, 1, 1, 1)   # 本文件测试用

        res = self.body(x)
        res += x

        res = res.mean(0)

        out = self.refinement(res)
        out = self.tail(out)

        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input = torch.rand(1, 3, 128, 128).cuda(0)  # B C H W
    # model = ChannelAttention(in_planes=96).cuda()
    model = SpikeSR(n_feats=64).cuda(0)
    functional.set_step_mode(model, step_mode='m')
    functional.set_backend(model, backend='cupy')


    # model = Spiking_Residual_Block(dim=64).cuda()

    flops, params = profile(model, inputs=(input,))
    print("Param: {} K".format(params / 1e3))
    print("FLOPs: {} G".format(flops / 1e9))
    #
    out = model(input)
    print(out.size())