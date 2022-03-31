import torch
import torch.nn as nn
import models.norms as norms

import math
import numpy as np
import random

from torch.nn import functional as F
from models.conv2d_gradfix import conv2d_gradfix

from torch.nn import init
import torch.nn.utils.spectral_norm as spectral_norm


class OASIS_Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        output_channel = opt.semantic_nc + 1
        self.channels = [3, 128, 128, 256, 256, 512, 512]
        self.body_up   = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(opt.num_res_blocks):
            self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
        # decoder part
        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], opt, 1))
        for i in range(1, opt.num_res_blocks-1):
            self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i], opt, 1))
        self.body_up.append(residual_block_D(2*self.channels[1], 64, opt, 1))
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def forward(self, input, for_features=False):
        x = input
        #encoder
        encoder_res = list()
        features=[]
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
            features.append(x)
        #decoder
        x = self.body_up[0](x)
        features.append(x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](torch.cat((encoder_res[-i-1], x), dim=1))
            features.append(x)
        ans = self.layer_up_last(x)

        if for_features:
            return features
        else:
            return ans


class residual_block_D(nn.Module):
    def __init__(self, fin, fout, opt, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        norm_layer = norms.get_spectral_norm(opt)
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s


class StyleGAN2Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, no_antialias=False, size=None, opt=None):
        super().__init__()
        self.opt = opt
        self.stddev_group = 16
        if size is None:
            size = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size)))))
            if "patch" in self.opt.netDu and self.opt.Du_patch_size is not None:
                size = 2 ** int(np.log2(self.opt.Du_patch_size))
            elif 'tile' in self.opt.netDu and self.opt.Du_patch_size is not None:
                size = 2 ** int(np.log2(self.opt.Du_patch_size))


        blur_kernel = [1, 3, 3, 1]
        channel_multiplier = ndf / 64
        channels = {
            4: min(384, int(4096 * channel_multiplier)),
            8: min(384, int(2048 * channel_multiplier)),
            16: min(384, int(1024 * channel_multiplier)),
            32: min(384, int(512 * channel_multiplier)),
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        if "smallpatch" in self.opt.netDu:
            final_res_log2 = 4
        elif "patch" in self.opt.netDu:
            final_res_log2 = 3
        else:
            final_res_log2 = 2

        for i in range(log_size, final_res_log2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        if False and "tile" in self.opt.netDu:
            in_channel += 1
        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        if "patch" in self.opt.netDu:
            self.final_linear = ConvLayer(channels[4], 1, 3, bias=False, activate=False)
        else:
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                EqualLinear(channels[4], 1),
            )

    def forward(self, input, for_features=False):
        if "patch" in self.opt.netDu and self.opt.Du_patch_size is not None:
            print(input.size())
            h, w = input.size(2), input.size(3)
            y = torch.randint(h - self.opt.Du_patch_size, ())
            x = torch.randint(w - self.opt.Du_patch_size, ())
            input = input[:, :, y:y + self.opt.Du_patch_size, x:x + self.opt.Du_patch_size]
        out = input
        features = []
        for i, conv in enumerate(self.convs):
            out = conv(out)
            features.append(out)
            # print(i, out.abs().mean())
        # out = self.convs(input)

        batch, channel, height, width = out.shape

        if False and "tile" in self.opt.netDu:
            group = min(batch, self.stddev_group)
            stddev = out.view(
                group, -1, 1, channel // 1, height, width
            )
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
            stddev = stddev.repeat(group, 1, height, width)
            out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        # print(out.abs().mean())

        if "patch" not in self.opt.netDu:
            out = out.view(batch, -1)

        out = self.final_linear(out)

        if for_features:
            return features
        else:
            return out

class UnconditionalDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, no_antialias=False, size=None, opt=None):
        super().__init__()
        self.opt = opt

        if 'tile' in opt.netDu :
            size = 2 ** int(np.log2(self.opt.Du_patch_size))
        else :
            size = opt.crop_size
        channel_multiplier = 2
        blur_kernel = [1, 3, 3, 1]

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        if 'tile' in opt.netDu :
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1),
            )
        else :
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * 2 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1),
            )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

    def init_weights(self, init_type='normal', gain=0.02):
        return


class TileStyleGAN2Discriminator(UnconditionalDiscriminator):
    def forward(self, input):
        if 'tile' in self.opt.netDu :
            B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
            size = self.opt.Du_patch_size
            Y = H // size
            X = W // size
            input = input.view(B, C, Y, size, X, size)
            input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
            out = super().forward(input)
            return out
        else :
            out = super().forward(input)
            return out



class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = math.sqrt(1) / math.sqrt(in_channel * (kernel_size ** 2))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        # print("Before EqualConv2d: ", input.abs().mean())
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # print("After EqualConv2d: ", out.abs().mean(), (self.weight * self.scale).abs().mean())

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        # print("FusedLeakyReLU: ", input.abs().mean())
        out = fused_leaky_relu(input, self.bias,
                               self.negative_slope,
                               self.scale)
        # print("FusedLeakyReLU: ", out.abs().mean())
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True, skip_gain=1.0):
        super().__init__()

        self.skip_gain = skip_gain
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel)

        if in_channel != out_channel or downsample:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out * self.skip_gain + skip) / math.sqrt(self.skip_gain ** 2 + 1.0)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if len(k.shape) == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (math.sqrt(1) / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(
        out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        :,
        max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0),
    ]

    # out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class WaveletDiscriminator(nn.Module):
    def __init__(self,opt, size=None, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        if size is None:
            size = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size)))))

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.dwt = HaarTransform(3)

        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2)) - 1

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.from_rgbs.append(FromRGB(channels[4]))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 2 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input,for_features = False):

        input = self.dwt(input)
        out = None
        features = []

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)
            features.append(out)

        _, out = self.from_rgbs[-1](input, out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        if for_features :
            return features
        else :
            return out

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class FromRGB(nn.Module):
    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.downsample = downsample

        if downsample:
            self.iwt = InverseHaarTransform(3)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = ConvLayer(3 * 4, out_channel, 1)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out


def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels,four_channels = True,levels =1):
        super().__init__()


        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        self.four_channels = four_channels

        if levels > 1 and not four_channels :
            self.next_level = HaarTransform(in_channels,four_channels,levels-1)
        else :
            self.next_level = None

    def forward(self, input):

        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        if self.next_level != None :
            ll = self.next_level(ll)

        if self.four_channels :
            return torch.cat((ll, lh, hl, hh), 1)
        else :
            return torch.cat((torch.cat((ll,lh),-2),torch.cat((hl,hh),-2)),-1)


class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels,four_channels = True,levels = 1):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

        self.four_channels = four_channels

        if levels > 1 and not four_channels :
            self.next_level = InverseHaarTransform(in_channels,four_channels,levels-1)
        else :
            self.next_level = None

    def forward(self, input):
        if self.four_channels :
            ll, lh, hl, hh = input.chunk(4, 1)
        else :
            toprow,bottomrow = input.chunk(2,-1)
            ll,lh = toprow.chunk(2,-2)
            hl,hh = bottomrow.chunk(2,-2)

        if self.next_level != None :
            ll = self.next_level(ll)

        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out




class BaseDiscriminator(nn.Module):
    def __init__(self):
        super(BaseDiscriminator, self).__init__()

    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     return parser

    # def print_network(self):
    #     if isinstance(self, list):
    #         self = self[0]
    #     num_params = 0
    #     for param in self.parameters():
    #         num_params += param.numel()
    #     print('Network [%s] was created. Total number of parameters: %.1f million. '
    #           'To see the architecture, do print(network).'
    #           % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseDiscriminator):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument('--n_layers_D', type=int, default=4,
    #                         help='# layers in each discriminator')
    #     return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        n_layers_D = 4

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = 38

        norm_layer = get_nonspade_norm_layer(opt, "spectralinstance")
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    # def compute_D_input_nc(self, opt):
    #     input_nc = opt.label_nc + opt.output_nc
    #     if opt.contain_dontcare_label:
    #         input_nc += 1
    #     if not opt.no_instance:
    #         input_nc += 1
    #     return input_nc

    def get_nonspade_norm_layer(opt, norm_type='instance'):
        # helper function to get # output channels of the previous layer
        def get_out_channel(layer):
            if hasattr(layer, 'out_channels'):
                return getattr(layer, 'out_channels')
            return layer.weight.size(0)

        # this function will be returned
        def add_norm_layer(layer):
            nonlocal norm_type
            if norm_type.startswith('spectral'):
                layer = spectral_norm(layer)
                subnorm_type = norm_type[len('spectral'):]

            if subnorm_type == 'none' or len(subnorm_type) == 0:
                return layer

            # remove bias in the previous layer, which is meaningless
            # since it has no effect after normalization
            if getattr(layer, 'bias', None) is not None:
                delattr(layer, 'bias')
                layer.register_parameter('bias', None)

            if subnorm_type == 'batch':
                norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
            # elif subnorm_type == 'sync_batch':
            #     norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
            elif subnorm_type == 'instance':
                norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
            else:
                raise ValueError('normalization layer %s is not recognized' % subnorm_type)

            return nn.Sequential(layer, norm_layer)

        return add_norm_layer

    def forward(self, input, for_features=False):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = for_features
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class MultiscaleDiscriminator(BaseDiscriminator):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument('--netD_subarch', type=str, default='n_layer',
    #                         help='architecture of each discriminator')
    #     parser.add_argument('--num_D', type=int, default=2,
    #                         help='number of discriminators to be used in multiscale')
    #     opt, _ = parser.parse_known_args()
    #
    #     # define properties of each discriminator of the multiscale discriminator
    #     subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
    #                                         'models.networks.discriminator')
    #     subnetD.modify_commandline_options(parser, is_train)
    #
    #     return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        num_D = 2


        for i in range(num_D):
            subnetD = NLayerDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)



    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input, for_features=False):
        result = []
        get_intermediate_features = for_features
        for name, D in self.named_children():
            out = D(input, for_features)
            if not for_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result






