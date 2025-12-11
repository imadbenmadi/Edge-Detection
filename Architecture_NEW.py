# ============================================================
#   XYW-NET: COMPLETE ARCHITECTURE (ENCODER + ITM + ELC)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
#  PDC CONVOLUTION (Pixel Difference Convolution)
# ============================================================
def createPDCFunc(PDC_type):
    assert PDC_type in ['cv', '2sd']
    
    if PDC_type == 'cv':
        return F.conv2d
    
    if PDC_type == '2sd':
        # Pixel difference convolution (paper uses this for ELC)
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert weights.size(2) == 3 and weights.size(3) == 3
            shape = weights.shape
            offset = weights[:, :, 
                             [1,1,1,0,0,0,2,2,2],
                             [0,1,2,0,1,2,0,1,2]
                             ].view(shape)
            
            diff_weights = weights - offset
            return F.conv2d(x, diff_weights, bias, stride, padding, dilation, groups)
        
        return func


class Conv2d(nn.Module):
    """PDC-enabled conv layer"""
    def __init__(self, pdc_func='cv', in_channels=1, out_channels=1, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.pdc = createPDCFunc(pdc_func)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.pdc(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# ============================================================
#  XYW COMPONENTS (ENCODER)
# ============================================================
class Xc1x1(nn.Module):
    """X: small RF center-surround"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.center = nn.Conv2d(in_channels, out_channels, 1)
        self.surround = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        c = self.relu(self.center(x))
        s = self.relu(self.surround(x))
        s = self.proj(s)
        return s - c


class Yc1x1(nn.Module):
    """Y: large RF dilated center-surround"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.center = nn.Conv2d(in_channels, out_channels, 1)
        self.surround = nn.Conv2d(in_channels, out_channels, 5, padding=4, dilation=2, groups=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        c = self.relu(self.center(x))
        s = self.relu(self.surround(x))
        s = self.proj(s)
        return s - c


class W(nn.Module):
    """W: directional horizontal + vertical pathway"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.h = nn.Conv2d(in_ch, in_ch, (1,3), padding=(0,1), groups=in_ch)
        self.v = nn.Conv2d(in_ch, in_ch, (3,1), padding=(1,0), groups=in_ch)
        self.proj1 = nn.Conv2d(in_ch, in_ch, 1)
        self.proj2 = nn.Conv2d(in_ch, out_ch, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.h(x))
        h = self.proj1(h)
        v = self.relu(self.v(h))
        return self.proj2(v)


class XYW_S(nn.Module):
    """Start block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.x = Xc1x1(in_ch, out_ch)
        self.y = Yc1x1(in_ch, out_ch)
        self.w = W(in_ch, out_ch)

    def forward(self, x):
        return self.x(x), self.y(x), self.w(x)


class XYW(nn.Module):
    """Middle block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.x = Xc1x1(in_ch, out_ch)
        self.y = Yc1x1(in_ch, out_ch)
        self.w = W(in_ch, out_ch)

    def forward(self, xc, yc, w):
        return self.x(xc), self.y(yc), self.w(w)


class XYW_E(nn.Module):
    """End block: fusion of X+Y+W"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.x = Xc1x1(in_ch, out_ch)
        self.y = Yc1x1(in_ch, out_ch)
        self.w = W(in_ch, out_ch)

    def forward(self, xc, yc, w):
        return self.x(xc) + self.y(yc) + self.w(w)


# ============================================================
#  ENCODER STAGES (4 RESOLUTION LEVELS)
# ============================================================
class s1(nn.Module):
    def __init__(self, ch=30):
        super().__init__()
        self.stem = nn.Conv2d(3, ch, 7, padding=6, dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.b1 = XYW_S(ch, ch)
        self.b2 = XYW(ch, ch)
        self.b3 = XYW_E(ch, ch)

    def forward(self, x):
        t = self.relu(self.stem(x))
        xc, yc, w = self.b1(t)
        xc, yc, w = self.b2(xc, yc, w)
        out = self.b3(xc, yc, w)
        return out + t


class s2(nn.Module):
    def __init__(self, ch=60):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.b1 = XYW_S(ch//2, ch)
        self.b2 = XYW(ch, ch)
        self.b3 = XYW_E(ch, ch)
        self.short = nn.Conv2d(ch//2, ch, 1)

    def forward(self, x):
        x = self.pool(x)
        xc, yc, w = self.b1(x)
        xc, yc, w = self.b2(xc, yc, w)
        out = self.b3(xc, yc, w)
        return out + self.short(x)


class s3(nn.Module):
    def __init__(self, ch=120):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.b1 = XYW_S(ch//2, ch)
        self.b2 = XYW(ch, ch)
        self.b3 = XYW_E(ch, ch)
        self.short = nn.Conv2d(ch//2, ch, 1)

    def forward(self, x):
        x = self.pool(x)
        sc = self.short(x)
        xc, yc, w = self.b1(x)
        xc, yc, w = self.b2(xc, yc, w)
        return self.b3(xc, yc, w) + sc


class s4(nn.Module):
    def __init__(self, ch=120):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.b1 = XYW_S(ch, ch)
        self.b2 = XYW(ch, ch)
        self.b3 = XYW_E(ch, ch)
        self.short = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        x = self.pool(x)
        sc = self.short(x)
        xc, yc, w = self.b1(x)
        xc, yc, w = self.b2(xc, yc, w)
        return self.b3(xc, yc, w) + sc


class encode(nn.Module):
    def forward(self, x):
        s1_out = s1()(x)
        s2_out = s2()(s1_out)
        s3_out = s3()(s2_out)
        s4_out = s4()(s3_out)
        return s1_out, s2_out, s3_out, s4_out


# ============================================================
#  UPSAMPLE HELPERS (SAME AS IN YOUR CODE)
# ============================================================
def upsample_filt(size):
    factor = (size + 1) // 2
    center = factor - 1 if size % 2 == 1 else factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, C):
    fs = 2 * factor - factor % 2
    w = np.zeros((C, C, fs, fs), dtype=np.float32)
    kern = upsample_filt(fs)
    for i in range(C):
        w[i, i] = kern
    return torch.Tensor(w)


class Refine_block2_1(nn.Module):
    """Refinement block used in ITM"""
    def __init__(self, in_ch, out_ch, factor):
        super().__init__()
        self.pre1 = Conv2d('2sd', in_ch[0], out_ch, 3, padding=1)
        self.pre2 = Conv2d('2sd', in_ch[1], out_ch, 3, padding=1)
        self.deconv_w = nn.Parameter(bilinear_upsample_weights(factor, out_ch), requires_grad=False)
        self.factor = factor

    def forward(self, x_high, x_low):
        h = self.pre1(x_high)
        l = self.pre2(x_low)
        l = F.conv_transpose2d(l, self.deconv_w, stride=self.factor,
                               padding=int(self.factor/2),
                               output_padding=(h.size(2) - l.size(2)*self.factor,
                                               h.size(3) - l.size(3)*self.factor))
        return h + l


# ============================================================
#  ELC BLOCK (Edge Localization Convolution)
# ============================================================
class ELCBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pdc = Conv2d('2sd', ch, ch, 3, padding=1)
        self.norm = nn.InstanceNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        x = self.pdc(x)
        x = self.norm(x)
        x = self.relu(x)
        return torch.sigmoid(self.out(x))


# ============================================================
#  ITM + ELC DECODER (XYW-Net)
# ============================================================
class decode_xyw(nn.Module):
    def __init__(self):
        super().__init__()
        self.f43 = Refine_block2_1((120,120), 64, 2)
        self.f32 = Refine_block2_1((60,64),   48, 2)
        self.f21 = Refine_block2_1((30,48),   32, 2)
        self.elc = ELCBlock(32)

    def forward(self, endpoints):
        s1, s2, s3, s4 = endpoints
        x3 = self.f43(s3, s4)
        x2 = self.f32(s2, x3)
        x1 = self.f21(s1, x2)
        return self.elc(x1)


# ============================================================
#  FULL XYW-NET (ENCODER + DECODER)
# ============================================================
class XYWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encode()
        self.decoder = decode_xyw()

    def forward(self, x):
        endpoints = self.encoder(x)
        return self.decoder(endpoints)
