# Cell 3: XYW-Net Model (Complete)

# ============ PDC Convolution ============
def createPDCFunc(PDC_type):
    """Create Pixel Difference Convolution function"""
    assert PDC_type in ['cv', 'cd', 'ad', 'rd', 'sd', 'p2d', '2sd', '2cd']
    
    if PDC_type == 'cv':
        return F.conv2d
    
    if PDC_type == '2sd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert weights.size(2) == 3 and weights.size(3) == 3
            shape = weights.shape
            if groups == shape[0]:
                weights_conv = (weights - weights[:, :, [1, 1, 1, 0, 0, 0, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]].view(shape))
            else:
                weights_conv = (weights - weights[:, :, [1, 1, 1, 0, 0, 0, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]].view(shape).flip(0))
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    
    return F.conv2d

class Conv2d(nn.Module):
    """PDC-enabled Conv2d"""
    def __init__(self, pdc_func='cv', in_channels=1, out_channels=1, kernel_size=3, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__()
        self.pdc = createPDCFunc(pdc_func)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return self.pdc(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# ============ Core XYW Components ============
class Xc1x1(nn.Module):
    """X pathway: Local contrast (center-surround with 3x3)"""
    def __init__(self, in_channels, out_channels):
        super(Xc1x1, self).__init__()
        self.Xcenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Xcenter_relu = nn.ReLU(inplace=True)
        self.Xsurround = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Xsurround_relu = nn.ReLU(inplace=True)

    def forward(self, input):
        xcenter = self.Xcenter_relu(self.Xcenter(input))
        xsurround = self.Xsurround_relu(self.Xsurround(input))
        xsurround = self.conv1_1(xsurround)
        return xsurround - xcenter

class Yc1x1(nn.Module):
    """Y pathway: Large receptive field (center-surround with 5x5 dilated)"""
    def __init__(self, in_channels, out_channels):
        super(Yc1x1, self).__init__()
        self.Ycenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Ycenter_relu = nn.ReLU(inplace=True)
        self.Ysurround = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=4, dilation=2, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Ysurround_relu = nn.ReLU(inplace=True)

    def forward(self, input):
        ycenter = self.Ycenter_relu(self.Ycenter(input))
        ysurround = self.Ysurround_relu(self.Ysurround(input))
        ysurround = self.conv1_1(ysurround)
        return ysurround - ycenter

class W(nn.Module):
    """W pathway: Directional (horizontal + vertical)"""
    def __init__(self, inchannel, outchannel):
        super(W, self).__init__()
        self.h = nn.Conv2d(inchannel, inchannel, kernel_size=(1, 3), padding=(0, 1), groups=inchannel)
        self.v = nn.Conv2d(inchannel, inchannel, kernel_size=(3, 1), padding=(1, 0), groups=inchannel)
        self.convh_1 = nn.Conv2d(inchannel, inchannel, kernel_size=1, bias=False)
        self.convv_1 = nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.h(x))
        h = self.convh_1(h)
        v = self.relu(self.v(h))
        v = self.convv_1(v)
        return v

# ============ XYW Blocks ============
class XYW_S(nn.Module):
    """XYW Start block"""
    def __init__(self, inchannel, outchannel, stride=1):
        super(XYW_S, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)

    def forward(self, x):
        return self.x_c(x), self.y_c(x), self.w(x)

class XYW(nn.Module):
    """XYW middle block"""
    def __init__(self, inchannel, outchannel, stride=1):
        super(XYW, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)

    def forward(self, xc, yc, w):
        return self.x_c(xc), self.y_c(yc), self.w(w)

class XYW_E(nn.Module):
    """XYW End block (combines X+Y+W)"""
    def __init__(self, inchannel, outchannel):
        super(XYW_E, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)

    def forward(self, xc, yc, w):
        return self.x_c(xc) + self.y_c(yc) + self.w(w)

# ============ Encoder Stages ============
class s1(nn.Module):
    def __init__(self, channel=30):
        super(s1, self).__init__()
        self.conv1 = nn.Conv2d(3, channel, kernel_size=7, padding=6, dilation=2)
        self.xyw1_1 = XYW_S(channel, channel)
        self.xyw1_2 = XYW(channel, channel)
        self.xyw1_3 = XYW_E(channel, channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.relu(self.conv1(x))
        xc, yc, w = self.xyw1_1(temp)
        xc, yc, w = self.xyw1_2(xc, yc, w)
        xyw1_3 = self.xyw1_3(xc, yc, w)
        return xyw1_3 + temp

class s2(nn.Module):
    def __init__(self, channel=60):
        super(s2, self).__init__()
        self.xyw2_1 = XYW_S(channel//2, channel, stride=2)
        self.xyw2_2 = XYW(channel, channel)
        self.xyw2_3 = XYW_E(channel, channel)
        self.shortcut = nn.Conv2d(channel//2, channel, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        xc, yc, w = self.xyw2_1(x)
        xc, yc, w = self.xyw2_2(xc, yc, w)
        xyw2_3 = self.xyw2_3(xc, yc, w)
        return xyw2_3 + self.shortcut(x)

class s3(nn.Module):
    def __init__(self, channel=120):
        super(s3, self).__init__()
        self.xyw3_1 = XYW_S(channel//2, channel, stride=2)
        self.xyw3_2 = XYW(channel, channel)
        self.xyw3_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(channel//2, channel, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc, yc, w = self.xyw3_1(x)
        xc, yc, w = self.xyw3_2(xc, yc, w)
        xyw3_3 = self.xyw3_3(xc, yc, w)
        return xyw3_3 + shortcut

class s4(nn.Module):
    def __init__(self, channel=120):
        super(s4, self).__init__()
        self.xyw4_1 = XYW_S(channel, channel, stride=2)
        self.xyw4_2 = XYW(channel, channel)
        self.xyw4_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc, yc, w = self.xyw4_1(x)
        xc, yc, w = self.xyw4_2(xc, yc, w)
        xyw4_3 = self.xyw4_3(xc, yc, w)
        return xyw4_3 + shortcut

# ============ Encoder ============
class encode(nn.Module):
    def __init__(self):
        super(encode, self).__init__()
        self.s1 = s1()
        self.s2 = s2()
        self.s3 = s3()
        self.s4 = s4()

    def forward(self, x):
        s1_out = self.s1(x)
        s2_out = self.s2(s1_out)
        s3_out = self.s3(s2_out)
        s4_out = self.s4(s3_out)
        return s1_out, s2_out, s3_out, s4_out

# ============ Adaptive Convolution ============
def upsample_filt(size):
    factor = (size + 1) // 2
    center = factor - 1 if size % 2 == 1 else factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, num_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((num_classes, num_classes, filter_size, filter_size), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(num_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)

class adap_conv(nn.Module):
    """Adaptive convolution with learnable weight"""
    def __init__(self, in_channels, out_channels, kz=3, pd=1):
        super(adap_conv, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(pdc_func='2sd', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.weight = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        return self.conv(x) * self.weight.sigmoid()

class Refine_block2_1(nn.Module):
    """Refinement block for decoder"""
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel, kz=3, pd=1)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel, kz=3, pd=1)
        self.factor = factor
        self.deconv_weight = nn.Parameter(bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)

    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, 
                                padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, 
                                               x1.size(3) - x2.size(3)*self.factor))
        return x1 + x2

# ============ RCF Decoder ============
class decode_rcf(nn.Module):
    def __init__(self):
        super(decode_rcf, self).__init__()
        self.f43 = Refine_block2_1(in_channel=(120, 120), out_channel=60, factor=2)
        self.f32 = Refine_block2_1(in_channel=(60, 60), out_channel=30, factor=2)
        self.f21 = Refine_block2_1(in_channel=(30, 30), out_channel=24, factor=2)
        self.f = nn.Conv2d(24, 1, kernel_size=1, padding=0)

    def forward(self, x):
        s3 = self.f43(x[2], x[3])
        s2 = self.f32(x[1], s3)
        s1 = self.f21(x[0], s2)
        out = self.f(s1)
        return out.sigmoid()

# ============ Full XYW-Net ============
class XYWNet(nn.Module):
    def __init__(self):
        super(XYWNet, self).__init__()
        self.encode = encode()
        self.decode = decode_rcf()

    def forward(self, x):
        endpoints = self.encode(x)
        out = self.decode(endpoints)
        return out
    
    def forward_with_stages(self, x):
        """Forward pass returning intermediate stage outputs for visualization"""
        s1, s2, s3, s4 = self.encode(x)
        final = self.decode((s1, s2, s3, s4))
        return final, (s1, s2, s3, s4)

# Create model
model = XYWNet().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")