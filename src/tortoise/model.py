import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f'initialize network with {init_type}')
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t),
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class U_Net(nn.Module):
    def __init__(self, img_ch=13, output_ch=1, base=16, depth=4):
        super().__init__()

        assert depth >= 2, "depth must be at least 2"

        self.depth = depth
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = [base * (2 ** i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        in_ch = img_ch
        for ch in channels:
            self.enc_blocks.append(conv_block(in_ch, ch))
            in_ch = ch

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.up_convs.append(up_conv(channels[idx], channels[idx - 1]))
            self.dec_blocks.append(conv_block(channels[idx - 1] * 2, channels[idx - 1]))

        self.Conv_1x1 = nn.Conv2d(channels[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.pool(x)

        for i in range(self.depth - 1):
            x = self.up_convs[i](x)
            skip = skips[-(i + 1)]
            x = torch.cat((skip, x), dim=1)
            x = self.dec_blocks[i](x)

        return self.Conv_1x1(x)


class R2U_Net(nn.Module):
    def __init__(self, img_ch=13, output_ch=1, t=2, base=16, depth=4):
        super().__init__()

        assert depth >= 2, "depth must be at least 2"

        self.depth = depth
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = [base * (2 ** i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        in_ch = img_ch
        for ch in channels:
            self.enc_blocks.append(RRCNN_block(in_ch, ch, t=t))
            in_ch = ch

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.up_convs.append(up_conv(channels[idx], channels[idx - 1]))
            self.dec_blocks.append(RRCNN_block(channels[idx - 1] * 2, channels[idx - 1], t=t))

        self.Conv_1x1 = nn.Conv2d(channels[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.pool(x)

        for i in range(self.depth - 1):
            x = self.up_convs[i](x)
            skip = skips[-(i + 1)]
            x = torch.cat((skip, x), dim=1)
            x = self.dec_blocks[i](x)

        return self.Conv_1x1(x)


class AttU_Net(nn.Module):
    def __init__(self, img_ch=13, output_ch=1, base=16, depth=4):
        super().__init__()

        assert depth >= 2, "depth must be at least 2"

        self.depth = depth
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = [base * (2 ** i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        in_ch = img_ch
        for ch in channels:
            self.enc_blocks.append(conv_block(in_ch, ch))
            in_ch = ch

        self.up_convs = nn.ModuleList()
        self.att_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.up_convs.append(up_conv(channels[idx], channels[idx - 1]))
            self.att_blocks.append(
                Attention_block(F_g=channels[idx - 1], F_l=channels[idx - 1], F_int=channels[idx - 1] // 2)
            )
            self.dec_blocks.append(conv_block(channels[idx - 1] * 2, channels[idx - 1]))

        self.Conv_1x1 = nn.Conv2d(channels[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.pool(x)

        for i in range(self.depth - 1):
            x = self.up_convs[i](x)
            skip = skips[-(i + 1)]
            skip = self.att_blocks[i](g=x, x=skip)
            x = torch.cat((skip, x), dim=1)
            x = self.dec_blocks[i](x)

        return self.Conv_1x1(x)


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=13, output_ch=1, t=2, base=16, depth=4):
        super().__init__()

        assert depth >= 2, "depth must be at least 2"

        self.depth = depth
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = [base * (2 ** i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        in_ch = img_ch
        for ch in channels:
            self.enc_blocks.append(RRCNN_block(in_ch, ch, t=t))
            in_ch = ch

        self.up_convs = nn.ModuleList()
        self.att_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for idx in range(depth - 1, 0, -1):
            self.up_convs.append(up_conv(channels[idx], channels[idx - 1]))
            self.att_blocks.append(
                Attention_block(F_g=channels[idx - 1], F_l=channels[idx - 1], F_int=channels[idx - 1] // 2)
            )
            self.dec_blocks.append(RRCNN_block(channels[idx - 1] * 2, channels[idx - 1], t=t))

        self.Conv_1x1 = nn.Conv2d(channels[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.pool(x)

        for i in range(self.depth - 1):
            x = self.up_convs[i](x)
            skip = skips[-(i + 1)]
            skip = self.att_blocks[i](g=x, x=skip)
            x = torch.cat((skip, x), dim=1)
            x = self.dec_blocks[i](x)

        return self.Conv_1x1(x)
