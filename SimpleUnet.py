import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Basic blocks
# ------------------------------------------------------------
class ConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class ResidualStack(nn.Module):
    def __init__(self, ch, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(ch) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


class DownBlock(nn.Module):
    """
    conv_down + resblk
    """
    def __init__(self, in_ch, out_ch, num_resblocks=1):
        super().__init__()
        self.down = ConvAct(in_ch, out_ch, k=3, s=2, p=1, act=True)
        self.res = ResidualStack(out_ch, num_resblocks)

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class PixelShuffleUp(nn.Module):
    """
    conv1x1 -> pixel_shuffle(2)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.expand = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, stride=1, padding=0)
        self.ps = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.expand(x)
        x = self.ps(x)
        return x


class SkipFuse(nn.Module):
    """
    concat된 encoder/decoder feature를 1x1 conv로 embedding
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.fuse(x)


# ------------------------------------------------------------
# UNet-like model
# ------------------------------------------------------------
class ResidualPixelShuffleUNet(nn.Module):
    """
    Structure:
        stem
        -> conv_down + resblk
        -> conv_down + resblk
        -> conv_down + resblk*4
        -> conv1x1 + pixel_shuffle + skip concat/embed + resblk
        -> conv1x1 + pixel_shuffle + skip concat/embed + resblk
        -> conv1x1 + pixel_shuffle + skip concat/embed + resblk
        -> out conv
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # stem
        self.stem = ConvAct(in_channels, c1, k=3, s=1, p=1, act=True)

        # encoder
        self.enc1 = DownBlock(c1, c2, num_resblocks=1)   # 1/2
        self.enc2 = DownBlock(c2, c3, num_resblocks=1)   # 1/4
        self.enc3 = DownBlock(c3, c4, num_resblocks=4)   # 1/8

        # decoder stage 1: c4 -> c3
        self.up1 = PixelShuffleUp(c4, c3)
        self.fuse1 = SkipFuse(c3 + c3, c3)
        self.dec1 = ResidualStack(c3, num_blocks=1)

        # decoder stage 2: c3 -> c2
        self.up2 = PixelShuffleUp(c3, c2)
        self.fuse2 = SkipFuse(c2 + c2, c2)
        self.dec2 = ResidualStack(c2, num_blocks=1)

        # decoder stage 3: c2 -> c1
        self.up3 = PixelShuffleUp(c2, c1)
        self.fuse3 = SkipFuse(c1 + c1, c1)
        self.dec3 = ResidualStack(c1, num_blocks=1)

        # output
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # stem
        x0 = self.stem(x)     # [B, c1, H, W]

        # encoder
        x1 = self.enc1(x0)    # [B, c2, H/2, W/2]
        x2 = self.enc2(x1)    # [B, c3, H/4, W/4]
        x3 = self.enc3(x2)    # [B, c4, H/8, W/8]

        # decoder
        d1 = self.up1(x3)     # [B, c3, H/4, W/4]
        d1 = self.fuse1(d1, x2)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)     # [B, c2, H/2, W/2]
        d2 = self.fuse2(d2, x1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)     # [B, c1, H, W]
        d3 = self.fuse3(d3, x0)
        d3 = self.dec3(d3)

        out = self.out_conv(d3)
        return out


# ------------------------------------------------------------
# Example
# ------------------------------------------------------------
if __name__ == "__main__":
    model = ResidualPixelShuffleUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
    )

    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
