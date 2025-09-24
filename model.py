import torch
import torch.nn as nn
from torch.nn import functional as F
import numbers
from einops import rearrange


#### LAYERNORM ver 2 ####
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


## Resizing modules ##
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


############
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_AdaLN(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., param_dim=18):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.adaLN_modulation = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(param_dim, c, bias=True),
            nn.SiLU(),
            nn.Linear(c, 4 * c, bias=True)
        )

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, control_param):
        shift_attn, scale_attn, shift_mlp, scale_mlp = self.adaLN_modulation(
            control_param).unsqueeze(-1).unsqueeze(-1).chunk(4, dim=1)

        x = inp

        x = self.norm1(x)
        x = modulate(x, shift_attn, scale_attn)  # added

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = modulate(x, shift_mlp, scale_mlp)  # added

        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


############ Denoising Networks ############
class BaseDenoiser_SID(nn.Module):
    def __init__(self, channels=32, num_blocks=[4, 4, 4, 8]):
        super(BaseDenoiser_SID, self).__init__()

        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.NAFBlock_enc1 = nn.ModuleList(
            [NAFBlock(c=channels) for i in range(num_blocks[0])])
        self.NAFBlock_enc2 = nn.ModuleList(
            [NAFBlock(c=channels * 2) for i in range(num_blocks[1])])
        self.NAFBlock_enc3 = nn.ModuleList(
            [NAFBlock(c=channels * 4) for i in range(num_blocks[2])])
        self.NAFBlock_mid = nn.ModuleList(
            [NAFBlock(c=channels * 8) for i in range(num_blocks[3])])
        self.NAFBlock_dec3 = nn.ModuleList(
            [NAFBlock(c=channels * 4) for i in range(num_blocks[2])])
        self.NAFBlock_dec2 = nn.ModuleList(
            [NAFBlock(c=channels * 2) for i in range(num_blocks[1])])
        self.NAFBlock_dec1 = nn.ModuleList(
            [NAFBlock(c=channels) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(channels)
        self.down2_3 = Downsample(channels * 2)
        self.down3_4 = Downsample(channels * 4)

        self.up4_3 = Upsample(channels * 8)
        self.channel_reduce3 = nn.Conv2d(channels * 8, channels * 4, kernel_size=1, bias=False)
        self.up3_2 = Upsample(channels * 4)
        self.channel_reduce2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False)
        self.up2_1 = Upsample(channels * 2)
        self.channel_reduce1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, inp, param):
        # network
        x = self.conv_in(inp)

        # NAFBlock + AdaLN
        for block in self.NAFBlock_enc1:
            x = block(x)
        down1 = x
        x = self.down1_2(x)

        for block in self.NAFBlock_enc2:
            x = block(x)
        down2 = x
        x = self.down2_3(x)

        for block in self.NAFBlock_enc3:
            x = block(x)
        down3 = x
        x = self.down3_4(x)

        for block in self.NAFBlock_mid:
            x = block(x)

        x = self.up4_3(x)
        x = torch.cat([x, down3], 1)
        x = self.channel_reduce3(x)
        for block in self.NAFBlock_dec3:
            x = block(x)

        x = self.up3_2(x)
        x = torch.cat([x, down2], 1)
        x = self.channel_reduce2(x)
        for block in self.NAFBlock_dec2:
            x = block(x)

        x = self.up2_1(x)
        x = torch.cat([x, down1], 1)
        x = self.channel_reduce1(x)
        for block in self.NAFBlock_dec1:
            x = block(x)

        # long skip connection
        x = self.conv_out(x) + inp
        return x


class BaseDenoiser_SIDD(nn.Module):
    def __init__(self, channels=32, num_blocks=[4, 4, 4, 8]):
        super(BaseDenoiser_SIDD, self).__init__()

        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.NAFBlock_enc1 = nn.ModuleList(
            [NAFBlock(c=channels) for i in range(num_blocks[0])])
        self.NAFBlock_enc2 = nn.ModuleList(
            [NAFBlock(c=channels * 2) for i in range(num_blocks[1])])
        self.NAFBlock_enc3 = nn.ModuleList(
            [NAFBlock(c=channels * 4) for i in range(num_blocks[2])])
        self.NAFBlock_mid = nn.ModuleList(
            [NAFBlock(c=channels * 8) for i in range(num_blocks[3])])
        self.NAFBlock_dec3 = nn.ModuleList(
            [NAFBlock(c=channels * 4) for i in range(num_blocks[2])])
        self.NAFBlock_dec2 = nn.ModuleList(
            [NAFBlock(c=channels * 2) for i in range(num_blocks[1])])
        self.NAFBlock_dec1 = nn.ModuleList(
            [NAFBlock(c=channels) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(channels)
        self.down2_3 = Downsample(channels * 2)
        self.down3_4 = Downsample(channels * 4)

        self.up4_3 = Upsample(channels * 8)
        self.channel_reduce3 = nn.Conv2d(channels * 8, channels * 4, kernel_size=1, bias=False)
        self.up3_2 = Upsample(channels * 4)
        self.channel_reduce2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False)
        self.up2_1 = Upsample(channels * 2)
        self.channel_reduce1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, inp, param, phone):
        # network
        x = self.conv_in(inp)

        # NAFBlock + AdaLN
        for block in self.NAFBlock_enc1:
            x = block(x)
        down1 = x
        x = self.down1_2(x)

        for block in self.NAFBlock_enc2:
            x = block(x)
        down2 = x
        x = self.down2_3(x)

        for block in self.NAFBlock_enc3:
            x = block(x)
        down3 = x
        x = self.down3_4(x)

        for block in self.NAFBlock_mid:
            x = block(x)

        x = self.up4_3(x)
        x = torch.cat([x, down3], 1)
        x = self.channel_reduce3(x)
        for block in self.NAFBlock_dec3:
            x = block(x)

        x = self.up3_2(x)
        x = torch.cat([x, down2], 1)
        x = self.channel_reduce2(x)
        for block in self.NAFBlock_dec2:
            x = block(x)

        x = self.up2_1(x)
        x = torch.cat([x, down1], 1)
        x = self.channel_reduce1(x)
        for block in self.NAFBlock_dec1:
            x = block(x)

        # long skip connection
        x = self.conv_out(x) + inp
        return x


class CPADNet_SID(nn.Module):
    def __init__(self, channels=32, num_blocks=[4, 4, 4, 8]):
        super(CPADNet_SID, self).__init__()

        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.NAFBlock_enc1 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels, param_dim=27) for i in range(num_blocks[0])])
        self.NAFBlock_enc2 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 2, param_dim=27) for i in range(num_blocks[1])])
        self.NAFBlock_enc3 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 4, param_dim=27) for i in range(num_blocks[2])])
        self.NAFBlock_mid = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 8, param_dim=27) for i in range(num_blocks[3])])
        self.NAFBlock_dec3 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 4, param_dim=27) for i in range(num_blocks[2])])
        self.NAFBlock_dec2 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 2, param_dim=27) for i in range(num_blocks[1])])
        self.NAFBlock_dec1 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels, param_dim=27) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(channels)
        self.down2_3 = Downsample(channels * 2)
        self.down3_4 = Downsample(channels * 4)

        self.up4_3 = Upsample(channels * 8)
        self.channel_reduce3 = nn.Conv2d(channels * 8, channels * 4, kernel_size=1, bias=False)
        self.up3_2 = Upsample(channels * 4)
        self.channel_reduce2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False)
        self.up2_1 = Upsample(channels * 2)
        self.channel_reduce1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers :
        for block in self.NAFBlock_enc1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_enc2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_enc3:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_mid:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_dec3:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_dec2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_dec1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, inp, param_vector):
        # getting parameters ready
        # network
        x = self.conv_in(inp)

        # NAFBlock + AdaLN
        for block in self.NAFBlock_enc1:
            x = block(x, param_vector)
        down1 = x
        x = self.down1_2(x)

        for block in self.NAFBlock_enc2:
            x = block(x, param_vector)
        down2 = x
        x = self.down2_3(x)

        for block in self.NAFBlock_enc3:
            x = block(x, param_vector)
        down3 = x
        x = self.down3_4(x)

        for block in self.NAFBlock_mid:
            x = block(x, param_vector)

        x = self.up4_3(x)
        x = torch.cat([x, down3], 1)
        x = self.channel_reduce3(x)
        for block in self.NAFBlock_dec3:
            x = block(x, param_vector)

        x = self.up3_2(x)
        x = torch.cat([x, down2], 1)
        x = self.channel_reduce2(x)
        for block in self.NAFBlock_dec2:
            x = block(x, param_vector)

        x = self.up2_1(x)
        x = torch.cat([x, down1], 1)
        x = self.channel_reduce1(x)
        for block in self.NAFBlock_dec1:
            x = block(x, param_vector)

        # long skip connection
        x = self.conv_out(x) + inp
        return x


class CPADNet_SIDD(nn.Module):
    def __init__(self, channels=32, num_blocks=[4, 4, 4, 8]):
        super(CPADNet_SIDD, self).__init__()
        self.phone2vector = nn.Embedding(num_embeddings=5, embedding_dim=9)

        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.NAFBlock_enc1 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels, param_dim=27) for i in range(num_blocks[0])])
        self.NAFBlock_enc2 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 2, param_dim=27) for i in range(num_blocks[1])])
        self.NAFBlock_enc3 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 4, param_dim=27) for i in range(num_blocks[2])])
        self.NAFBlock_mid = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 8, param_dim=27) for i in range(num_blocks[3])])
        self.NAFBlock_dec3 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 4, param_dim=27) for i in range(num_blocks[2])])
        self.NAFBlock_dec2 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 2, param_dim=27) for i in range(num_blocks[1])])
        self.NAFBlock_dec1 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels, param_dim=27) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(channels)
        self.down2_3 = Downsample(channels * 2)
        self.down3_4 = Downsample(channels * 4)

        self.up4_3 = Upsample(channels * 8)
        self.channel_reduce3 = nn.Conv2d(channels * 8, channels * 4, kernel_size=1, bias=False)
        self.up3_2 = Upsample(channels * 4)
        self.channel_reduce2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False)
        self.up2_1 = Upsample(channels * 2)
        self.channel_reduce1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers :
        for block in self.NAFBlock_enc1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_enc2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_enc3:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_mid:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_dec3:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_dec2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.NAFBlock_dec1:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, inp, param, phone):
        # getting parameters ready
        phone_vector = torch.squeeze(self.phone2vector(phone), dim=1)
        param_vector = torch.cat([phone_vector, param], dim=1)

        # network
        x = self.conv_in(inp)

        # NAFBlock + AdaLN
        for block in self.NAFBlock_enc1:
            x = block(x, param_vector)
        down1 = x
        x = self.down1_2(x)

        for block in self.NAFBlock_enc2:
            x = block(x, param_vector)
        down2 = x
        x = self.down2_3(x)

        for block in self.NAFBlock_enc3:
            x = block(x, param_vector)
        down3 = x
        x = self.down3_4(x)

        for block in self.NAFBlock_mid:
            x = block(x, param_vector)

        x = self.up4_3(x)
        x = torch.cat([x, down3], 1)
        x = self.channel_reduce3(x)
        for block in self.NAFBlock_dec3:
            x = block(x, param_vector)

        x = self.up3_2(x)
        x = torch.cat([x, down2], 1)
        x = self.channel_reduce2(x)
        for block in self.NAFBlock_dec2:
            x = block(x, param_vector)

        x = self.up2_1(x)
        x = torch.cat([x, down1], 1)
        x = self.channel_reduce1(x)
        for block in self.NAFBlock_dec1:
            x = block(x, param_vector)

        # long skip connection
        x = self.conv_out(x) + inp
        return x
