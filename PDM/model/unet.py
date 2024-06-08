import math
import torch
from torch import nn
from torch.nn import functional as F

from .rrdb import RRDBNet


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            dropout: float = 0.1,
            max_len: int = 1000,
            apply_dropout: bool = True,
    ):
        """Section 3.5 of attention is all you need paper.

        Extended slicing method is used to fill even and odd position of sin, cos with increment of 2.
        Ex, `[sin, cos, sin, cos, sin, cos]` for `embedding_dim = 6`.

        `max_len` is equivalent to number of noise steps or patches. `embedding_dim` must same as image
        embedding dimension of the model.

        Args:
            embedding_dim: `d_model` in given positional encoding formula.
            dropout: Dropout amount.
            max_len: Number of embeddings to generate. Here, equivalent to total noise steps.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.apply_dropout = apply_dropout

        pos_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(start=0, end=max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim)

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(name='pos_encoding', tensor=pos_encoding, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """Get precalculated positional embedding at timestep t. Outputs same as video implementation
        code but embeddings are in [sin, cos, sin, cos] format instead of [sin, sin, cos, cos] in that code.
        Also batch dimension is added to final output.
        """
        positional_encoding = self.pos_encoding[t].squeeze(1)
        if self.apply_dropout:
            return self.dropout(positional_encoding)
        return positional_encoding


class DoubleConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int = None,
            residual: bool = False
    ):
        """Double convolutions as applied in the unet paper architecture.
        """
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=mid_channels, kernel_size=(3, 3), padding=(1, 1), bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=mid_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.double_conv(x))

        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor) -> torch.Tensor:
        """Downsamples input tensor, calculates embedding and adds embedding channel wise.

        If, `x.shape == [4, 64, 64, 64]` and `out_channels = 128`, then max_conv outputs [4, 128, 32, 32] by
        downsampling in h, w and outputting specified amount of feature maps/channels.

        `t_embedding` is embedding of timestep of shape [batch, time_dim]. It is passed through embedding layer
        to output channel dimentsion equivalent to channel dimension of x tensor, so they can be summbed elementwise.

        Since emb_layer output needs to be summed its output is also `emb.shape == [4, 128]`. It needs to be converted
        to 4D tensor, [4, 128, 1, 1]. Then the channel dimension is duplicated in all of `H x W` dimension to get
        shape of [4, 128, 32, 32]. 128D vector is sample for each pixel position is image. Now the emb_layer output
        is summed with max_conv output.
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_dim, out_features=out_channels),
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor, t_embedding: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_embedding)
        emb = emb.view(emb.shape[0], emb.shape[1], 1, 1).repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class TransformerEncoderSA(nn.Module):
    def __init__(self, num_channels: int, size: int, num_heads: int = 4):
        """A block of transformer encoder with mutli head self attention from vision transformers paper,
         https://arxiv.org/pdf/2010.11929.pdf.
        """
        super(TransformerEncoderSA, self).__init__()
        self.num_channels = num_channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm([num_channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=num_channels),
            nn.LayerNorm([num_channels]),
            nn.Linear(in_features=num_channels, out_features=num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self attention.

        Input feature map [4, 128, 32, 32], flattened to [4, 128, 32 x 32]. Which is reshaped to per pixel
        feature map order, [4, 1024, 128].

        Attention output is same shape as input feature map to multihead attention module which are added element wise.
        Before returning attention output is converted back input feature map x shape. Opposite of feature map to
        mha input is done which gives output [4, 128, 32, 32].
        """
        x = x.view(-1, self.num_channels, self.size * self.size).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(-1, self.num_channels, self.size, self.size).contiguous()

class ConditionEmbedding(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, emd_dim:int = 32) -> None:
        super().__init__()
        self.down1 = Down(input_dim, emd_dim)
        self.down2 = Down(emd_dim, output_dim)
    
    def forward(self, x, t):
        x1 = x
        x2 = self.down1(x1, t)
        return self.down2(x2, t)


class UNet(nn.Module):
    def __init__(
            self,
            cfg,
            image_size,
            num_channels,
            time_dim: int = 256,
    ):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        self.pos_encoding = PositionalEncoding(embedding_dim=time_dim, max_len=cfg.training.noise_steps)

        # RRDB
        num_down = int(cfg.training.scaling_factor ** 0.5)
        assert num_down > 0
        # self.condition_conv = RRDBNet(input_channel=condition_ch,
        #                                 num_feature=16, 
        #                                 output_channel=3,
        #                                 num_of_blocks=num_rrdb_blocks, 
        #                                 scaling_factor=scaling_factor)

        if cfg.group.type == 'Swin':
            self.extract_condition = RRDBNet(input_channel=cfg.rrdb.cond_channel,
                                            num_feature=cfg.rrdb.num_feature, 
                                            output_channel=num_channels,
                                            num_of_blocks=cfg.model.num_rrdb_blocks, 
                                            scaling_factor=cfg.training.scaling_factor)
            cond_proj1 = [nn.Conv2d(cfg.rrdb.num_feature * ((cfg.model.num_rrdb_blocks + 1)), 64, kernel_size=(3, 3), padding=(1, 1))]
            # ele_proj = [nn.GELU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))]
            ele_proj = [nn.GELU(), nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))]
            self.condition_proj = nn.Sequential(*(cond_proj1 + ele_proj * (num_down)))
            self.rrdb = True
        else:
            self.extract_condition = nn.Sequential(*([nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)), nn.GELU(), 
                                                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))]))
            self.condition_proj = nn.GELU()
            self.rrdb = False


        self.input_conv = DoubleConv(num_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = TransformerEncoderSA(128, int(image_size/2))
        self.down2 = Down(128, 256)
        self.sa2 = TransformerEncoderSA(256, int(image_size/4))
        self.down3 = Down(256, 256)
        self.sa3 = TransformerEncoderSA(256, int(image_size/8))

        self.bottleneck1 = DoubleConv(256, 512)
        self.bottleneck2 = DoubleConv(512, 512)
        self.bottleneck3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = TransformerEncoderSA(128, int(image_size/4))
        self.up2 = Up(256, 64)
        self.sa5 = TransformerEncoderSA(64, int(image_size/2))
        self.up3 = Up(128, 64)
        self.sa6 = TransformerEncoderSA(64, image_size)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.LongTensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass with image tensor and timestep reduce noise.

        Args:
            x: Image tensor of shape, [batch_size, channels, height, width].
            t: Time step defined as long integer. If batch size is 4, noise step 500, then random timesteps t = [10, 26, 460, 231].
        """

        t = self.pos_encoding(t)

        # condition for original pyramid diffusion
        # if condition is not None:
        #     assert x.shape == condition.shape
        #     x_in = x + condition
        # else:
        #     x_in = x

        x = self.input_conv(x)

        if condition is not None:
            if self.rrdb:
                _, condition = self.extract_condition(condition)
            else:
                condition = self.extract_condition(condition)
            condition = self.condition_proj(condition)
            x1 = x + condition
        else:
            x1 = x

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bottleneck1(x4)
        x4 = self.bottleneck2(x4)
        x4 = self.bottleneck3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        return self.out_conv(x)