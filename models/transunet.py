from turtle import forward

from numpy import identity
from einops import rearrange, repeat
import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads) -> None:
        
        """_summary_

        Args:
            embedding_dim (_type_): _description_

            num_heads (_type_): _description_
        """
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads

        self.query = torch.nn.Linear(embedding_dim, self.head_dim*self.num_heads, bias=False)
        self.key = torch.nn.Linear(embedding_dim, self.head_dim*self.num_heads, bias=False)
        self.value = torch.nn.Linear(embedding_dim, self.head_dim*self.num_heads, bias=False)

        self.out = torch.nn.Linear(self.num_heads*self.head_dim, embedding_dim, bias=False)
        self.scale = self.head_dim ** (1/2)
        
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.scale

        attention = torch.nn.Softmax()(energy)
        out = torch.einsum("... i j , ... j d -> ... i d", attention, value)
        out = self.out(out)
        return out

class TransformerBlock(torch.nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout, linear_dim) -> None:
        """_summary_

        Args:
            embedding_dim (_type_): _description_

            num_heads (_type_): _description_

            dropout (_type_): _description_

            linear_dim (_type_): _description_
        """
        super().__init__()

        self.mhsa = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)

        self.dropout = torch.nn.Dropout(dropout)
        self.norm_1 = torch.nn.LayerNorm(embedding_dim)
        self.norm_2 = torch.nn.LayerNorm(embedding_dim)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, linear_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(linear_dim, embedding_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        x1 = x
        x = self.mhsa(x)
        x = self.dropout(x)
        x = x + x1

        x = self.norm_1(x)
        x2 = x
        x = self.linear(x)
        x = x + x2
        out = self.norm_2(x)
        
        return out

class TransformerEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, linear_dim, num_blocks, num_heads, dropout) -> None:
        """_summary_

        Args:
            embedding_dim (_type_): _description_

            linear_dim (_type_): _description_

            num_blocks (_type_): _description_

            num_heads (_type_): _description_

            dropout (_type_): _description_
        """
        super(TransformerEncoder, self).__init__()

        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout, linear_dim=linear_dim) for i in range(num_blocks)]
        )
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ViT(torch.nn.Module):
    def __init__(self, img_dim, in_channels, patch_dim, embedding_dim, num_blocks, num_heads, linear_dim, dropout) -> None:
        """_summary_

        Args:
            img_dim (_type_): _description_

            in_channels (_type_): _description_

            patch_dim (_type_): _description_

            embedding_dim (_type_): _description_

            num_blocks (_type_): _description_

            num_heads (_type_): _description_

            linear_dim (_type_): _description_

            dropout (_type_): _description_
        """
        super(ViT, self).__init__()
        
        self.num_tokens = (img_dim//patch_dim)**2
        self.patch_dim = patch_dim
        self.token_dim = (self.patch_dim**2)*in_channels

        self.project = torch.nn.Linear(self.token_dim, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

        self.cls_token = torch.nn.Parameter(torch.rand(1, 1, embedding_dim))
        self.embedding = torch.nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))
        
        self.transformer = TransformerEncoder(embedding_dim=embedding_dim, linear_dim=linear_dim, num_blocks=num_blocks, num_heads=num_heads, dropout=dropout)

        self.final = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        patches = rearrange(
            x, 
            "b c (x_patch x) (y_patch y) -> b (x y) (c x_patch y_patch)",
            x_patch = self.patch_dim,
            y_patch = self.patch_dim
        )
        batch_size = patches.shape[0]
        patches = self.project(patches)

        cls_token = repeat(
            self.cls_token, "b ... -> (b batch_size) ...", batch_size=batch_size
        )
        patches = torch.cat([cls_token, patches], dim=1) + self.embedding

        out = self.dropout(patches)
        out = self.transformer(out)
        out = self.final(out[:, 1:, :])
        return out

class BottleNeckUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        """_summary_

        Args:
            in_channels (_type_): _description_

            out_channels (_type_): _description_
        """
        super(BottleNeckUnit, self).__init__()

        self.downsample = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.out = torch.nn.ReLU()
    
    def forward(self, x):
        x_down = self.downsample(x)
        x = self.layer(x)
        x = x + x_down
        x = self.out(x)
        return x


class TransUnetEncoder(torch.nn.Module):

    def __init__(self, img_dim, init_features, patch_dim, in_channels, embedding_dim, num_blocks, num_heads, linear_dim, dropout) -> None:
        """_summary_

        Args:
            img_dim (_type_): _description_

            init_features (_type_): _description_

            patch_dim (_type_): _description_

            in_channels (_type_): _description_

            embedding_dim (_type_): _description_

            num_blocks (_type_): _description_

            num_heads (_type_): _description_

            linear_dim (_type_): _description_

            dropout (_type_): _description_
        """
        super(TransUnetEncoder, self).__init__()

        self.features = init_features
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        
        self.layer1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=self.features, kernel_size=7, stride=1, padding=3, bias=False)
        self.layer2 = BottleNeckUnit(self.features, self.features*2)
        self.layer3 = BottleNeckUnit(self.features*2, self.features*4)
        self.layer4 = BottleNeckUnit(self.features*4, self.features*8)
        self.layer5 = ViT(img_dim = self.img_dim // self.patch_dim, in_channels=self.features*8, patch_dim=1, embedding_dim=embedding_dim, num_blocks=num_blocks, num_heads=num_heads, linear_dim=linear_dim, dropout=dropout)
        self.layer6 = torch.nn.Conv2d(self.features*8, out_channels=self.features*4, kernel_size=3, stride=1, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(num_features=self.features*4)
        self.out = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.layer4(x3)
        x = self.layer5(x)

        x = rearrange(
            x,
            "b (x y) c -> b c x y",
            x = self.img_dim // self.patch_dim,
            y = self.img_dim // self.patch_dim
        )
        x = self.layer6(x)
        x = self.batchnorm(x)
        x = self.out(x)
        return x, x1, x2, x3

class TransUnetDecoderUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:

        """_summary_

        Args:
            in_channels (_type_): _description_
            
            out_channels (_type_): _description_

        """
        super(TransUnetDecoderUnit, self).__init__()

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(inplace=True)
        )

    def forward(self, x, x_skip=None):

        x = self.upsample(x)
        if x_skip is not None:
            x = torch.cat([x_skip, x], dim=1)
        
        out = self.layer(x)

        return out

class TransUnet(torch.nn.Module):
    def __init__(self, img_dim=128, patch_dim=8, embedding_dim=512, init_features=64, in_channels=1, classes=3, num_blocks=12, num_heads=12, linear_dim=3072, dropout=0.1) -> None:
        
        """_summary_
        
        Args:
            img_dim (int, optional): size of the input dimension for the model. Defaults to 128.

            patch_dim (int, optional): size of the patches. Defaults to 16.

            embedding_dim (int, optional): _description_. Defaults to 512.

            init_features (int, optional): number of initial features for the encoder. Defaults to 64.

            in_channels (int, optional): number of channels in the input image. Defaults to 1.

            classes (int, optional): number of classes to segment. Defaults to 3.

            num_blocks (int, optional): number of attention blocks in the transformer module. Defaults to 12.

            num_heads (int, optional): number of heads in the attention module. Defaults to 12.

            linear_dim (int, optional): _description_. Defaults to 3072.

            dropout (float, optional): percentage of neurons to deactivate during training. Defaults to 0.1.

        """
        super(TransUnet, self).__init__()

        self.encoder = TransUnetEncoder(img_dim=img_dim, init_features=init_features, patch_dim=patch_dim, in_channels=in_channels, embedding_dim=embedding_dim, num_blocks=num_blocks, num_heads=num_heads, linear_dim=linear_dim, dropout=dropout)

        self.decoder1 = TransUnetDecoderUnit(in_channels=init_features*8, out_channels=init_features*2)
        self.decoder2 = TransUnetDecoderUnit(in_channels=init_features*4, out_channels=init_features)
        self.decoder3 = TransUnetDecoderUnit(in_channels=init_features*2, out_channels=int(init_features * 1/2))
        #self.decoder4 = TransUnetDecoderUnit(in_channels=int(init_features * 1/2), out_channels=int(init_features * 1/8))

        self.conv = torch.nn.Conv2d(in_channels=int(init_features * 1/2), out_channels=classes, kernel_size=1, bias=False)

    def forward(self, x):
        
        x, x1, x2, x3 = self.encoder(x)

        out = self.decoder1(x, x3)
        out = self.decoder2(out, x2)
        out = self.decoder3(out, x1)
        #out = self.decoder4(out)
        out = self.conv(out)

        return torch.softmax(out, dim=1)