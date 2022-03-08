import torch
from collections import OrderedDict

class UNET(torch.nn.Module):

    def __init__(self, input_channels=1, output_channels=1, init_features=32) -> None:
        super(UNet, self).__init__()

        features = init_features

        self.encoder_1 = UNet._block(input_channels, features, name='encoder_1')
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_2 = UNet._block(features, features*2, name='encoder_2')
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_3 = UNet._block(features*2, features*4, name='encoder_3')
        self.pool_3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_4 = UNet._block(features*4, features*8, name='encoder_4')
        self.pool_4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features*8, features*16, name='bottleneck')

        self.upconv_4 = torch.nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder_4 = UNet._block((features*8)*2, features*8, name='decoder_4')
        self.upconv_3 = torch.nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder_3 = UNet._block((features*4)*2, features*4, name='decoder_3')
        self.upconv_2 = torch.nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder_2 = UNet._block((features*2)*2, features*2, name='decoder_2')
        self.upconv_1 = torch.nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder_1 = UNet._block(features*2, features, name='decoder_1')
        self.conv = torch.nn.Conv2d(features, output_channels, kernel_size=1)

    def forward(self, sample):
        encoder_1 = self.encoder_1(sample)
        encoder_2 = self.encoder_2(self.pool_1(encoder_1))
        encoder_3 = self.encoder_3(self.pool_2(encoder_2))
        encoder_4 = self.encoder_4(self.pool_3(encoder_3))

        bottleneck = self.bottleneck(self.pool_4(encoder_4))

        decoder_4 = self.upconv_4(bottleneck)
        decoder_4 = torch.cat((decoder_4, encoder_4), dim=1)
        decoder_4 = self.decoder_4(decoder_4)
        decoder_3 = self.upconv_3(decoder_4)
        decoder_3 = torch.cat((decoder_3, encoder_3), dim=1)
        decoder_3 = self.decoder_3(decoder_3)
        decoder_2 = self.upconv_2(decoder_3)
        decoder_2 = torch.cat((decoder_2, encoder_2), dim=1)
        decoder_2 = self.decoder_2(decoder_2)
        decoder_1 = self.upconv_1(decoder_2)
        decoder_1 = torch.cat((decoder_1, encoder_1), dim=1)
        decoder_1 = self.decoder_1(decoder_1)

        return torch.sigmoid(self.conv(decoder_1))


    @staticmethod
    def _block(in_channels, features, name):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", torch.nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", torch.nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", torch.nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", torch.nn.ReLU(inplace=True)),
                ]
            )
        )