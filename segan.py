from pickle import FALSE
from turtle import forward
import torch


class NetSeg(torch.nn.Module):

    def __init__(self, input_channels=1, output_channels=3) -> None:
        super(NetSeg, self).__init__()

        self.encoder1 = NetSeg.block(in_channels=input_channels, features=32)
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = NetSeg.block(in_channels=32, features=64)
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = NetSeg.block(in_channels=64, features=128)
        self.pooling3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = NetSeg.block(in_channels=128, features=256)
        self.pooling4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = NetSeg.block(in_channels=256, features=512)

        self.upconv1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.decoder1 = NetSeg.block(in_channels=512, features=256)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder2 = NetSeg.block(in_channels=256, features=128)
        self.upconv3 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder3 = NetSeg.block(in_channels=128, features=64)
        self.upconv4 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.decoder4 = NetSeg.block(in_channels=64, features=32)
        self.output = torch.nn.Conv2d(in_channels=32, out_channels=output_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, sample):

        encoder1 = self.encoder1(sample)
        encoder2 = self.encoder2(self.pooling1(encoder1))
        encoder3 = self.encoder3(self.pooling2(encoder2))
        encoder4 = self.encoder4(self.pooling3(encoder3))

        bottleneck = self.bottleneck(self.pooling4(encoder4))

        decoder1 = self.decoder1(torch.cat((self.upconv1(bottleneck), encoder4), dim=1))
        decoder2 = self.decoder2(torch.cat((self.upconv2(decoder1), encoder3), dim=1))
        decoder3 = self.decoder3(torch.cat((self.upconv3(decoder2), encoder2), dim=1))
        decoder4 = self.decoder4(torch.cat((self.upconv4(decoder3), encoder1), dim=1))

        return torch.softmax(self.output(decoder4), dim=1)

    @staticmethod
    def block(in_channels, features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding='same', bias=False),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding='same', bias=False),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(inplace=True),  
        )

class NetDis(torch.nn.Module):
    
    def __init__(self, input_channels=1) -> None:
        super(NetDis, self).__init__()

        #self.encoder1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
        #                torch.nn.LeakyReLU(inplace=True)
        #                )
        #self.encoder2 = NetDis.block(in_channels=64, features=128)
        #self.encoder3 = NetDis.block(in_channels=128, features=256)
        #self.encoder4 = NetDis.block(in_channels=256, features=512)
        #self.encoder5 = NetDis.block(in_channels=512, features=1024)

        self.encoder1 = NetSeg.block(in_channels=input_channels, features=32)
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = NetSeg.block(in_channels=32, features=64)
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = NetSeg.block(in_channels=64, features=128)
        self.pooling3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = NetSeg.block(in_channels=128, features=256)
        self.pooling4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, sample):
        #encoder1 = self.encoder1(sample)
        #encoder2 = self.encoder2(encoder1)
        #encoder3 = self.encoder3(encoder2)
        #encoder4 = self.encoder4(encoder3)
        #encoder5 = self.encoder5(encoder4)
        
        encoder1 = self.encoder1(sample)
        encoder2 = self.encoder2(self.pooling1(encoder1))
        encoder3 = self.encoder3(self.pooling2(encoder2))
        encoder4 = self.encoder4(self.pooling3(encoder3))

        return torch.cat((sample.flatten(start_dim=1), encoder1.flatten(start_dim=1), encoder2.flatten(start_dim=1), encoder3.flatten(start_dim=1), encoder4.flatten(start_dim=1)), dim=1)

    @staticmethod
    def block(in_channels, features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding='same', bias=False),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding='same', bias=False),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(inplace=True),  
        )

    #@staticmethod
    #def block(in_channels, features):
    #    return torch.nn.Sequential(
    #        torch.nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=4, stride=2, padding=1, bias=False),
    #        torch.nn.BatchNorm2d(num_features=features),
    #        torch.nn.LeakyReLU(inplace=True)
    #    )