import sys

sys.path.append("..")
from models._unet import UNet_6, UNet_5
from models._mmdensenet import MMDenseNet
from models._mdensenet import MDenseNet
from torch import nn


class dedicated_model(nn.Module):
    def __init__(self,
                 model_name,
                 device,
                 inchannels,
                 outchannels,
                 sources=2,
                 drop_rate = 0.1
                 ):
        super(dedicated_model, self).__init__()
        self.sources = sources
        self.model_name = model_name
        self.device = device
        self.cnt = 0
        self.sigmoid = nn.Sigmoid()
        for channel in range(self.sources):
            if self.model_name == "Unet-5":
                model = UNet_5(n_channels=inchannels, n_classes=outchannels,dropout=drop_rate)
            elif self.model_name == "Unet-6":
                model = UNet_6(n_channels=inchannels, n_classes=outchannels,dropout=drop_rate)
            elif self.model_name == "MMDenseNet":
                model = MMDenseNet(input_channel=inchannels,drop_rate=drop_rate)
            elif self.model_name == "MDenseNet":
                model = MDenseNet(in_channel=inchannels, out_channel=inchannels,drop_rate=drop_rate)
            else:
                raise ValueError("Error: Non-exist model name")
            if 'cpu' in str(device):
                exec("self.unet{}=model".format(channel))
            else:
                exec("self.unet{}=model.cuda(".format(channel)+"\""+str(self.device)+"\")")

    def forward(self, track_i, zxx):
        layer = self.__dict__['_modules']['unet' + str(track_i)]
        out = layer(zxx)
        return self.sigmoid(out)
