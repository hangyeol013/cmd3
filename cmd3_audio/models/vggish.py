import numpy as np
import torch
import torch.nn as nn
from torch import hub


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())



class VGGish(VGG):
    def __init__(self, path, num_classes, feature_extraction):
        super().__init__(make_layers())
        if path:
            state_dict = torch.load(path)
            super().load_state_dict(state_dict)

            if feature_extraction:
                for param in self.parameters():
                    param.requires_grad = False

            self.classifier = nn.Linear(128, num_classes)

        else:
            
            self.classifier = nn.Linear(128, num_classes)


    def forward(self, x, fs=None):
        
        x = self.features(x)
        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        x = self.classifier(x)


        return x


if __name__ == "__main__":
    model = VGGish()
    print(model)
