

import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)


vgg_model = vgg16(pretrained=True)
if torch.cuda.is_available():
    vgg_model.cuda()
loss_network = LossNetwork(vgg_model)
loss_network.eval()


def perceptual_loss(x,y):

    cs = nn.CosineSimilarity(dim=1)
    v1 = loss_network(x.to('cuda'))
    v2 = loss_network(y.to('cuda'))
    score = 0

    for i in range(len(v1)):
        score += cs(v1[i].flatten(1),v2[i].flatten(1)) / len(v1)

    return score.item()