import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable

class BackBone(torch.nn.Module):
    def __init__(self, backbone_model, input_dim, rep_dim, pretrained, device):
        super(BackBone, self).__init__()
        assert 'resnet' in backbone_model #Only resnet is supported for now

        self.model = models.__dict__[backbone_model](pretrained=pretrained)

        if input_dim[0] != 3: #perform model surgery. Need to change the first layer to take as input 5 channels instead of the standard 3
            old_conv1_weight = self.model.conv1.weight.clone()
            new_conv1 = nn.Conv2d(input_dim[0], 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False).requires_grad_(True)
            new_conv1.weight[:, :3, :, :].data = Variable(old_conv1_weight, requires_grad=True)
            self.model.conv1 = new_conv1
        
        #final layer is a linear layer with a rep_dim dimensions
        self.model.fc = nn.LazyLinear(rep_dim)
        self.model.forward(torch.randn(2, *input_dim))
        
        #push model to device
        self.model.to(device)

    def forward(self, x):
        return self.model(x)


class Projector(torch.nn.Module):
    def __init__ (self, rep_dim, hidden_dim, output_dim, device):
        super(Projector, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        self.model.to(device)
    
    def forward(self, x):
        return self.model(x)



class WhaleDoModel(torch.nn.Module):
    def __init__(self, config):
        super(WhaleDoModel, self).__init__()

        self.config = config
        self.backbone = BackBone(config['backbone_model'], config['input_dim'], config['rep_dim'], config['pretrained'], config['device'])
        
        if config['projector'] is not None:
            self.projector = Projector(config['rep_dim'], config['projector']['hidden_dim'], config['projector']['output_dim'], config['device'])
        else:
            self.projector = None
        
    def forward(self, x):
        x = self.backbone(x)
        if self.projector is not None:
            x = self.projector(x)
        return x


'''
Example config

config = {
    'backbone_model': 'resnet18',
    'input_dim': (3, 224, 224),
    'rep_dim': 512,
    'pretrained': True,
    'device': 'cuda',

    'projector' : {
        'hidden_dim': 1024,
        'output_dim': 2
    }
}

'''

        
        


