import sys
sys.path.append('..')
from models.chetpose import resnet101
from models.darknet import Darknet
import torch

def build_model(config):

    if config.model_type == 'resnet-base':
        model = resnet101(config.num_kpt, pretrained=True)
    elif config.model_type == 'darknet-base':
        model = Darknet('models/network.cfg')
        model.print_network()

    if config.cuda:
        model = model.cuda()

    if config.pre_trained:
        ignore_keys = []
        model_dict = model.state_dict()
        if config.cuda:
            pretrained_dict = torch.load(config.model_path)
        else:
            pretrained_dict = torch.load(config.model_path, map_location='cpu')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)  

    return model
