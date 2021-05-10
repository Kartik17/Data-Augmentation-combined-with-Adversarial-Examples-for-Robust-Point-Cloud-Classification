import torch.nn as nn

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.0)