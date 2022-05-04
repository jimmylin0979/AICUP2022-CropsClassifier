#
import torch
import torch.nn as nn


class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name='saved_model.ckpt'):
        '''
        '''
        torch.save(self.state_dict(), name)
        return name
