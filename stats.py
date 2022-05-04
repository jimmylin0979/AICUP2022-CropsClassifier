#
import fire

#
import torch
import torch.nn as nn
#
import numpy as np

#
from tqdm import tqdm

#
import models
from config import DefualtConfig

#
from fvcore.nn import FlopCountAnalysis

###################################################################################

config = DefualtConfig()
device = torch.device(f'cuda:{config.use_gpu_index}' if torch.cuda.is_available(
) else'cpu') if config.use_gpu_index != -1 else torch.device('cpu')


def main():

    model = getattr(models, config.model_name)(config)
    model.to(device)

    flops = FlopCountAnalysis(model, torch.zeros((1, 3, 224, 224)).to(device))
    print(f'FLOPs : {flops.total()}')
    print(f'Operators : \n{flops.by_operator()}')


if __name__ == '__main__':
    main()
