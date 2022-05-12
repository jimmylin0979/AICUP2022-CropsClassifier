#
import torch
import torch.nn as nn
import torch.nn.functional as F

#
import numpy as np

# Hugging face API
from transformers import ViTFeatureExtractor, ViTModel

from models import BaseModule
from config import DefualtConfig


class ViT(nn.Module):

    def __init__(self, config: DefualtConfig):

        super(ViT, self).__init__()

        self.config = config
        self.num_labels = config.num_classes

        ###############################################
        # input image with shape (batch_size, 3, 224, 224)
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7),    # (32, 215, 215)
            nn.Conv2d(32, 32, kernel_size=7),    # (32, 209, 209)
            nn.MaxPool2d(2, stride=2),          # (32, 104, 104)
            nn.ReLU(True),

            nn.Conv2d(32, 32, kernel_size=7),   # (32, 98, 98)
            nn.MaxPool2d(2, stride=2),          # (32, 46, 46)
            nn.ReLU(True),

            nn.Conv2d(32, 32, kernel_size=5),   # (32, 42, 42)
            nn.MaxPool2d(2, stride=2),          # (32, 20, 20)
            nn.ReLU(True),

            nn.Conv2d(32, 32, kernel_size=5),   # (32, 8, 8)
            nn.MaxPool2d(2, stride=2),          # (32, 4, 4)
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 90),
            nn.ReLU(True),
            nn.Linear(90, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        ###############################################
        # ViT
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            config.pretrained_model)
        self.vit = ViTModel.from_pretrained(config.pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.vit.config.hidden_size, self.num_labels)

    def feature_extract(self, imgs):
        '''
        '''
        # Change input array into list with each batch being one element

        # convert it to numpy array first
        device = torch.device('cpu')
        if imgs.device != torch.device('cpu'):
            device = torch.device(f'cuda:{self.config.use_gpu_index}')

        imgs = imgs.cpu().numpy()
        imgs = np.split(np.squeeze(np.array(imgs)), imgs.shape[0])

        # Remove unecessary dimension
        for index, array in enumerate(imgs):
            imgs[index] = np.squeeze(array)

        # Apply feature extractor, stack back into 1 tensor and then convert to tensor
        # imgs = (batch_size, 3, 224, 224)
        imgs = torch.tensor(
            np.stack(self.feature_extractor(imgs)['pixel_values'], axis=0))
        imgs = imgs.to(device)

        return imgs

    def stn(self, x):
        '''
        Spatial transformer network forward function
        '''
        xs = self.localization(x)
        xs = torch.reshape(xs, (-1, 32 * 9 * 9))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x, labels=None):
        '''
        Model forward function
        '''

        # Feature extraction
        x = self.feature_extract(x)

        # # Spatial Transformer
        # x = self.stn(x)

        # ViT
        x = self.vit(pixel_values=x)
        x = self.dropout(x.last_hidden_state[:, 0])
        logits = self.classifier(x)

        # # loss = None
        # # if labels is not None:
        # #     loss_fct = nn.CrossEntropyLoss()
        # #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # # if loss is not None:
        # #     return logits, loss.item()
        # # else:
        # #     return logits, None

        return logits
