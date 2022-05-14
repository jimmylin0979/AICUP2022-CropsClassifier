#
import fire

#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ExponentialLR

#
from torch_ema import ExponentialMovingAverage

#
import numpy as np

#
from sklearn.model_selection import train_test_split

#
from tqdm import tqdm

#
import os

#
import models
from data.dataset import OrchidDataSet
from config import DefualtConfig
from utils import get_confidence_score
from utils.losses import LabelSmoothingCrossEntropy
from utils.pseudo_label import get_pseudo_labels
from utils.scheduler import GradualWarmupScheduler

###################################################################################

config = DefualtConfig()
device = torch.device(f'cuda:{config.use_gpu_index}' if torch.cuda.is_available() else'cpu') if config.use_gpu_index != -1 else torch.device('cpu')

# Mapping
ds = OrchidDataSet(config.trainset_path)
idx_to_class = {}
for k in ds.class_to_idx:
    idx_to_class[ds.class_to_idx[k]] = k


def get_fileName(ds):

    fileNames = []

    for i in range(len(ds.imgs)):
        fileNames.append(ds.imgs[i][0])

    return fileNames


def test(output_file_path='predictions.csv'):
    '''
    @ Params:

    '''

    # Step 1 : Model Define & Load
    model = getattr(models, config.model_name)(config)
    model.to(device)
    model.load_state_dict(torch.load(config.model_path, map_location=device))

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema.load_state_dict(torch.load(config.ema_path, map_location=device))

    # Step 2 : DataSet & DataLoader
    ds_test = OrchidDataSet(config.testset_path)
    test_loader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Step 3 : Make prediction via trained model
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []

    with ema.average_parameters():

        # Iterate the validation set by batches.
        for batch in tqdm(test_loader):

            # A batch consists of image data and corresponding labels.
            imgs, _ = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            predictions += logits.argmax(dim=-1)

    # imgs_file_names = os.listdir(config.testset_path)
    imgs_file_names = get_fileName(ds_test)

    # Step 4 : Save predictions into the file.
    with open(output_file_path, "w") as f:

        # The first row must be "Id, Category"
        f.write("image_filename,label\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in  enumerate(predictions):
            ans = idx_to_class[pred.item()]
            if ans == "innudated":
                ans = "bardland"
            f.write(f"{imgs_file_names[i][-16:]},{ans}\n")

    # Step 5


###################################################################################

if __name__ == '__main__':

    # # parser = ArgumentParser(description='AICUP - Orchid Classifier')

    # # parser.add_argument('--lr', default=2e-5, type=float,
    # #                     help='Base learning rate')
    # # parser.add_argument('--bs', default=32, type=int, help='Batch size')
    # # parser.add_argument('--e', default=50, type=int, help='Numbers of epoch')
    # # parser.add_argument('--v', default=50, type=int, help='Experiment version')
    # # parser.add_argument('--device', default=-1, type=int,
    # #                     help='GPU index, -1 for cpu')

    # # args = parser.parse_args()

    # #
    # fire.Fire({
    #     'test' : test
    # })

    test()

    # Train model via command below :
    #       python main.py main --visualization=True

    # Inference model (test()) via command below :
    #       python main.py test --output_file_path=predictions.csv


###################################################################################