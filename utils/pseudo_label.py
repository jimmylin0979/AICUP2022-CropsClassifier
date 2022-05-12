import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import DefualtConfig
from data import TensorIntDataset

import gc

###################################################################################

config = DefualtConfig()


def get_pseudo_labels(model, *datasets, threshold=0.75):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # it costs too more time, uncomment here if you have enough time or have a string GPU
    # # combines unlabeled_set & test_sets
    # unlabeled, test = datasets
    # dataset = ConcatDataset([unlabeled, test])
    dataset = datasets[0]

    # Construct a data loader.
    data_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    # temporary variables
    maxConfidence, pseudo_label = None, None
    masks, img = None, None

    # input to dataloader
    to_train_x, to_train_y = torch.tensor([]), []
    cnt = 0

    # Iterate over the dataset by batches.
    for batch in tqdm(data_loader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        # dimension [128, 11]
        probs = softmax(logits)
        # probs = probs.cpu().detach().numpy()

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.

        maxConfidence, pseudo_label = torch.max(probs, 1)

        # create a mask & delete non-fit datas
        masks = (maxConfidence >= threshold)
        img = img[masks]
        pseudo_label = pseudo_label[masks]

        # append result of each batch to all
        to_train_x = torch.cat((to_train_x, img), 0)
        for label in (pseudo_label):
            to_train_y.append(int(label.item()))

    cnt = len(to_train_y)

    # # Turn off the eval mode.
    model.train()

    #
    if cnt != 0:
        print(f"[ {cnt} Unlabeled Images append into train_set ]")

        # to_train_x = torch.tensor(to_train_x)
        # to_train_y = torch.tensor(to_train_y, dtype=torch.int)

        # # reshape
        to_train_x = torch.reshape(to_train_x, (-1, 3, 128, 128))
        # to_train_y = torch.reshape(to_train_y, (-1,))

        print(to_train_x.shape)
        print(len(to_train_y))

        # transfer list of Tensor into dataSet (TensorDataset)
        # res_dataset = TensorDataset(to_train_x, to_train_y)
        res_dataset = TensorIntDataset(to_train_x, to_train_y)

        # free the resources, or it will collapse eventually
        del maxConfidence, pseudo_label, masks, img
        del to_train_x, to_train_y
        gc.collect()

        return res_dataset

    return None
