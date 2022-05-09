# 
import fire

#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 
import numpy as np

#
from sklearn.model_selection import train_test_split

#
from tqdm import tqdm

#
import models
from data.dataset import OrchidDataSet
from config import DefualtConfig
from utils import get_confidence_score

###################################################################################

config = DefualtConfig()
device = torch.device(f'cuda:{config.use_gpu_index}' if torch.cuda.is_available() else'cpu') if config.use_gpu_index != -1 else torch.device('cpu')

def main(**kwargs):

    # Step 1 : prepare logging writer
    writer = SummaryWriter()

    # Step 2 : 
    model = getattr(models, config.model_name)(config)
    model.to(device)

    # 
    writer.add_graph(model, torch.zeros((1, 3, 224, 224)).to(device))
    writer.flush()

    # Step 3 : DataSets
    ds = OrchidDataSet(config.trainset_path)

    # Step 3
    # Deal with imbalance dataset
    #   For the classification task, we use cross-entropy as the measurement of performance.
    #   Since the wafer dataset is serverly imbalance, we add class weight to make it classifier better
    class_weights = [1 - (ds.targets.count(c))/len(ds) for c in range(config.num_classes)]
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    lambda0 = lambda cur_iter: (cur_iter / config.lr_warmup_epoch)* config.lr if  cur_iter < config.lr_warmup_epoch else config.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

    # Step 4
    train_loader, valid_loader = get_loader(ds)

    # Step 5
    history = {'train_acc' : [], 'train_loss' : [], 'valid_acc' : [], 'valid_loss' : []}
    best_epoch, best_loss = 0, 1e100
    nonImprove_epochs = 0

    for epoch in tqdm(range(config.num_epochs)):

        # 
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, scheduler)
        print(f"[ Train | {epoch + 1:03d}/{config.num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        
        # 
        valid_acc, valid_loss = valid(model, valid_loader, criterion)
        print(f"[ Valid | {epoch + 1:03d}/{config.num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        # Append the training statstics into history
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        # Tensorboard Visualization
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("valid_acc", valid_acc, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("valid_loss", valid_loss, epoch)

        # EarlyStop
        # if the model improves, save a checkpoint at this epoch
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), config.model_path)
            print(f'Saving model with loss {valid_loss:.4f}'.format(valid_loss))
            nonImprove_epochs = 0
        else:
            nonImprove_epochs += 1

        # Stop training if your model stops improving for "config['early_stop']" epochs.    
        if nonImprove_epochs >= config.earlyStop_interval:
            break

    writer.flush()
    writer.close()

    # Step 6 : Explanation & Visualization
    get_confidence_score(model, loader=valid_loader, use_gpu_index=config.use_gpu_index, batch_size=config.batch_size)

###################################################################################

def get_loader(ds):

    # Split the train/test with each class should appear on both train/test dataset
    valid_split = config.train_valid_split

    indices = list(range(len(ds)))  # indices of the dataset
    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_split, stratify=ds.targets)

    # Creating sub dataset from valid indices
    # Do not shuffle valid dataset, let the image in order
    valid_indices.sort()
    ds_valid = torch.utils.data.Subset(ds, valid_indices)

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    # Construct data loaders.
    train_loader = DataLoader(
        ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers, pin_memory=True)
    valid_loader = DataLoader(
        ds_valid, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_loader, valid_loader


def train(model, train_loader, criterion, optimizer, scheduler=None):
    
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        # imgs = (batch_size, 3, 224, 224)
        # labels = (batch_size)
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # # STN : Allow transformes to do like translation, cropping, isotropic scaling but rotation
        # #           , with a intention to let STN learns where to focus on instead of how to transform the image.
        # # Below is what matrix should look like : 
        # #    [ x_ratio, 0 ] [offset_X]
        # #    [ 0, y_ratio ] [offset_y]
        # model.fc_loc[-1].weight.grad[1].zero_()
        # model.fc_loc[-1].weight.grad[3].zero_()

        # Update the parameters with computed gradients.
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # # Clip the gradient norms for stable training.
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    return acc.item(), loss.item()

def valid(model, valid_loader, criterion):
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    return acc.item(), loss.item()


def test(output_file_path='predictions.csv'):
    '''
    @ Params:
    
    '''

    # Step 1 : Model Define & Load
    model = getattr(models, config.model_name)(config)
    model.to(device)
    model.load_state_dict(torch.load(config.model_path, map_location=device))

    # Step 2 : DataSet & DataLoader
    ds_test = OrchidDataSet(config.testset_path)
    test_loader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Step 3 : Make prediction via trained model
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []

    # Iterate the validation set by batches.
    for batch in tqdm(test_loader):

        # A batch consists of image data and corresponding labels.
        imgs, _ = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        predictions += logits.argmax(dim=-1)

    # Step 4 : Save predictions into the file.
    with open(output_file_path, "w") as f:

        # The first row must be "Id, Category"
        f.write("Id,Category\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in  enumerate(predictions):
            f.write(f"{i},{pred}\n")

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

    #
    fire.Fire({
        'main' : main, 
        'test' : test
    })

    # Train model via command below : 
    #       python main.py main --visualization=True

    # Inference model (test()) via command below : 
    #       python main.py test --output_file_path=predictions.csv


###################################################################################