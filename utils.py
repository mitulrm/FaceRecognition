# Name : Mitul Modi
# mitulraj@buffalo.edu

import sys
import os
import os.path
import csv
import time
import PIL
import math
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tqdm 

# Check if GPU is available
gpu = torch.cuda.is_available()

def save_model(filename, model, optimizer, scheduler, epoch, loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist, early_stop_counter):
    """
        Function to save model.
        
        Function saves model and other training related information so that it can be loaded later to resume training or for inference.
        It is called by fit() function to save best model during training.
    """
    state_dict = {
        'epoch':epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_tr_hist': loss_tr_hist,
        'loss_val_hist': loss_val_hist,
        'accuracy_tr_hist': accuracy_tr_hist,
        'accuracy_val_hist': accuracy_val_hist,
        'early_stop_counter': early_stop_counter
    }
    torch.save(state_dict, filename)

def load_model(filename, model, optimizer = None, scheduler = None, mode = 'test'):
    """
        This function loads previously saved model and its related training details from file specified by filename.
        
        Parameters:
            filename : path of saved model file.
            model : Instance of model to be loaded.
            optimizer : Instance of optimizer to be loaded to previous saved state. Useful to resume training of model from saved state.
            scheduler : Instance of scheduler to be loaded to previous saved state. Useful to resume training of model from saved state.
            mode : Values should be 'train' or 'test'. If value is 'train', it returns model and all other information required to resume training from saved state.
                   If value is 'test', it loads and returns only model.
    """
    state_dict = torch.load(filename)

    model.load_state_dict(state_dict['model'])
    if mode == 'test':
        return model

    epoch = state_dict['epoch']
    optimizer.load_state_dict(state_dict['optimizer'])
    loss_tr_hist = state_dict['loss_tr_hist']
    loss_val_hist = state_dict['loss_val_hist']
    accuracy_tr_hist = state_dict['accuracy_tr_hist']
    accuracy_val_hist = state_dict['accuracy_val_hist']
    early_stop_counter = state_dict['early_stop_counter']
    if scheduler is not None:
        scheduler.load_state_dict(state_dict['scheduler'])

    return epoch, model, optimizer, scheduler, early_stop_counter, loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist
	
class Dataset_Train(Dataset):
    """
        Customized Dataset class to load training and validation dataset.

        Code is adapted from code of pytorch sourcecode of DatasetFolder class.
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.
    """
    def __init__(self, img_dir, protocol_dir, split_n, transform):

        self.transform = transform

        self.img_dir = os.path.expanduser(img_dir)
        self.protocol_dir = os.path.expanduser(protocol_dir)
        train_file = os.path.join(protocol_dir,'split' + str(split_n), 'train' + '_' + str(split_n) + '.csv')
        train_df = pd.read_csv(train_file, delimiter = ',', usecols = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE'])
        train_df['FILE'] = self.img_dir + '/' + train_df['SUBJECT_ID'].map(str) + '/' + train_df['FILE']
        train_df = train_df[train_df['FILE'].map(lambda x:os.path.isfile(x))]

        subject_ids = train_df['SUBJECT_ID'].unique()
        self.class_to_idx = {subject_ids[i]:i for i in range(subject_ids.shape[0])}
        train_df['SUBJECT_ID'] = train_df['SUBJECT_ID'].apply(lambda x:self.class_to_idx[x])
        self.labels = list(train_df['SUBJECT_ID'])
        self.images = list(train_df['FILE'])

    def __len__(self):
        # return size of dataset
        return len(self.images)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.images[idx])  # PIL image
        image = image.convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def get_labels(self):
        return self.labels

    def get_class_to_idx(self):
        return self.class_to_idx
		
class Dataset_Test(Dataset):
    """
        Customized Dataset class to load training and validation dataset.

        Code is adapted from code of pytorch sourcecode of DatasetFolder class.
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.
    """
    def __init__(self, image_list, transform):

        self.transform = transform
        self.image_list = image_list

    def __len__(self):
        # return size of dataset
        return len(self.image_list)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.image_list[idx])  # PIL image
        image = image.convert('RGB')
        image = self.transform(image)
        return image
		
		
def train(dataloader, model, optimizer, criterion):
    """
        Function to perform training step.
        
        This function performs primary step of training. It performs forward and backward pass to update model parameters. It is called by fit() function during each epoch. 
        Returns loss and accuracy for current epoch.
    """
    batch = 0
    loss = 0.0
    correct = 0.0

    model.train()    
    
    for X, Y in dataloader:
        if gpu:
            X = X.to('cuda', non_blocking=True)
            Y = Y.to('cuda', non_blocking=True)
        optimizer.zero_grad()        
        logits = model(X)
        cur_loss = criterion(logits, Y)
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        pred = logits.argmax(dim = 1)
        correct += pred.eq(Y).sum()

        # Display Progres Bar. 
        # Reference - https://stackoverflow.com/questions/46141302/how-to-make-a-still-progress-in-python/46141777
        batch += 1
        completed = math.floor(batch * dataloader.batch_size / len(dataloader.dataset) * 50)
        print('\r' + 'Training: ' + '▮' * completed + '▯' * (50-completed) + str(completed*2) + '%', end='')
    
    print('\r', end='')
    
    loss = loss / float(len(dataloader.dataset))
    accuracy = float(correct) / float(len(dataloader.dataset)) * 100
    
    return loss, accuracy
	
def validate(dataloader, model, criterion):
    """
        Function to perform validation step.
        
        This function is used to perform validation of a model. It is called by function fit() during each epoch.
        Returns validation loss and accuracy for current epoch.
    """

    batch = 0    
    loss = 0.0
    correct = 0.0
    
    model.eval()
    
    for X, Y in dataloader:
        if gpu:
            X = X.to('cuda', non_blocking=True)
            Y = Y.to('cuda', non_blocking=True)
        logits = model(X)
        loss += criterion(logits, Y).item()
        pred = logits.argmax(dim = 1)
        correct += pred.eq(Y).sum()

        # Display Progres Bar. 
        # Reference - https://stackoverflow.com/questions/46141302/how-to-make-a-still-progress-in-python/46141777        
        batch += 1
        completed = math.floor(batch * dataloader.batch_size / len(dataloader.dataset) * 50)
        print('\r' + 'Validation: ' + '▮' * completed + '▯' * (50-completed) + str(completed*2) + '%', end='')
    
    print('\r', end='')        
        
    loss = loss / float(len(dataloader.dataset))
    accuracy = float(correct) / float(len(dataloader.dataset)) * 100
    
    return loss, accuracy
	
def test(model, dataloader):
    """ Infers output of given trained model for given test data. """
    loss = 0.0
    correct = 0.0
    accuracy = 0.0

    model.eval()
    
    for X, Y in dataloader:
        if gpu:
            X = X.to('cuda', non_blocking=True)
            Y = Y.to('cuda', non_blocking=True)
        logits = model(X)
        loss += criterion(logits, Y).item()
        pred = logits.argmax(dim = 1)
        correct += pred.eq(Y).sum()
        
    loss = loss / float(len(dataloader_test.dataset))
    accuracy = float(correct) / float(len(dataloader_test.dataset)) * 100
    return pred, loss, accuracy

def fit(dataloader_tr, dataloader_val, model, criterion, optimizer, max_epoch = 100, scheduler = None, filename = None, early_stop = True, patience = 10, resume = False):
    """
        Function to train and validate model for given epochs. It calls train and validate functions.
        
        Parameters: 
            dataloader_tr : data loader for training dataset.
            dataloader_val : dataloader for validation dataset.
            model : Instance of a Model. which is to be trained.
            criterion : criterion or loss function
            optimizer : Instance of Optimizer.
            max_epoch : Maximum number of epochs to train model
            scheduler : learning rate scheduler to change value of learning rate while model is trained.
            filename : Filename to save the best model during training. Function will save model with lowest validation loss, so that best model can be retrieved after training.
                       If resume = True, filename will be used to load previously saved model and resume the training.
            early_stop : If True, training will be stopped when validation_loss doesnt improve for epochs specified by patience. Recommended to prevent overfitting.
            patience : number of epochs to wait for early_stopping.
            resume : If True, model specified by filename will be loaded and training will be resumed for loaded model.
        Returns history of Training and Validation loss, and Training and Validation Accuracy.
    """
    start_epoch = 0
    early_stop_counter = 0
    min_loss_val = 1e10    
    loss_tr_hist = []
    loss_val_hist = []
    accuracy_tr_hist = []
    accuracy_val_hist = []

    if resume == True:
        if filename is None:
            print('Please Provide File Name to load model')
            return
        start_epoch, model, optimizer, scheduler, early_stop_counter, loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist = load_model(filename, model, optimizer, scheduler, mode = 'train')
        
        
    for epoch in range(start_epoch+1, max_epoch + 1):
        t0 = time.time()

        loss_tr, accuracy_tr = train(dataloader_tr, model, optimizer, criterion)
        loss_tr_hist.append(loss_tr)
        accuracy_tr_hist.append(accuracy_tr)

        loss_val, accuracy_val = validate(dataloader_val, model, criterion)
        loss_val_hist.append(loss_val)
        accuracy_val_hist.append(accuracy_val)

        if scheduler is not None:
            scheduler.step(loss_val)

        early_stop_counter += 1
        if loss_val < min_loss_val:
            if filename is not None:
                save_model(filename, model, optimizer, scheduler, epoch, loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist, early_stop_counter)
            min_loss_val = loss_val
            early_stop_counter = 0
        
        print("[{0:3d} / {1:3d}]  |  Loss_Tr: {2:7.4f}  |  Loss_Val: {3:7.4f}  |  Acc_Tr: {4:7.4f}  |  Acc_Val: {5:7.4f}  |  Time taken: {6:7.4f}s  |  {7}".format(epoch, max_epoch, loss_tr, loss_val, accuracy_tr, accuracy_val, time.time() - t0, "Best Model" if early_stop_counter == 0 else ""))
        
        if early_stop == True and early_stop_counter > patience:
            print('\nEarly Stopping ... !')
            break
    return loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist

def plot(loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist):
    """ Plots training loss vs validation loss and training accuracy vs validation accuracy graphs. """
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(10)

    plt.subplot(121)
    plt.plot(loss_tr_hist)
    plt.plot(loss_val_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(('Training', 'Validation'))

    plt.subplot(122)
    plt.plot(accuracy_tr_hist)
    plt.plot(accuracy_val_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(('Training', 'Validation'))
    plt.show()

def extract_features(model, dataloader):
    ''' This function extracts features of images from pretrained model.'''
    # Remove last classification layer of the model to get feature representation from penultimate layer.
    model_layers = list(model.children())
    del model_layers[-1]
    feature_extractor = nn.Sequential(*model_layers)

    # Mapping numpy array to disk so that if size of array exceeds RAM size, it will utilize disk.
    features_file='./image_features_temp.npy'
    image_features = np.memmap(features_file, dtype='float64', mode='w+', shape=(len(dataloader), dataloader.batch_size, 512))
    
    # Evaluation mode of model
    feature_extractor.eval()

    # Extract features of each image.
    for batch_idx, images in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc='Extracting features'):
        image_features[batch_idx] = feature_extractor(images).view(64,-1).detach().numpy()    
    
    image_features = image_features.reshape(-1,512)
    image_features = image_features[:len(dataloader)*dataloader.batch_size][:]
    
    return image_features
	
def plot_roc(far, tar, auc, split_n = None):
    ''' Plots ROC curve for given FAR and TAR values.'''
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    plt.figure()
    plt.plot(far, tar, color = 'darkgreen', lw = 2, label = 'ROC curve (area = %0.2f)' % auc)
    plt.xlim([0.0001, 1.0])
    plt.ylim([0.0, 1.05])
    plt.semilogx(far)
    plt.xlabel('False Accept Rate')
    plt.ylabel('True Accept Rate')
    plt.title('ROC for split ' + str(split_n))
    plt.legend(loc = "lower right")
    plt.grid(b=True, which='major', axis='both')
    plt.savefig('ROC_Curve_Split' + str(split_n) + '.png', dpi = 100)
    #plt.show()
	
def get_template_pairs(protocol_dir, img_dir, split_n):
    pair_file = 'verify_comparisons_'+str(split_n)+'.csv'
    pair_file = os.path.join(protocol_dir, 'split' + str(split_n), pair_file)
    pairs_df = pd.read_csv(pair_file, header = 0, names = ['TEMPLATE_ID1', 'TEMPLATE_ID2'])
    
    return pairs_df
	
def get_metadata(protocol_dir, img_dir, split_n):
    # Define Files for metadata and template pairs
    metadata_file = 'verify_metadata_'+str(split_n)+'.csv'
    metadata_file = os.path.join(protocol_dir, 'split' + str(split_n), metadata_file)
    
    # Read files in pandas dataframe
    metadata_df = pd.read_csv(metadata_file, usecols = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE'])
    metadata_df['FILE'] = metadata_df['FILE'] = img_dir + '/' + metadata_df['SUBJECT_ID'].map(str) + '/' + metadata_df['FILE']

    # Remove rows for missing files.
    metadata_df = metadata_df[:][metadata_df['FILE'].apply(lambda x:os.path.isfile(x))]
    
    return metadata_df