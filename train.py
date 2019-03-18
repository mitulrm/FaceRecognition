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
import argparse

from utils import *

def main(args):

	img_dir = os.path.expanduser(args.img_dir) if args.img_dir is not None else os.getcwd()
	protocol_dir = os.path.expanduser(args.protocol_dir) if args.protocol_dir is not None else os.getcwd()
	model_dir = os.path.expanduser(args.model_dir) if args.model_dir is not None else os.getcwd()
	workers = args.workers if args.workers is not None else 0
	batch_size = args.batch_size if args.batch_size is not None else 64
	split_n = args.split_n

	transform2 = transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

	# Check if GPU is available
	gpu = torch.cuda.is_available()
	if gpu:
		print('GPU is available.')
		pin_memory = True
	else:
		print('GPU not available. Executing on CPU.')
		pin_memory = False

	dataset = Dataset_Train(img_dir, protocol_dir, split_n = split_n, transform=transform)
	labels = dataset.get_labels()

	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
	idx_tr, idx_val = next(sss.split(labels, labels))

	dataset_tr = torch.utils.data.Subset(dataset, idx_tr)
	dataset_val = torch.utils.data.Subset(dataset, idx_val)

	dataloader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory = pin_memory)
	dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory = pin_memory)

	print('Training Dataset Length: ' + str(len(dataset_tr)))
	print('Validation Dataset Length: ' + str(len(dataset_val)))
	print('No of Labels: ' + str(len(set(labels))))

	model = models.resnet18(pretrained = True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, len(set(labels)))
	if gpu:
		model.cuda()

	# Initialize hyper parameters and functions required for training
	n_epochs = 100
	lr = 0.075

	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.001, nesterov=True)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=False, threshold=0.0005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
	criterion = nn.CrossEntropyLoss()
	start_epoch = 0
	model_filename = 'best_model_resnet18_'+str(split_n)+'.pth'

	# Train the model.
	loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist = fit(dataloader_tr, dataloader_val, model, criterion, optimizer, scheduler = scheduler, filename = model_filename, patience = 8, resume = False)

def parse_arguments(argv):
	parser = argparse.ArgumentParser(description='IJB-A dataset Training.')

	parser.add_argument('--img_dir', type=str, help='Path to the data directory containing aligned face images.')
	parser.add_argument('--protocol_dir', type=str, help='Path to the data directory containing csv files for metadata and template pairs.')
	parser.add_argument('--model_dir', type=str, help='Path to directory to save trained models.')	
	parser.add_argument('--workers', type=int, help='Number of workers for data loader.')
	parser.add_argument('--batch_size', type=int, help='Number of batches for data loader to divide data into batches.')
	parser.add_argument('--split_n', type=int, help='Split NUmber for which model is to be trained.')

	return parser.parse_args()

if __name__ == "__main__":
	main(parse_arguments(sys.argv[1:]))